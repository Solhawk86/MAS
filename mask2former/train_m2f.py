#!/usr/bin/env python3
"""
在冻结（可注入 LoRA）的 DINOv3 主干上训练 HuggingFace Mask2Former 分割头。
所有路径、超参与 Mask2Former 配置均来自 config_m2f_segmentation.yaml，可通过 torchrun 单节点多卡运行：

    torchrun --nproc_per_node=4 MAS/train_m2f.py --config MAS/config_m2f_segmentation.yaml
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
import sys

from MAS.mask2former.utils import _as_dtype

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import amp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from PIL import Image

from MAS.data.data import build_segmentation_image_dataloaders_from_cfg
from MAS.eval_inference import _resolve_eval_inference_params, _run_full_image_inference
from MAS.log import init_training_logger
from MAS.eval_metrics import evaluate_prediction_directory, cal_iou
from MAS.model.backbone_utils import build_backbone
from MAS.utils import _distributed_mean, _setup_device_and_distributed, load_config
from MAS.mask2former.utils import *
from MAS.mask2former.model import DinoMask2Former
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HF Mask2Former head on top of frozen DINOv3")
    default_cfg = Path(__file__).resolve().with_name("config_m2f_segmentation.yaml")
    parser.add_argument("--config", type=str, default=str(default_cfg), help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="可选：恢复训练的 checkpoint")
    return parser.parse_args()

def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module

def _build_targets(
    masks: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for mask in masks:
        unique_vals = torch.unique(mask)
        bin_masks: list[torch.Tensor] = []
        cls_ids: list[int] = []
        for value in unique_vals.tolist():
            if value == ignore_index or value < 0 or value >= num_classes:
                continue
            binary = (mask == value).float()
            if torch.count_nonzero(binary) == 0:
                continue
            bin_masks.append(binary)
            cls_ids.append(int(value))
        if bin_masks:
            mask_labels.append(torch.stack(bin_masks))
            class_labels.append(torch.as_tensor(cls_ids, device=mask.device, dtype=torch.long))
        else:
            h, w = mask.shape[-2:]
            mask_labels.append(torch.zeros((0, h, w), device=mask.device, dtype=torch.float32))
            class_labels.append(torch.zeros((0,), device=mask.device, dtype=torch.long))
    return mask_labels, class_labels


def _resize_mask_to_size(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2:] == (height, width):
        return mask
    resized = (
        F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )
    return resized.to(mask.dtype)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler | None,
    scheduler,
    scheduler_interval: str,
    device: torch.device,
    epoch: int,
    global_step: int,
    logger,
    world_size: int,
    num_classes: int,
    ignore_index: int,
    grad_clip: float,
    log_interval: int,
    autocast_dtype: torch.dtype,
    grad_accum_steps: int,
) -> tuple[int, float, dict[str, float]]:
    model.train()
    total_steps = len(loader)
    loss_sum = 0.0
    metrics: defaultdict[str, float] = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for step, batch in enumerate(loader):
        images, masks, *_ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        mask_labels, class_labels = _build_targets(masks, num_classes=num_classes, ignore_index=ignore_index)
        with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
            outputs = model(images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs["loss"]
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and scheduler_interval == "step":
                scheduler.step()
            global_step += 1
        loss_sum += float(outputs["loss"].item())
        for k, v in outputs["loss_dict"].items():
            metrics[k] += float(v.item())
        if logger is not None and log_interval > 0 and (step + 1) % log_interval == 0:
            logger.log_message(f"[Epoch {epoch} Step {step+1}/{total_steps}] loss={outputs['loss'].item():.4f}")
    mean_loss = _distributed_mean(torch.tensor([loss_sum / max(1, total_steps)], device=device), world_size)
    avg_metrics = {
        k: float(_distributed_mean(torch.tensor([v / max(1, total_steps)], device=device), world_size).item())
        for k, v in metrics.items()
    }
    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()
    return global_step, float(mean_loss.item()), avg_metrics


def evaluate(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    autocast_dtype: torch.dtype,
    world_size: int,
    eval_params: dict,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    count = 0
    iou_metric = cal_iou()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch in loader:
            images, masks, _indices, orig_sizes, *_ = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            mask_labels, class_labels = _build_targets(masks, num_classes=num_classes, ignore_index=ignore_index)
            with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(images, mask_labels=mask_labels, class_labels=class_labels)
            loss_sum += float(outputs["loss"].item())
            count += 1
            batch_size = masks.shape[0]
            for b in range(batch_size):
                target_mask = masks[b]
                target_h = int(orig_sizes[b][0].item()) if orig_sizes is not None else int(target_mask.shape[-2])
                target_w = int(orig_sizes[b][1].item()) if orig_sizes is not None else int(target_mask.shape[-1])
                if target_h <= 0 or target_w <= 0:
                    target_h, target_w = target_mask.shape[-2:]
                target_mask_resized = _resize_mask_to_size(target_mask, target_h, target_w)
                logits = _run_full_image_inference(
                    model,
                    images[b],
                    eval_params=eval_params,
                    num_classes=num_classes,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    target_size=(target_h, target_w),
                )
                pred_mask = logits.argmax(dim=1).squeeze(0)
                valid = target_mask_resized != ignore_index
                pred_clean = torch.where(valid, pred_mask.float(), torch.zeros_like(pred_mask, dtype=torch.float32))
                target_clean = torch.where(
                    valid,
                    target_mask_resized.float(),
                    torch.zeros_like(pred_mask, dtype=torch.float32),
                )
                iou_metric.update(pred_clean.detach().cpu().numpy(), target_clean.detach().cpu().numpy())
    mean_loss = _distributed_mean(torch.tensor([loss_sum / max(1, count)], device=device), world_size)
    local_miou = torch.tensor([float(iou_metric.show())], device=device)
    miou = float(_distributed_mean(local_miou, world_size).item())
    return float(mean_loss.item()), miou


def export_predictions(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    output_dir: Path,
    num_classes: int,
    eval_params: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, _masks, sample_indices, orig_sizes, *rest = batch
            file_names = rest[0] if rest else None
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            for b in range(batch_size):
                target_h = int(orig_sizes[b][0].item())
                target_w = int(orig_sizes[b][1].item())
                if target_h <= 0 or target_w <= 0:
                    target_h, target_w = images.shape[-2:]
                logits = _run_full_image_inference(
                    model,
                    images[b],
                    eval_params=eval_params,
                    num_classes=num_classes,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    target_size=(target_h, target_w),
                )
                pred_mask = logits.argmax(dim=1).squeeze(0).float()
                mask_tensor = torch.clamp(pred_mask, 0.0, 1.0)
                mask_tensor = (mask_tensor * 255).to(torch.uint8).cpu().numpy()
                if file_names:
                    stem = str(file_names[b])
                else:
                    if isinstance(sample_indices, torch.Tensor):
                        stem = f"sample_{int(sample_indices[b].item())}"
                    else:
                        stem = f"sample_{int(sample_indices[b])}"
                out_path = output_dir / f"{stem}.png"
                Image.fromarray(mask_tensor, mode="L").save(out_path)


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("未找到需要训练的参数")
    return AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict, steps_per_epoch: int, total_epochs: int):
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "")).strip().lower()
    if sched_type in {"", "none", "null"}:
        sched_type = ""
    interval = sched_cfg.get("interval", "epoch")
    if not sched_type:
        return None, "step"
    if sched_type == "fractional_step_decay":
        fractions = sched_cfg.get("fractions", [0.9, 0.95])
        gamma = float(sched_cfg.get("decay_factor", 0.1))
        total_steps = max(1, steps_per_epoch * total_epochs)
        milestones: list[int] = []
        for frac in fractions:
            if isinstance(frac, (int, float)):
                if 0 < frac < 1:
                    milestones.append(max(1, int(total_steps * frac)))
                else:
                    milestones.append(int(frac))
        milestones = sorted(set(milestones))

        def lr_lambda(step: int) -> float:
            factor = 1.0
            for milestone in milestones:
                if step >= milestone:
                    factor *= gamma
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "step"
    if sched_type == "cosine":
        t_max = total_epochs if interval == "epoch" else max(1, steps_per_epoch * total_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(sched_cfg.get("eta_min", 0.0)),
        )
        return scheduler, interval
    if sched_type == "multistep":
        milestones = sched_cfg.get("decay_milestones", [0.9, 0.95])
        if all(0 < m < 1 for m in milestones):
            milestones = [int(total_epochs * m) for m in milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(sched_cfg.get("decay_factor", 0.1)),
        )
        return scheduler, "epoch"
    raise ValueError(f"未知的 scheduler 类型: {sched_type}")

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    best_miou: float,
) -> None:
    module = _unwrap(model)
    state = {
        "model": module.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_miou": best_miou,
    }
    torch.save(state, path)

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}
    weights_cfg = cfg.get("weights", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_inference_params = _resolve_eval_inference_params(dataset_cfg, eval_cfg)

    device, rank, world_size, local_rank, distributed = _setup_device_and_distributed()
    mask2former_config = build_mask2former_config(cfg)
    feature_order = cfg.get("mask2former", {}).get("feature_order", ["1", "2", "3", "4"])

    logger = init_training_logger(
        rank=rank,
        distributed=distributed,
        tensorboard_enabled=bool(train_cfg.get("tensorboard_enabled", True)),
        tensorboard_subdir=train_cfg.get("tensorboard_log_dir"),
        checkpoint_subdir=train_cfg.get("checkpoint_dir"),
        weights_root=weights_cfg.get("storage_dir"),
        local_weights_root=None,
    )
    if rank == 0:
        logger.log_config(cfg)
    best_ckpt_path = logger.checkpoint_dir / "best.pt"
    last_ckpt_path = logger.checkpoint_dir / "last.pt"

    pred_dir_setting = eval_cfg.get("pred_dir", "predictions")
    gt_dir_setting = eval_cfg.get("gt_dir") or dataset_cfg.get("val_masks")
    if gt_dir_setting is None:
        raise ValueError("请在 config 中设置 evaluation.gt_dir 或 dataset.val_masks")
    gt_dir_path = Path(gt_dir_setting).expanduser()

    backbone = build_backbone(cfg, device)
    indices = resolve_interaction_indices(backbone, cfg)
    adapter = DINOv3_Adapter(backbone, interaction_indexes=indices).to(device)
    model = DinoMask2Former(adapter, mask2former_config, feature_order).to(device)
    if rank == 0:
        unwrapped = _unwrap(model)
        backbone_params = sum(p.numel() for p in unwrapped.feature_extractor.parameters())
        head_params = sum(
            p.numel()
            for module in (unwrapped.pixel_decoder, unwrapped.transformer_module, unwrapped.class_predictor)
            for p in module.parameters()
        )
        total_params = backbone_params + head_params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log_debug(
            "Model parameters -> backbone: {:,} | head: {:,} | total: {:,} | trainable: {:,}".format(
                backbone_params, head_params, total_params, trainable_params
            )
        )
        logger.log_module_grad_status(unwrapped, label="model")

    train_loader, val_loader = build_segmentation_image_dataloaders_from_cfg(
        dataset_cfg,
        train_cfg,
        device.type,
        distributed=distributed,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(model, train_cfg)
    epochs = int(train_cfg.get("epochs", 30))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    scheduler, scheduler_interval = build_scheduler(optimizer, train_cfg, steps_per_epoch, epochs)

    autocast_dtype = _as_dtype(train_cfg.get("autocast_dtype", "bfloat16"), device.type)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = amp.GradScaler(amp_device, enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

    eval_interval = max(1, int(train_cfg.get("eval_interval", 1)))
    save_after_epoch = max(1, int(train_cfg.get("save_after_epoch", 90)))

    start_epoch = 0
    global_step = 0
    best_miou = 0.0

    num_classes = int(dataset_cfg.get("num_classes", mask2former_config.num_labels))
    ignore_index = int(dataset_cfg.get("ignore_index", mask2former_config.ignore_value))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip", train_cfg.get("grad_clip_norm", 0.0)))

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        if distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        global_step, train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            device=device,
            epoch=epoch,
            global_step=global_step,
            logger=logger,
            world_size=world_size,
            num_classes=num_classes,
            ignore_index=ignore_index,
            grad_clip=grad_clip,
            log_interval=log_interval,
            autocast_dtype=autocast_dtype,
            grad_accum_steps=grad_accum_steps,
        )
        should_eval = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == epochs)
        eval_results: tuple[float, float] | None = None
        if should_eval:
            eval_results = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
                autocast_dtype=autocast_dtype,
                world_size=world_size,
                eval_params=eval_inference_params,
            )
        if rank == 0:
            scalars = {
                "train/loss": train_loss,
                "optimizer/lr": optimizer.param_groups[0]["lr"],
            }
            for k, v in train_metrics.items():
                scalars[f"train/{k}"] = v
            if eval_results is not None:
                eval_loss, miou = eval_results
                scalars["eval/loss"] = eval_loss
                scalars["eval/miou"] = miou
            logger.log_scalars(global_step, scalars)
            if eval_results is not None:
                eval_loss, miou = eval_results
                logger.log_message(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} mIoU={miou:.4f}"
                )
            else:
                logger.log_message(f"[Epoch {epoch}] train_loss={train_loss:.4f} (跳过评估 interval={eval_interval})")
            if eval_results is not None:
                _, miou = eval_results
                if miou > best_miou:
                    best_miou = miou
                    if (epoch + 1) >= save_after_epoch:
                        save_checkpoint(best_ckpt_path, model, epoch, global_step, best_miou)
                        logger.log_message(
                            f"新的最佳 mIoU={best_miou:.4f}，已更新最佳 checkpoint (epoch {epoch})."
                        )
                    else:
                        logger.log_message(
                            f"新的最佳 mIoU={best_miou:.4f}"
                        )
        last_epoch = epoch
        if distributed:
            dist.barrier()

    if rank == 0:
        save_checkpoint(last_ckpt_path, model, last_epoch if last_epoch >= 0 else epochs - 1, global_step, best_miou)
        logger.log_message(f"最后一个 epoch 的权重已保存至: {last_ckpt_path}")

    pred_dir_path = Path(pred_dir_setting)
    if not pred_dir_path.is_absolute():
        pred_dir_path = logger.run_dir / pred_dir_path
    export_predictions(
        model=model,
        loader=val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        output_dir=pred_dir_path,
        num_classes=num_classes,
        eval_params=eval_inference_params,
    )
    if distributed:
        dist.barrier()
    if rank == 0:
        logger.log_message(f"训练完成，预测结果已保存到: {pred_dir_path}")
        try:
            metrics = evaluate_prediction_directory(pred_dir_path, gt_dir_path)
            metric_str = " ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            logger.log_message(f"[Final Evaluation] {metric_str}")
        except Exception as exc:
            logger.log_message(f"评估预测结果失败: {exc}")


if __name__ == "__main__":
    main()
