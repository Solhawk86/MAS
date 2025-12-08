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
from typing import Iterable, Sequence
import sys

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

from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerTransformerModule,
)
from MAS.hf_pixel_decoder_custom import DinoCompatibleMask2FormerPixelDecoder

from MAS.data.data import build_segmentation_image_dataloaders_from_cfg
from MAS.log import init_training_logger
from MAS.eval_metrics import evaluate_prediction_directory, cal_iou
from MAS.model.backbone_utils import build_backbone
from MAS.utils import _distributed_mean, _setup_device_and_distributed, load_config
from dinov3.eval.segmentation.models import BackboneLayersSet
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HF Mask2Former head on top of frozen DINOv3")
    default_cfg = Path(__file__).resolve().with_name("config_m2f_segmentation.yaml")
    parser.add_argument("--config", type=str, default=str(default_cfg), help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="可选：恢复训练的 checkpoint")
    return parser.parse_args()


def build_mask2former_config(cfg: dict) -> Mask2FormerConfig:
    dataset_cfg = cfg.get("dataset", {}) or {}
    mask2former_cfg = cfg.get("mask2former", {}) or {}
    num_classes = int(dataset_cfg.get("num_classes", mask2former_cfg.get("num_classes", 2)))
    ignore_value = int(dataset_cfg.get("ignore_index", mask2former_cfg.get("ignore_value", -1)))
    hidden_dim = int(mask2former_cfg.get("hidden_dim", 256))
    ffn_dim = int(mask2former_cfg.get("ffn_dim", 2048))
    return Mask2FormerConfig(
        feature_size=int(mask2former_cfg.get("feature_size", hidden_dim)),
        mask_feature_size=int(mask2former_cfg.get("mask_feature_size", hidden_dim)),
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=ffn_dim,
        encoder_layers=int(mask2former_cfg.get("encoder_layers", 6)),
        decoder_layers=int(mask2former_cfg.get("decoder_layers", 9)),
        num_attention_heads=int(mask2former_cfg.get("num_heads", 8)),
        dim_feedforward=ffn_dim,
        dropout=float(mask2former_cfg.get("dropout", 0)),
        pre_norm=bool(mask2former_cfg.get("pre_norm", False)),
        enforce_input_projection=bool(mask2former_cfg.get("enforce_input_projection", False)),
        common_stride=int(mask2former_cfg.get("common_stride", 4)),
        num_queries=int(mask2former_cfg.get("num_queries", 100)),
        no_object_weight=float(mask2former_cfg.get("no_object_weight", 0.1)),
        class_weight=float(mask2former_cfg.get("class_weight", 2.0)),
        mask_weight=float(mask2former_cfg.get("mask_weight", 5.0)),
        dice_weight=float(mask2former_cfg.get("dice_weight", 5.0)),
        train_num_points=int(mask2former_cfg.get("train_num_points", 11250)),
        oversample_ratio=float(mask2former_cfg.get("oversample_ratio", 3.0)),
        importance_sample_ratio=float(mask2former_cfg.get("importance_sample_ratio", 0.75)),
        feature_strides=list(mask2former_cfg.get("feature_strides", [4, 8, 16, 32])),
        num_labels=num_classes,
        ignore_value=ignore_value,
        use_auxiliary_loss=True,
        output_auxiliary_logits=True,
    )


def resolve_interaction_indices(backbone: torch.nn.Module, cfg: dict) -> list[int]:
    spec = (
        cfg.get("model", {}).get("backbone_out_layers")
        or cfg.get("backbone", {}).get("backbone_out_layers")
        or BackboneLayersSet.FOUR_EVEN_INTERVALS.name
    )
    n_blocks = getattr(backbone, "n_blocks", None)
    if isinstance(spec, str):
        try:
            layerset = BackboneLayersSet[spec]
        except KeyError:
            values = [int(part.strip()) for part in spec.split(",") if part.strip()]
            if not values:
                raise ValueError(f"无法解析 backbone_out_layers = {spec}")
            return values
        if n_blocks is None:
            raise ValueError("backbone 缺少 n_blocks，无法根据枚举推断层索引")
        if layerset == BackboneLayersSet.LAST:
            return [n_blocks - 1]
        if layerset == BackboneLayersSet.FOUR_LAST:
            return list(range(max(0, n_blocks - 4), n_blocks))
        if layerset == BackboneLayersSet.FOUR_EVEN_INTERVALS:
            if n_blocks == 24:
                return [4, 11, 17, 23]
            step = max(1, n_blocks // 4)
            return [min(n_blocks - 1, step * i - 1) for i in range(1, 5)]
        raise ValueError(f"暂不支持的 BackboneLayersSet: {layerset}")
    if isinstance(spec, int):
        if n_blocks is None:
            raise ValueError("整数模式需要 backbone.n_blocks")
        count = max(1, spec)
        return list(range(max(0, n_blocks - count), n_blocks))
    if isinstance(spec, Iterable):
        values = [int(v) for v in spec]
        if not values:
            raise ValueError("backbone_out_layers 不能为空")
        return values
    raise TypeError(f"无法解析的 backbone_out_layers 类型: {type(spec)}")


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _distributed_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    clone = tensor.clone()
    dist.all_reduce(clone, op=dist.ReduceOp.SUM)
    return clone


def _as_dtype(name: str, device_type: str) -> torch.dtype:
    key = (name or "").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "half", "float16"}:
        if device_type != "cuda":
            raise ValueError("float16 自动混合精度仅支持 CUDA")
        return torch.float16
    if key in {"fp32", "float32", "f32", ""}:
        return torch.float32
    raise ValueError(f"未知的 autocast dtype: {name}")


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


def _semantic_from_logits(class_logits: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask_classes = class_logits.softmax(dim=-1)[..., :-1]
    mask_probs = mask_logits.sigmoid()
    logits = torch.einsum("bqc,bqhw->bchw", mask_classes, mask_probs)
    return logits.argmax(dim=1)


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


class DinoMask2Former(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, config: Mask2FormerConfig, feature_order: Sequence[str]):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_order = list(feature_order)
        backbone = getattr(feature_extractor, "backbone", feature_extractor)
        embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("无法从 DINOv3 主干中推断 embed_dim")
        self.pixel_decoder = DinoCompatibleMask2FormerPixelDecoder(
            config,
            feature_channels=[int(embed_dim)] * len(self.feature_order),
        )
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)
        self.class_predictor = torch.nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = Mask2FormerLoss(
            config=config,
            weight_dict={
                "loss_cross_entropy": config.class_weight,
                "loss_mask": config.mask_weight,
                "loss_dice": config.dice_weight,
            },
        )
        self.config = config
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)
        self._logged_feature_info = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ):
        with torch.no_grad():
            features_dict = self.feature_extractor(pixel_values)
        features = [features_dict[key] for key in self.feature_order if key in features_dict]
        if not self._logged_feature_info:
            info = ", ".join(f"{key}:{tuple(features_dict[key].shape)}" for key in self.feature_order if key in features_dict)
            print(f"[DINOv3 Adapter Features] {info}")
            self._logged_feature_info = True
        decoder_out = self.pixel_decoder(
            features,
            output_hidden_states=True,
            output_attentions=self.config.output_attentions,
            return_dict=True,
        )
        transformer_out = self.transformer_module(
            multi_scale_features=list(decoder_out.multi_scale_features),
            mask_features=decoder_out.mask_features,
            output_hidden_states=True,
            output_attentions=self.config.output_attentions,
        )
        class_queries = tuple(
            self.class_predictor(hidden_state.transpose(0, 1))
            for hidden_state in transformer_out.intermediate_hidden_states
        )
        masks_queries = transformer_out.masks_queries_logits
        aux = [
            {
                "class_queries_logits": cls,
                "masks_queries_logits": mask,
            }
            for cls, mask in zip(class_queries[:-1], masks_queries[:-1])
        ]
        loss = None
        loss_dict: dict[str, torch.Tensor] = {}
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.criterion(
                masks_queries_logits=masks_queries[-1],
                class_queries_logits=class_queries[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=aux,
            )
            loss = sum(loss_dict.values())
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "class_queries_logits": class_queries[-1],
            "masks_queries_logits": masks_queries[-1],
        }


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
            preds = _semantic_from_logits(outputs["class_queries_logits"], outputs["masks_queries_logits"])
            batch_size = preds.shape[0]
            for b in range(batch_size):
                pred_mask = preds[b]
                target_mask = masks[b]
                if orig_sizes is not None:
                    target_h = int(orig_sizes[b][0].item())
                    target_w = int(orig_sizes[b][1].item())
                    if target_h > 0 and target_w > 0:
                        pred_mask = _resize_mask_to_size(pred_mask, target_h, target_w)
                        target_mask = _resize_mask_to_size(target_mask, target_h, target_w)
                valid = target_mask != ignore_index
                pred_clean = torch.where(valid, pred_mask.float(), torch.zeros_like(pred_mask, dtype=torch.float32))
                target_clean = torch.where(valid, target_mask.float(), torch.zeros_like(target_mask, dtype=torch.float32))
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
            with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(images)
            preds = _semantic_from_logits(outputs["class_queries_logits"], outputs["masks_queries_logits"])
            batch_size = preds.shape[0]
            for b in range(batch_size):
                pred_mask = preds[b]
                target_h = int(orig_sizes[b][0].item())
                target_w = int(orig_sizes[b][1].item())
                if target_h > 0 and target_w > 0:
                    pred_mask = _resize_mask_to_size(pred_mask, target_h, target_w)
                mask_tensor = torch.clamp(pred_mask.float(), 0.0, 1.0)
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


def _extract_head_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    module = _unwrap(model)
    return {
        "pixel_decoder": module.pixel_decoder.state_dict(),
        "transformer_module": module.transformer_module.state_dict(),
        "class_predictor": module.class_predictor.state_dict(),
    }


def _load_head_state(model: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    module = _unwrap(model)
    if "pixel_decoder" in state:
        module.pixel_decoder.load_state_dict(state["pixel_decoder"], strict=False)
    if "transformer_module" in state:
        module.transformer_module.load_state_dict(state["transformer_module"], strict=False)
    if "class_predictor" in state:
        module.class_predictor.load_state_dict(state["class_predictor"], strict=False)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    best_miou: float,
) -> None:
    state = {
        "head": _extract_head_state(model),
        "epoch": epoch,
        "global_step": global_step,
        "best_miou": best_miou,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    head_state = ckpt.get("head") or ckpt.get("model")
    if head_state is not None:
        _load_head_state(model, head_state)
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), float(ckpt.get("best_miou", 0.0))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}
    weights_cfg = cfg.get("weights", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}

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

    resume = args.resume or train_cfg.get("resume")
    start_epoch = 0
    global_step = 0
    best_miou = 0.0
    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.is_file():
            start_epoch, global_step, best_miou = load_checkpoint(ckpt_path, model)
            if rank == 0:
                logger.log_message(f"已从 {ckpt_path} 恢复 (epoch={start_epoch}, step={global_step})")

    num_classes = int(dataset_cfg.get("num_classes", mask2former_config.num_labels))
    ignore_index = int(dataset_cfg.get("ignore_index", mask2former_config.ignore_value))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip", train_cfg.get("grad_clip_norm", 0.0)))

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
        eval_loss, miou = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            autocast_dtype=autocast_dtype,
            world_size=world_size,
        )
        if rank == 0:
            scalars = {
                "train/loss": train_loss,
                "eval/loss": eval_loss,
                "eval/miou": miou,
                "optimizer/lr": optimizer.param_groups[0]["lr"],
            }
            for k, v in train_metrics.items():
                scalars[f"train/{k}"] = v
            logger.log_scalars(global_step, scalars)
            logger.log_message(f"[Epoch {epoch}] train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} mIoU={miou:.4f}")
            improved = miou > best_miou
            if improved:
                best_miou = miou
                save_checkpoint(best_ckpt_path, model, epoch, global_step, best_miou)
                logger.log_message(f"新的最佳 mIoU={best_miou:.4f}，已更新最佳 checkpoint。")
            save_checkpoint(last_ckpt_path, model, epoch, global_step, best_miou)
        if distributed:
            dist.barrier()

    pred_dir_path = Path(pred_dir_setting)
    if not pred_dir_path.is_absolute():
        pred_dir_path = logger.run_dir / pred_dir_path
    export_predictions(
        model=model,
        loader=val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        output_dir=pred_dir_path,
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
