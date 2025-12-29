#!/usr/bin/env python3
"""
使用 COCO 格式实例分割标注训练 Mask2Former。
相较于 train_m2f.py，本脚本直接读取 instances_*.json，构造实例掩码 supervision，
并在训练结束后导出 COCO 官方指标与可视化结果。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAS.mask2former.utils import (
    _as_dtype,
    build_mask2former_config,
    resolve_interaction_indices,
)  # noqa: E402
from MAS.train_m2f import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    _unwrap,
)  # noqa: E402
from MAS.data.coco_instance_data import (  # noqa: E402
    build_coco_instance_datasets_from_cfg,
    coco_evaluate_instances,
    coco_instance_collate_fn,
    visualize_instance_predictions,
)
from MAS.data.data import _build_dataloader  # noqa: E402
from MAS.eval_inference import (  # noqa: E402
    _resolve_eval_inference_params,
    _run_full_image_inference,
)
from MAS.log import init_training_logger  # noqa: E402
from MAS.model.backbone_utils import build_backbone  # noqa: E402
from MAS.utils import _setup_device_and_distributed, load_config  # noqa: E402
from MAS.mask2former.model import DinoMask2Former  # noqa: E402
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
    DINOv3_Adapter,
)  # noqa: E402
from MAS.trainers.coco_instance import (
    train_one_epoch,
    evaluate,
    run_coco_inference,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 COCO 标注训练 Mask2Former 实例分割头"
    )
    default_cfg = Path(__file__).resolve().with_name("config_m2f_segmentation.yaml")
    parser.add_argument(
        "--config", type=str, default=str(default_cfg), help="YAML 配置文件路径"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="可选：恢复训练的 checkpoint"
    )
    return parser.parse_args()


def _build_class_palette(num_classes: int) -> list[tuple[int, int, int]]:
    hsv = np.linspace(0, 179, max(1, num_classes), dtype=np.uint8)
    palette: list[tuple[int, int, int]] = []
    for h in hsv:
        bgr = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
    return palette


def _colorize_mask(mask: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb
    return color


def export_semantic_predictions(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    output_dir: Path,
    num_classes: int,
    eval_params: dict,
) -> None:
    palette = _build_class_palette(num_classes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
                if num_classes > 0:
                    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
                color_mask = _colorize_mask(pred_mask, palette)
                if file_names:
                    stem = str(file_names[b])
                else:
                    if isinstance(sample_indices, torch.Tensor):
                        stem = f"sample_{int(sample_indices[b].item())}"
                    else:
                        stem = f"sample_{int(sample_indices[b])}"
                out_path = output_dir / f"{stem}.png"
                Image.fromarray(color_mask, mode="RGB").save(out_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {}) or {}
    dataset_cfg = cfg.get("coco_dataset", {}) or {}
    if not dataset_cfg:
        raise ValueError("配置文件中缺少 coco_dataset 字段")
    weights_cfg = cfg.get("weights", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_inference_params = _resolve_eval_inference_params(dataset_cfg, eval_cfg)
    pred_cfg = dataset_cfg.get("prediction", {}) or {}
    vis_cfg = dataset_cfg.get("visualization", {}) or {}

    device, rank, world_size, local_rank, distributed = _setup_device_and_distributed()
    mask2former_config = build_mask2former_config(cfg, dataset_key="coco_dataset")
    feature_order = cfg.get("mask2former", {}).get(
        "feature_order", ["1", "2", "3", "4"]
    )

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
    gt_dir_setting = eval_cfg.get("gt_dir")
    gt_dir_path = Path(gt_dir_setting).expanduser() if gt_dir_setting else None

    backbone = build_backbone(cfg, device)
    indices = resolve_interaction_indices(backbone, cfg)
    adapter = DINOv3_Adapter(backbone, interaction_indexes=indices).to(device)
    model = DinoMask2Former(adapter, mask2former_config, feature_order).to(device)
    if rank == 0:
        unwrapped = _unwrap(model)
        backbone_params = sum(
            p.numel() for p in unwrapped.feature_extractor.parameters()
        )
        head_params = sum(
            p.numel()
            for module in (
                unwrapped.pixel_decoder,
                unwrapped.transformer_module,
                unwrapped.class_predictor,
            )
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

    train_dataset, val_dataset = build_coco_instance_datasets_from_cfg(dataset_cfg)
    pin_memory = device.type == "cuda"
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        num_workers=train_cfg.get("num_workers", 8),
        shuffle=True,
        pin_memory=pin_memory,
        distributed=distributed,
        collate_fn=coco_instance_collate_fn,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", 1),
        num_workers=max(1, train_cfg.get("num_workers", 8) // 2),
        shuffle=False,
        pin_memory=pin_memory,
        distributed=distributed,
        collate_fn=coco_instance_collate_fn,
    )
    coco_eval_loader = _build_dataloader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", 1),
        num_workers=max(1, train_cfg.get("num_workers", 8) // 2),
        shuffle=False,
        pin_memory=pin_memory,
        distributed=False,
        collate_fn=coco_instance_collate_fn,
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
    scheduler, scheduler_interval = build_scheduler(
        optimizer, train_cfg, steps_per_epoch, epochs
    )

    autocast_dtype = _as_dtype(train_cfg.get("autocast_dtype", "bfloat16"), device.type)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = amp.GradScaler(
        amp_device, enabled=(device.type == "cuda" and autocast_dtype == torch.float16)
    )

    eval_interval = max(1, int(train_cfg.get("eval_interval", 1)))
    save_after_epoch = max(1, int(train_cfg.get("save_after_epoch", 90)))

    start_epoch = 0
    global_step = 0
    best_map = 0.0

    num_classes = int(dataset_cfg.get("num_classes", mask2former_config.num_labels))
    ignore_index = int(dataset_cfg.get("ignore_index", mask2former_config.ignore_value))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip", train_cfg.get("grad_clip_norm", 0.0)))

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        if (
            distributed
            and hasattr(train_loader, "sampler")
            and train_loader.sampler is not None
        ):
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
        eval_loss: float | None = None
        coco_metrics: dict[str, float] | None = None
        if should_eval:
            eval_loss = evaluate(
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
                eval_model = _unwrap(model)
                coco_predictions = run_coco_inference(
                    model=eval_model,
                    loader=coco_eval_loader,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_queries=int(pred_cfg.get("max_queries", 100)),
                )
                coco_metrics = coco_evaluate_instances(
                    coco_predictions,
                    dataset_cfg["val_annotations"],
                    output_dir=None,
                )
        if rank == 0:
            scalars = {
                "train/loss": train_loss,
                "optimizer/lr": optimizer.param_groups[0]["lr"],
            }
            for k, v in train_metrics.items():
                scalars[f"train/{k}"] = v
            if eval_loss is not None:
                scalars["eval/loss"] = eval_loss
            if coco_metrics is not None:
                for key, value in coco_metrics.items():
                    scalars[f"eval/{key}"] = value
            logger.log_scalars(global_step, scalars)
            if eval_loss is not None:
                metric_msg = ""
                if coco_metrics is not None:
                    metric_msg = " ".join(
                        f"{k}={v:.4f}" for k, v in coco_metrics.items()
                    )
                logger.log_message(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} {metric_msg}".strip()
                )
                if coco_metrics is not None:
                    current_map = coco_metrics.get("mAP", 0.0)
                    if current_map > best_map:
                        best_map = current_map
                        if (epoch + 1) >= save_after_epoch:
                            save_checkpoint(
                                best_ckpt_path, model, epoch, global_step, best_map
                            )
                            logger.log_message(
                                f"新的最佳 mAP={best_map:.4f}，已更新最佳 checkpoint (epoch {epoch})."
                            )
                        else:
                            logger.log_message(f"新的最佳 mAP={best_map:.4f}")
            else:
                logger.log_message(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f} (跳过评估 interval={eval_interval})"
                )
        last_epoch = epoch
        if distributed:
            dist.barrier()

    if rank == 0:
        save_checkpoint(
            last_ckpt_path,
            model,
            last_epoch if last_epoch >= 0 else epochs - 1,
            global_step,
            best_map,
        )
        logger.log_message(f"最后一个 epoch 的权重已保存至: {last_ckpt_path}")

    pred_dir_path = Path(pred_dir_setting)
    if not pred_dir_path.is_absolute():
        pred_dir_path = logger.run_dir / pred_dir_path
    export_semantic_predictions(
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
        eval_model = _unwrap(model)
        logger.log_message(f"训练完成，语义预测 PNG 已保存到: {pred_dir_path}")
        coco_predictions = run_coco_inference(
            model=eval_model,
            loader=coco_eval_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            max_queries=int(pred_cfg.get("max_queries", 100)),
        )
        gt_annotations = str(gt_dir_path) if gt_dir_path is not None else dataset_cfg["val_annotations"]
        coco_metrics = coco_evaluate_instances(
            coco_predictions,
            gt_annotations,
            output_dir=pred_dir_path,
        )
        coco_metric_str = " ".join(f"{k}:{v:.4f}" for k, v in coco_metrics.items())
        logger.log_message(f"[COCO Evaluation] {coco_metric_str}")

        vis_dir_setting = vis_cfg.get("output_dir", "coco_vis")
        vis_dir_path = Path(vis_dir_setting)
        if not vis_dir_path.is_absolute():
            vis_dir_path = logger.run_dir / vis_dir_path
        visualize_instance_predictions(
            val_dataset,
            coco_predictions,
            vis_dir_path,
            score_thresh=float(vis_cfg.get("score_thresh", 0.5)),
        )
        logger.log_message(f"COCO 实例可视化结果保存在: {vis_dir_path}")


if __name__ == "__main__":
    main()
