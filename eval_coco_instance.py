#!/usr/bin/env python3
"""Evaluate a trained Mask2Former checkpoint on a COCO-style dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAS.data.coco_instance_data import (  # noqa: E402
    build_coco_instance_datasets_from_cfg,
    coco_evaluate_instances,
    coco_instance_collate_fn,
    visualize_instance_predictions,
)
from MAS.data.data import _build_dataloader  # noqa: E402
from MAS.eval_inference import _resolve_eval_inference_params  # noqa: E402
from MAS.log import init_training_logger  # noqa: E402
from MAS.mask2former.model import DinoMask2Former  # noqa: E402
from MAS.mask2former.utils import (  # noqa: E402
    _as_dtype,
    build_mask2former_config,
    resolve_interaction_indices,
)
from MAS.model.backbone_utils import build_backbone  # noqa: E402
from MAS.train_coco_instance import export_semantic_predictions  # noqa: E402
from MAS.train_m2f import _unwrap  # noqa: E402
from MAS.trainers.coco_instance import run_coco_inference  # noqa: E402
from MAS.utils import _setup_device_and_distributed, load_config  # noqa: E402
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (  # noqa: E402
    DINOv3_Adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Mask2Former checkpoint")
    default_cfg = CURRENT_DIR / "config_m2f_segmentation.yaml"
    parser.add_argument("--config", type=str, default=str(default_cfg), help="YAML 配置文件")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/hong/weights/20251225/checkpoints/best.pt",
        help="要加载的 checkpoint 路径",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./eval_outputs",
        help="保存预测与可视化结果的目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg.get("coco_dataset", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    if not dataset_cfg:
        raise ValueError("配置文件中缺少 coco_dataset 字段")

    output_root = Path(args.output_root).expanduser()
    pred_dir = output_root / "semantic_png"
    vis_dir = output_root / "coco_vis"

    device, rank, world_size, local_rank, distributed = _setup_device_and_distributed()
    mask2former_config = build_mask2former_config(cfg, dataset_key="coco_dataset")
    feature_order = cfg.get("mask2former", {}).get("feature_order", ["1", "2", "3", "4"])
    autocast_dtype = _as_dtype(train_cfg.get("autocast_dtype", "bfloat16"), device.type)

    logger = init_training_logger(
        rank=rank,
        distributed=distributed,
        tensorboard_enabled=False,
        tensorboard_subdir=None,
        checkpoint_subdir=None,
        weights_root=None,
        local_weights_root=None,
    )

    eval_params = _resolve_eval_inference_params(dataset_cfg, eval_cfg)

    backbone = build_backbone(cfg, device)
    indices = resolve_interaction_indices(backbone, cfg)
    adapter = DINOv3_Adapter(backbone, interaction_indexes=indices).to(device)
    model = DinoMask2Former(adapter, mask2former_config, feature_order).to(device)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    logger.log_message(
        f"Loaded checkpoint from {checkpoint_path} (epoch={checkpoint.get('epoch')}, mIoU={checkpoint.get('best_miou'):.4f})"
    )

    _, val_dataset = build_coco_instance_datasets_from_cfg(dataset_cfg)
    pin_memory = device.type == "cuda"
    num_workers = max(1, train_cfg.get("num_workers", 8) // 2)
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", 1),
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        distributed=False,
        collate_fn=coco_instance_collate_fn,
    )

    coco_predictions = run_coco_inference(
        model=_unwrap(model),
        loader=val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        max_queries=int(dataset_cfg.get("prediction", {}).get("max_queries", 100)),
    )

    metrics = coco_evaluate_instances(
        coco_predictions,
        dataset_cfg["val_annotations"],
        output_dir=output_root,
    )
    metric_str = " ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
    logger.log_message(f"[COCO Evaluation] {metric_str}")

    export_semantic_predictions(
        model=_unwrap(model),
        loader=val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        output_dir=pred_dir,
        num_classes=int(dataset_cfg.get("num_classes", mask2former_config.num_labels)),
        eval_params=eval_params,
    )
    logger.log_message(f"Semantic predictions saved to: {pred_dir}")

    visualize_instance_predictions(
        val_dataset,
        coco_predictions,
        vis_dir,
        score_thresh=float(dataset_cfg.get("visualization", {}).get("score_thresh", 0.5)),
    )
    logger.log_message(f"Instance visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
