#!/usr/bin/env python3
"""Run sliding-window inference on a saved Mask2Former checkpoint and report metrics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
MAS_DIR = CURRENT_DIR.parent
PROJECT_ROOT = MAS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAS.data.data import SegmentationFolderDataset  # noqa: E402
from MAS.evaluation.inference import (  # noqa: E402
    _resolve_eval_inference_params,
    export_predictions,
)
from MAS.evaluation.metrics import evaluate_prediction_directory  # noqa: E402
from MAS.mask2former.model import DinoMask2Former  # noqa: E402
from MAS.mask2former.utils import (  # noqa: E402
    _as_dtype,
    build_mask2former_config,
    resolve_interaction_indices,
)
from MAS.model.backbone_utils import build_backbone  # noqa: E402
from MAS.utils import load_config  # noqa: E402
from dinov3.eval.segmentation.transforms import (
    make_segmentation_eval_transforms,
)  # noqa: E402
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
    DINOv3_Adapter,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a train_m2f checkpoint on a segmentation dataset"
    )
    default_cfg = CURRENT_DIR / "configs" / "m2f_eval_example.yaml"
    parser.add_argument(
        "--config", type=str, default=str(default_cfg), help="YAML 配置文件路径"
    )
    return parser.parse_args()


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_device(device_pref: str | None) -> torch.device:
    if not device_pref or device_pref.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    key = device_pref.lower()
    if key in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(key if key in {"cpu", "cuda"} else "cpu")


def _build_eval_dataset(dataset_cfg: Dict[str, Any]) -> SegmentationFolderDataset:
    val_images = dataset_cfg.get("val_images")
    val_masks = dataset_cfg.get("val_masks")
    if not val_images or not val_masks:
        raise ValueError("请在配置文件的 dataset 中提供 val_images 与 val_masks")
    transforms_cfg = dataset_cfg.get("transforms", {})
    eval_tf_cfg = transforms_cfg.get("eval", {}) or {}
    img_size = eval_tf_cfg.get("img_size", 512)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    eval_transforms = make_segmentation_eval_transforms(
        img_size=img_size,
        inference_mode=eval_tf_cfg.get("inference_mode", "slide"),
        use_tta=eval_tf_cfg.get("use_tta", False),
        tta_ratios=eval_tf_cfg.get("tta_ratios", [1.0]),
        mean=transforms_cfg.get("mean", (0.485, 0.456, 0.406)),
        std=transforms_cfg.get("std", (0.229, 0.224, 0.225)),
    )
    return SegmentationFolderDataset(
        Path(val_images).expanduser(),
        Path(val_masks).expanduser(),
        eval_transforms,
    )


def _load_checkpoint(
    model: torch.nn.Module, ckpt_path: str | Path, device: torch.device
) -> dict:
    checkpoint = torch.load(Path(ckpt_path), map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise RuntimeError(f"无法从 checkpoint {ckpt_path} 读取 state_dict")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state_dict}


def main() -> None:
    args = parse_args()
    eval_cfg = load_config(args.config)
    if not isinstance(eval_cfg, dict):
        raise ValueError("无法解析评估配置文件")

    base_cfg_path = eval_cfg.get("model_config")
    base_cfg = load_config(base_cfg_path) if base_cfg_path else {}
    merged_cfg = deepcopy(base_cfg) if base_cfg else {}

    dataset_override = eval_cfg.get("dataset") or {}
    if dataset_override:
        merged_cfg.setdefault("dataset", {})
        _deep_update(merged_cfg["dataset"], dataset_override)
    eval_override = eval_cfg.get("evaluation") or {}
    merged_eval_cfg = deepcopy(merged_cfg.get("evaluation", {}))
    _deep_update(merged_eval_cfg, eval_override)

    dataset_cfg = merged_cfg.get("dataset", {})
    if not dataset_cfg:
        raise ValueError("配置中缺少 dataset 字段")

    checkpoint_path = eval_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("请在配置文件中提供 checkpoint_path")

    device = _resolve_device(merged_eval_cfg.get("device"))
    autocast_dtype = _as_dtype(
        merged_eval_cfg.get("autocast_dtype", "bfloat16"), device.type
    )

    dataset = _build_eval_dataset(dataset_cfg)
    eval_loader = DataLoader(
        dataset,
        batch_size=int(merged_eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(merged_eval_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )

    mask2former_config = build_mask2former_config(merged_cfg)
    feature_order = merged_cfg.get("mask2former", {}).get(
        "feature_order", ["1", "2", "3", "4"]
    )

    backbone = build_backbone(merged_cfg, device)
    indices = resolve_interaction_indices(backbone, merged_cfg)
    adapter = DINOv3_Adapter(backbone, interaction_indexes=indices).to(device)
    model = DinoMask2Former(adapter, mask2former_config, feature_order).to(device)
    _ = _load_checkpoint(model, checkpoint_path, device)

    num_classes = int(dataset_cfg.get("num_classes", mask2former_config.num_labels))
    eval_params = _resolve_eval_inference_params(dataset_cfg, merged_eval_cfg)

    output_root = Path(merged_eval_cfg.get("output_dir", "runs/m2f_eval")).expanduser()
    pred_dir = output_root / merged_eval_cfg.get("prediction_dir", "predictions")
    output_root.mkdir(parents=True, exist_ok=True)

    export_predictions(
        model=model,
        loader=eval_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        output_dir=pred_dir,
        num_classes=num_classes,
        eval_params=eval_params,
    )
    print(f"预测结果已保存至: {pred_dir}")

    metrics = None
    if merged_eval_cfg.get("compute_metrics", True):
        gt_dir = merged_eval_cfg.get("gt_dir") or dataset_cfg.get("val_masks")
        if gt_dir:
            metrics = evaluate_prediction_directory(pred_dir, gt_dir)
            summary = " ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            print(f"评估指标 -> {summary}")
        else:
            print("未提供 GT 路径，跳过指标计算。")


if __name__ == "__main__":
    main()
