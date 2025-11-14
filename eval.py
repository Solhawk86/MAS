import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dinov3.eval.segmentation.config import (
    DecoderConfig,
    EvalConfig as SlidingEvalConfig,
    ModelDtype,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig as LossConfig,
    TransformConfig,
    TrainTransformConfig,
    EvalTransformConfig,
)
from dinov3.eval.segmentation.eval import RESULTS_FILENAME
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import (
    SEGMENTATION_METRICS,
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
)
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import (
    make_segmentation_eval_transforms,
    make_segmentation_train_transforms,
)
from dinov3.logging import MetricLogger, SmoothedValue
from dinov3.models import build_model_for_eval

from Dino_LoRA import inject_lora_qv_into_vit
from utils import dict_to_namespace, load_config, prepare_dinov3_cfg

logger = logging.getLogger("mas.seg_eval")


@dataclass
class DatasetFolders:
    train_images: str = ""
    train_masks: str = ""
    val_images: str = ""
    val_masks: str = ""


@dataclass
class SegmentationEvalConfig:
    output_dir: str = "./segmentation_eval"
    batch_size: int = 2
    val_batch_size: int = 1
    num_workers: int = 4
    max_steps: int = 4000
    eval_interval: int = 1000
    dataset: DatasetFolders = field(default_factory=DatasetFolders)
    decoder_head: DecoderConfig = field(default_factory=DecoderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    transforms: TransformConfig = field(
        default_factory=lambda: TransformConfig(
            train=TrainTransformConfig(img_size=518, crop_size=(512, 512), flip_prob=0.5),
            eval=EvalTransformConfig(img_size=518),
        )
    )
    losses: LossConfig = field(default_factory=LossConfig)
    eval: SlidingEvalConfig = field(default_factory=SlidingEvalConfig)
    model_dtype: ModelDtype = ModelDtype.FLOAT32
    ignore_index: int = 255
    save_best_head: bool = True


class SegmentationFolderDataset(Dataset):
    """读取 image/mask 文件夹并与 dinov3 增强保持一致。"""

    def __init__(self, image_dir: Path, mask_dir: Path, transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        image_map = self._scan(image_dir, {"jpg", "jpeg", "png"})
        mask_map = self._scan(mask_dir, {"png", "jpg", "jpeg"})
        common_keys = sorted(image_map.keys() & mask_map.keys())
        if not common_keys:
            raise ValueError(f"在 {image_dir} 与 {mask_dir} 中找不到匹配的图像/掩码文件")

        self.image_paths = [image_map[k] for k in common_keys]
        self.mask_paths = [mask_map[k] for k in common_keys]

    @staticmethod
    def _scan(directory: Path, extensions: Iterable[str]) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in extensions:
                continue
            result[path.stem] = path
        return result

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        image, mask = self.transforms(image, mask)
        image_tensor = self._ensure_float_tensor(image)
        mask_tensor = self._ensure_long_mask(mask)
        return image_tensor, mask_tensor

    @staticmethod
    def _ensure_float_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("当前评估脚本仅支持单尺度（use_tta=False）")
            image = image[0]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        return image.float()

    @staticmethod
    def _ensure_long_mask(mask: Any) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            tensor = mask
        else:
            tensor = torch.as_tensor(mask)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.long()


def load_segmentation_eval_config(base_cfg: Dict[str, Any], seg_cfg_path: Optional[str]) -> SegmentationEvalConfig:
    if seg_cfg_path:
        raw_cfg = OmegaConf.load(seg_cfg_path)
    else:
        seg_section = base_cfg.get("segmentation_eval")
        if seg_section is None:
            raise ValueError("未在主配置中找到 `segmentation_eval` 字段，请提供 --seg-config。")
        raw_cfg = OmegaConf.create(seg_section)
    structured = OmegaConf.structured(SegmentationEvalConfig)
    merged = OmegaConf.merge(structured, raw_cfg)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def build_backbone(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    processed = prepare_dinov3_cfg(cfg)
    ns_cfg = dict_to_namespace(processed)
    pretrained = processed.get("pretrained_weights")
    backbone = build_model_for_eval(config=ns_cfg, pretrained_weights=pretrained)
    backbone.to(device)
    backbone.eval()
    lora_layers = cfg.get("train", {}).get("lora_layers")
    if lora_layers not in (None, 0, []):
        inject_lora_qv_into_vit(
            backbone,
            layers=lora_layers,
            rank=int(cfg.get("train", {}).get("lora_rank", 8)),
            alpha=float(cfg.get("train", {}).get("lora_alpha", 16.0)),
            dropout=float(cfg.get("train", {}).get("lora_dropout", 0.0)),
        )
    finetuned_path = cfg.get("finetuned_weights")
    if finetuned_path:
        state = torch.load(Path(finetuned_path).expanduser(), map_location="cpu")
        backbone_state = _extract_backbone_state(state)
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            logger.warning("加载 finetuned_weights 时缺少参数: %s", missing)
        if unexpected:
            logger.warning("加载 finetuned_weights 时出现未使用参数: %s", unexpected)
    backbone.requires_grad_(False)
    return backbone


def _extract_backbone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if "student" in state and isinstance(state["student"], dict) and "backbone" in state["student"]:
        return state["student"]["backbone"]
    if "backbone" in state and isinstance(state["backbone"], dict):
        return state["backbone"]
    return state


def create_dataloaders(
    seg_cfg: SegmentationEvalConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_imgs = Path(seg_cfg.dataset.train_images).expanduser()
    train_masks = Path(seg_cfg.dataset.train_masks).expanduser()
    val_imgs = Path(seg_cfg.dataset.val_images).expanduser()
    val_masks = Path(seg_cfg.dataset.val_masks).expanduser()

    train_transforms = make_segmentation_train_transforms(
        img_size=seg_cfg.transforms.train.img_size,
        random_img_size_ratio_range=seg_cfg.transforms.train.random_img_size_ratio_range,
        crop_size=seg_cfg.transforms.train.crop_size,
        flip_prob=seg_cfg.transforms.train.flip_prob,
        reduce_zero_label=seg_cfg.transforms.train.reduce_zero_label,
        mean=seg_cfg.transforms.mean,
        std=seg_cfg.transforms.std,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=seg_cfg.transforms.eval.img_size,
        inference_mode=seg_cfg.eval.mode,
        use_tta=False,
        tta_ratios=seg_cfg.transforms.eval.tta_ratios,
        mean=seg_cfg.transforms.mean,
        std=seg_cfg.transforms.std,
    )

    train_dataset = SegmentationFolderDataset(train_imgs, train_masks, train_transforms)
    val_dataset = SegmentationFolderDataset(val_imgs, val_masks, val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=seg_cfg.batch_size,
        shuffle=True,
        num_workers=seg_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=seg_cfg.val_batch_size,
        shuffle=False,
        num_workers=max(1, seg_cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_linear_head(
    base_cfg: Dict[str, Any],
    seg_cfg: SegmentationEvalConfig,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    backbone = build_backbone(base_cfg, device)
    decoder_cfg = seg_cfg.decoder_head
    decoder_cfg.type = "linear"
    segmentation_model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=decoder_cfg.backbone_out_layers,
        decoder_type="linear",
        num_classes=decoder_cfg.num_classes,
        autocast_dtype=seg_cfg.model_dtype.autocast_dtype,
    ).to(device)
    decoder_head = segmentation_model.segmentation_model[1]
    params = [p for p in decoder_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": seg_cfg.optimizer.lr}],
        lr=seg_cfg.optimizer.lr,
        betas=(seg_cfg.optimizer.beta1, seg_cfg.optimizer.beta2),
        weight_decay=seg_cfg.optimizer.weight_decay,
    )
    scheduler = build_scheduler(
        seg_cfg.scheduler.type,
        optimizer=optimizer,
        lr=seg_cfg.optimizer.lr,
        total_iter=seg_cfg.scheduler.total_iter,
        constructor_kwargs=seg_cfg.scheduler.constructor_kwargs,
    )
    autocast_enabled = device.type == "cuda" and seg_cfg.model_dtype == ModelDtype.BFLOAT16
    autocast_ctx = torch.autocast if autocast_enabled else _null_autocast
    criterion = MultiSegmentationLoss(
        diceloss_weight=seg_cfg.losses.diceloss_weight,
        celoss_weight=seg_cfg.losses.celoss_weight,
    )
    train_loader, val_loader = create_dataloaders(seg_cfg)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    best_metric = 0.0
    best_metrics_dict: Dict[str, float] = {}
    output_dir = Path(seg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)

    train_iter = iter(train_loader)
    start_time = time.time()
    for global_step in range(seg_cfg.max_steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device_type=device.type if autocast_enabled else "cpu", dtype=seg_cfg.model_dtype.autocast_dtype):
            logits = segmentation_model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder_head.parameters(), seg_cfg.optimizer.gradient_clip)
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss=float(loss.item()))

        if (global_step + 1) % seg_cfg.eval_interval == 0 or (global_step + 1) == seg_cfg.max_steps:
            metrics = evaluate_model(
                segmentation_model,
                val_loader,
                device,
                seg_cfg.decoder_head.num_classes,
                seg_cfg,
                autocast_ctx,
            )
            m_iou = metrics.get("mIoU", 0.0)
            logger.info(
                "Step %d/%d  loss=%.4f  mIoU=%.2f",
                global_step + 1,
                seg_cfg.max_steps,
                metric_logger.loss.global_avg,
                m_iou,
            )
            if m_iou > best_metric:
                best_metric = m_iou
                best_metrics_dict = metrics
                if seg_cfg.save_best_head:
                    torch.save(
                        decoder_head.state_dict(),
                        output_dir / "linear_head_best.pth",
                    )

    elapsed = time.time() - start_time
    logger.info("训练完成，总耗时 %.1fs，最佳 mIoU=%.2f", elapsed, best_metric)
    torch.save(decoder_head.state_dict(), output_dir / "linear_head_last.pth")
    results_path = output_dir / RESULTS_FILENAME
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics_dict or {"mIoU": best_metric}, f, indent=2, ensure_ascii=False)
    return best_metrics_dict or {"mIoU": best_metric}


def evaluate_model(
    segmentation_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    seg_cfg: SegmentationEvalConfig,
    autocast_ctx,
) -> Dict[str, float]:
    segmentation_model.eval()
    all_metrics: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_ctx(device_type=device.type if device.type == "cuda" else "cpu", dtype=seg_cfg.model_dtype.autocast_dtype):
                logits = segmentation_model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            for pred_mask, target_mask in zip(preds, labels):
                metric_tensor = calculate_intersect_and_union(
                    pred_mask.cpu(),
                    target_mask.cpu(),
                    num_classes=num_classes,
                    ignore_index=seg_cfg.ignore_index,
                    reduce_zero_label=seg_cfg.eval.reduce_zero_label,
                )
                all_metrics.append(metric_tensor)
    segmentation_model.train()
    if not all_metrics:
        return {}
    stacked = torch.stack(all_metrics)
    metrics = calculate_segmentation_metrics(
        stacked,
        metrics=["mIoU"],
        beta=1,
    )
    return {"mIoU": round(float(metrics["mIoU"].cpu().item()) * 100.0, 2)}


class _null_autocast:
    """Fallback context when不需要 autocast."""

    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并评估线性分割头")
    parser.add_argument(
        "--config",
        default="config_eval.yaml",
        help="主配置文件，至少包含 DINOv3 backbone 信息",
    )
    parser.add_argument(
        "--seg-config",
        default=None,
        help="可选的分割评估配置文件；若不提供，则从主配置中的 segmentation_eval 读取",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    base_cfg = load_config(args.config)
    seg_cfg = load_segmentation_eval_config(base_cfg, args.seg_config)
    results = train_linear_head(base_cfg, seg_cfg)
    logger.info("最终评估结果: %s", results)


if __name__ == "__main__":
    main()
