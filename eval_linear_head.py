import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import sys

# 确保可以以包形式导入 MAS.*
_CWD = Path(__file__).resolve().parent
_PROJECT_ROOT = _CWD.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

from MAS.data.data import FeatureCacheDataset, SegmentationFolderDataset, build_basic_dataloader  # type: ignore
from MAS.feature_cache import cache_features_for_dataset
from MAS.model.backbone_utils import build_backbone
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
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

from utils import load_config,_tensor_to_uint8_image
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")


logger = logging.getLogger("mas.seg_eval")
RESULTS_FILENAME = "results-semantic-segmentation.csv"


def init_distributed_mode():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    dist.barrier()
    return rank, world_size, torch.device("cuda", local_rank)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main_process():
    return get_rank() == 0


def synchronize():
    if is_distributed():
        dist.barrier()


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
    epochs: int = 100
    eval_every_epochs: int = 10
    feature_cache_dir: str = "./feature_cache"
    reuse_cached_features: bool = True
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


def _prepare_mask_for_display(mask_tensor: torch.Tensor, target_hw: Optional[tuple[int, int]] = None):
    mask = mask_tensor.detach().clone()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if target_hw is not None and mask.shape[-2:] != target_hw:
        mask = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=target_hw,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        ).to(mask.dtype)
    return mask


def _save_mask_image(
    mask_tensor: torch.Tensor,
    path: Path,
    ignore_index: Optional[int],
    target_hw: Optional[tuple[int, int]] = None,
):
    mask = _prepare_mask_for_display(mask_tensor, target_hw=target_hw).cpu()
    if ignore_index is not None:
        mask = mask.clone()
        mask[mask == ignore_index] = -1
    if mask.max() <= 1:
        mask = mask * 255
    mask = mask.clamp(min=0, max=255)
    mask_np = mask.to(torch.uint8).numpy()
    Image.fromarray(mask_np).save(path)


def save_debug_samples_from_cache(
    decoder_head: torch.nn.Module,
    cache_dir: Optional[Path],
    output_dir: Path,
    device: torch.device,
    seg_cfg: SegmentationEvalConfig,
    num_samples: int = 10,
) -> None:
    if cache_dir is None:
        return
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.warning("调试样本缓存目录 %s 不存在，跳过保存。", cache_dir)
        return
    cache_files = sorted(cache_dir.glob("*.pt"))
    if not cache_files:
        logger.warning("缓存目录 %s 中没有样本文件，无法导出调试样本。", cache_dir)
        return
    cache_files = cache_files[: max(1, num_samples)]
    debug_root = Path(output_dir) / "debug_samples"
    debug_root.mkdir(parents=True, exist_ok=True)

    model = decoder_head.module if isinstance(decoder_head, torch.nn.parallel.DistributedDataParallel) else decoder_head
    was_training = model.training
    model.eval()

    mean_tensor = torch.as_tensor(seg_cfg.transforms.mean, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.as_tensor(seg_cfg.transforms.std, dtype=torch.float32).view(-1, 1, 1)

    for idx, sample_path in enumerate(cache_files):
        sample = torch.load(sample_path, map_location="cpu")
        feature_list = sample["features"]
        mask = sample["mask"].long()
        image_tensor = sample.get("image")

        feature_batch = [feat.unsqueeze(0).to(device, non_blocking=True) for feat in feature_list]
        with torch.no_grad():
            logits = model(feature_batch)
        logits_resized = logits
        if logits.shape[-2:] != mask.shape[-2:]:
            logits_resized = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = logits_resized.argmax(dim=1).squeeze(0).cpu().to(torch.long)

        sample_dir = debug_root / f"sample_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "features": [feat.clone() for feat in feature_list],
                "logits": logits.squeeze(0).cpu(),
                "logits_resized": logits_resized.squeeze(0).cpu(),
                "pred_mask": pred_mask.clone(),
                "gt_mask": mask.clone(),
                "image": image_tensor.clone() if image_tensor is not None else None,
                "cache_file": str(sample_path),
            },
            sample_dir / "debug_data.pt",
        )

        image_hw = tuple(image_tensor.shape[-2:]) if image_tensor is not None else mask.shape[-2:]
        _save_mask_image(pred_mask, sample_dir / "pred_mask.png", ignore_index=None, target_hw=image_hw)
        _save_mask_image(mask, sample_dir / "gt_mask.png", ignore_index=seg_cfg.ignore_index, target_hw=image_hw)

        image_array = _tensor_to_uint8_image(image_tensor, mean_tensor, std_tensor)
        if image_array is not None:
            Image.fromarray(image_array).save(sample_dir / "input_image.png")
        else:
            logger.warning("样本 %s 缺少原始输入图像张量，未能导出 input_image.png", sample_path)

        del feature_batch, logits

    if was_training:
        model.train()


def load_segmentation_eval_config(base_cfg: Dict[str, Any], seg_cfg_path: Optional[str]) -> SegmentationEvalConfig:
    if seg_cfg_path:
        raw_cfg = OmegaConf.load(seg_cfg_path)
    else:
        seg_section = base_cfg.get("segmentation_eval")
        if seg_section is None:
            raise ValueError("未在主配置中找到 segmentation_eval 字段")
        raw_cfg = OmegaConf.create(seg_section)
    structured = OmegaConf.structured(SegmentationEvalConfig)
    merged = OmegaConf.merge(structured, raw_cfg)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


"""Train linear head across epochs with periodic evaluation."""
def train_linear_head(
    seg_cfg,
    decoder_head,
    train_loader,
    val_loader,
    device,
    debug_cache_dir=None,
    debug_num_samples=10,
    finetuned_weights_path: Optional[str] = None,
):
    torch.backends.cudnn.benchmark = True
    decoder_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder_head.to(device))
    if is_distributed():
        decoder_head = torch.nn.parallel.DistributedDataParallel(
            decoder_head,
            device_ids=[device.index],
            output_device=device.index,
        )

    params = [p for p in decoder_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": seg_cfg.optimizer.lr}],
        lr=seg_cfg.optimizer.lr,
        betas=(seg_cfg.optimizer.beta1, seg_cfg.optimizer.beta2),
        weight_decay=seg_cfg.optimizer.weight_decay,
    )

    scheduler_kwargs = dict(seg_cfg.scheduler.constructor_kwargs)
    for key in ("warmup_ratio", "pct_start"):
        if key in scheduler_kwargs:
            try:
                scheduler_kwargs[key] = float(scheduler_kwargs[key])
            except (TypeError, ValueError):
                pass

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, seg_cfg.epochs)
    scheduler = build_scheduler(
        seg_cfg.scheduler.type,
        optimizer=optimizer,
        lr=seg_cfg.optimizer.lr,
        total_iter=total_steps,
        constructor_kwargs=scheduler_kwargs,
    )

    autocast_enabled = device.type == "cuda" and seg_cfg.model_dtype == ModelDtype.BFLOAT16
    autocast_ctx = torch.autocast if autocast_enabled else _NullContext
    criterion = MultiSegmentationLoss(
        diceloss_weight=seg_cfg.losses.diceloss_weight,
        celoss_weight=seg_cfg.losses.celoss_weight,
    )

    best_metric = 0.0
    best_metrics_dict: Dict[str, float] = {}
    output_dir = Path(seg_cfg.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    eval_history: List[Dict[str, float]] = []
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    val_sampler = val_loader.sampler if isinstance(val_loader.sampler, DistributedSampler) else None
    world_size = dist.get_world_size() if is_distributed() else 1
    train_dataset_size = len(train_loader.dataset)
    block_start_time = time.time()
    start_time = block_start_time

    progress_total = seg_cfg.epochs
    progress_bar = (
        tqdm(
            total=progress_total,
            desc=f"Epoch 0/{seg_cfg.epochs}",
            disable=not is_main_process(),
        )
        if is_main_process() and progress_total > 0
        else None
    )

    last_eval_metrics: Dict[str, float] = {}

    for epoch in range(seg_cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
        if progress_bar is not None:
            progress_bar.set_description(f"Epoch {epoch + 1}/{seg_cfg.epochs}")

        images_seen = 0
        for features, labels in train_loader:
            features = [feat.to(device, non_blocking=True) for feat in features]
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device_type=device.type if autocast_enabled else "cpu", dtype=seg_cfg.model_dtype.autocast_dtype):
                logits = decoder_head(features)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_head.parameters(), seg_cfg.optimizer.gradient_clip)
            optimizer.step()
            scheduler.step()
            metric_logger.update(loss=float(loss.item()))

            if progress_bar is not None:
                batch_seen = len(labels) * world_size
                images_seen = min(train_dataset_size, images_seen + batch_seen)
                progress_bar.set_postfix_str(f"{images_seen}/{train_dataset_size}")

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"{train_dataset_size}/{train_dataset_size}")

        should_eval = ((epoch + 1) % seg_cfg.eval_every_epochs == 0) or ((epoch + 1) == seg_cfg.epochs)
        if should_eval:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            block_elapsed = time.time() - block_start_time
            metrics = evaluate_model(
                decoder_head.module if isinstance(decoder_head, torch.nn.parallel.DistributedDataParallel) else decoder_head,
                val_loader,
                device,
                seg_cfg.decoder_head.num_classes,
                seg_cfg,
                autocast_ctx,
            )
            last_eval_metrics = metrics
            if is_main_process():
                logger.info(
                    "Epoch %d/%d  loss=%.4f  mIoU=%.2f  images=%d",
                    epoch + 1,
                    seg_cfg.epochs,
                    metric_logger.loss.global_avg,
                    metrics.get("mIoU", 0.0),
                    metrics.get("num_images", 0),
                )
                print(
                    f"[Eval] epoch {epoch + 1}/{seg_cfg.epochs} loss={metric_logger.loss.global_avg:.4f} "
                    f"mIoU={metrics.get('mIoU', 0.0):.2f} imgs={metrics.get('num_images', 0)} block_time={block_elapsed:.1f}s"
                )
            m_iou = metrics.get("mIoU", 0.0)
            if m_iou > best_metric:
                best_metric = m_iou
                best_metrics_dict = metrics
                if seg_cfg.save_best_head and is_main_process():
                    torch.save(
                        (decoder_head.module if isinstance(decoder_head, torch.nn.parallel.DistributedDataParallel) else decoder_head).state_dict(),
                        output_dir / "linear_head_best.pth",
                    )
            eval_history.append(
                {
                    "epoch": float(epoch + 1),
                    "mIoU": m_iou,
                    "best_mIoU": best_metric,
                    "block_time_sec": block_elapsed,
                }
            )
            block_start_time = time.time()

    if progress_bar is not None:
        progress_bar.close()

    if is_main_process():
        torch.save(
            (decoder_head.module if isinstance(decoder_head, torch.nn.parallel.DistributedDataParallel) else decoder_head).state_dict(),
            output_dir / "linear_head_last.pth",
        )
        results_path = output_dir / RESULTS_FILENAME
        results_entry = dict(best_metrics_dict or {"mIoU": best_metric})
        if "num_images" not in results_entry:
            results_entry["num_images"] = last_eval_metrics.get("num_images", 0)
        results_entry["finetuned_weights"] = finetuned_weights_path or ""
        with open(results_path, "a", encoding="utf-8") as f:
            json.dump(results_entry, f, ensure_ascii=False)
            f.write("\n")
        history_path = output_dir / "eval_history.csv"
        if eval_history:
            with open(history_path, "w", newline="", encoding="utf-8") as hf:
                writer = csv.DictWriter(hf, fieldnames=["epoch", "mIoU", "best_mIoU", "block_time_sec"])
                writer.writeheader()
                writer.writerows(eval_history)
        logger.info("训练完成，总耗时 %.1fs，最佳 mIoU=%.2f", time.time() - start_time, best_metric)
    if is_main_process() and debug_cache_dir is not None:
        try:
            save_debug_samples_from_cache(
                decoder_head,
                Path(debug_cache_dir),
                output_dir,
                device,
                seg_cfg,
                num_samples=debug_num_samples,
            )
            logger.info("已将调试样本保存到 %s/debug_samples", output_dir)
        except Exception as exc:
            logger.error("保存调试样本失败：%s", exc)

    return best_metrics_dict or {"mIoU": best_metric}


def evaluate_model(model, dataloader, device, num_classes, seg_cfg, autocast_ctx):
    model.eval()
    sampler = dataloader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(0)
    total_iou = torch.tensor(0.0, device=device)
    image_counter = torch.tensor(0, device=device, dtype=torch.long)
    with torch.no_grad():
        for features, labels in dataloader:
            features = [feat.to(device, non_blocking=True) for feat in features]
            labels = labels.to(device, non_blocking=True)
            image_counter += labels.shape[0]
            with autocast_ctx(device_type=device.type if device.type == "cuda" else "cpu", dtype=seg_cfg.model_dtype.autocast_dtype):
                logits = model(features)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            for pred_mask, target_mask in zip(preds, labels):
                valid_mask = target_mask != seg_cfg.ignore_index
                pred_fg = (pred_mask == 1) & valid_mask
                target_fg = (target_mask == 1) & valid_mask
                intersection = (pred_fg & target_fg).sum().float()
                union = (pred_fg | target_fg).sum().float()
                if union.item() == 0:
                    total_iou += 1.0
                else:
                    total_iou += intersection / union
    if is_distributed():
        dist.all_reduce(image_counter)
        dist.all_reduce(total_iou)
    total_images = int(image_counter.item())
    mean_iou = float(total_iou.item() / max(1, total_images))
    model.train()
    return {"mIoU": round(mean_iou * 100.0, 2), "num_images": total_images}


class _NullContext:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线性分割头评估")
    parser.add_argument("--config", default="config_eval.yaml", help="主配置文件")
    parser.add_argument("--seg-config", default=None, help="可选分割评估配置文件")
    return parser.parse_args()


def main():
    global_start = time.time()
    rank, world_size, device = init_distributed_mode()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR)
    if is_main_process():
        logger.info("分布式：world_size=%d, rank=%d", world_size, rank)
    args = parse_args()
    base_cfg = load_config(args.config)
    seg_cfg = load_segmentation_eval_config(base_cfg, args.seg_config)

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
    feature_extractor = segmentation_model.segmentation_model[0]
    decoder_head = segmentation_model.segmentation_model[1]

    train_dataset = SegmentationFolderDataset(
        Path(seg_cfg.dataset.train_images).expanduser(),
        Path(seg_cfg.dataset.train_masks).expanduser(),
        make_segmentation_train_transforms(
            img_size=seg_cfg.transforms.train.img_size,
            random_img_size_ratio_range=seg_cfg.transforms.train.random_img_size_ratio_range,
            crop_size=seg_cfg.transforms.train.crop_size,
            flip_prob=seg_cfg.transforms.train.flip_prob,
            mean=seg_cfg.transforms.mean,
            std=seg_cfg.transforms.std,
        ),
    )
    val_dataset = SegmentationFolderDataset(
        Path(seg_cfg.dataset.val_images).expanduser(),
        Path(seg_cfg.dataset.val_masks).expanduser(),
        make_segmentation_eval_transforms(
            img_size=seg_cfg.transforms.eval.img_size,
            inference_mode=seg_cfg.eval.mode,
            use_tta=False,
            tta_ratios=seg_cfg.transforms.eval.tta_ratios,
            mean=seg_cfg.transforms.mean,
            std=seg_cfg.transforms.std,
        ),
    )

    log_fn = (lambda msg: logger.info(msg)) if is_main_process() else None
    cache_train = cache_features_for_dataset(
        split_name="train",
        dataset=train_dataset,
        cache_root=seg_cfg.feature_cache_dir,
        feature_extractor=feature_extractor,
        feature_order=None,
        device=device,
        batch_size=seg_cfg.batch_size,
        num_workers=seg_cfg.num_workers,
        reuse_existing=seg_cfg.reuse_cached_features,
        store_images=True,
        binarize_masks=True,
        log_fn=log_fn,
    )
    cache_val = cache_features_for_dataset(
        split_name="val",
        dataset=val_dataset,
        cache_root=seg_cfg.feature_cache_dir,
        feature_extractor=feature_extractor,
        feature_order=None,
        device=device,
        batch_size=seg_cfg.val_batch_size,
        num_workers=max(1, seg_cfg.num_workers // 2),
        reuse_existing=seg_cfg.reuse_cached_features,
        store_images=True,
        binarize_masks=True,
        log_fn=log_fn,
    )

    del segmentation_model
    del feature_extractor
    torch.cuda.empty_cache()

    train_loader = build_basic_dataloader(
        FeatureCacheDataset(cache_train),
        batch_size=seg_cfg.batch_size,
        num_workers=seg_cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        distributed=is_distributed(),
    )
    val_loader = build_basic_dataloader(
        FeatureCacheDataset(cache_val),
        batch_size=seg_cfg.val_batch_size,
        num_workers=max(1, seg_cfg.num_workers // 2),
        shuffle=False,
        pin_memory=True,
        distributed=is_distributed(),
    )

    results = train_linear_head(
        seg_cfg,
        decoder_head,
        train_loader,
        val_loader,
        device,
        debug_cache_dir=cache_val,
        debug_num_samples=10,
        finetuned_weights_path=base_cfg.get("finetuned_weights"),
    )
    if is_distributed():
        dist.destroy_process_group()
    if is_main_process():
        logger.info("最终结果: %s", results)
        logger.info("总训练耗时 %.1fs", time.time() - global_start)
        print(f"[Final Results] mIoU={results.get('mIoU', 0.0):.2f}")


if __name__ == "__main__":
    main()
