from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import amp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from pycocotools import mask as mask_utils

from MAS.eval_inference import _run_full_image_inference
from MAS.utils import _distributed_mean

__all__ = [
    "train_one_epoch",
    "evaluate",
    "run_coco_inference",
]


def _build_instance_targets(extras: Sequence[Dict[str, torch.Tensor]], device: torch.device):
    """把 Dataset 中缓存的 instance_masks / class_labels 搬到目标设备。"""

    mask_labels: List[torch.Tensor] = []
    class_labels: List[torch.Tensor] = []
    for extra in extras:
        masks = extra.get("instance_masks", torch.zeros((0, 1, 1)))
        classes = extra.get("instance_class_labels", torch.zeros((0,), dtype=torch.long))
        mask_labels.append(masks.to(device, non_blocking=True))
        class_labels.append(classes.to(device, non_blocking=True))
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
        images, _semantic_masks, *_other, extras = batch
        images = images.to(device, non_blocking=True)
        mask_labels, class_labels = _build_instance_targets(extras, device)
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
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch in loader:
            images, _masks, _indices, _orig_sizes, *_rest, extras = batch
            images = images.to(device, non_blocking=True)
            mask_labels, class_labels = _build_instance_targets(extras, device)
            with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(images, mask_labels=mask_labels, class_labels=class_labels)
            loss_sum += float(outputs["loss"].item())
            count += 1
    mean_loss = _distributed_mean(torch.tensor([loss_sum / max(1, count)], device=device), world_size)
    return float(mean_loss.item())


def _queries_to_coco_instances(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    *,
    max_queries: int,
    category_mapping: Sequence[int] | None = None,
) -> List[Dict[str, object]]:
    probs = class_logits.softmax(dim=-1)[..., :-1]
    num_queries, num_classes = probs.shape
    flat_scores = probs.flatten(0, 1)
    k = min(int(max_queries), flat_scores.shape[0]) if max_queries > 0 else flat_scores.shape[0]
    scores_per_image, topk_indices = flat_scores.topk(k, sorted=False)
    labels_per_image = topk_indices % num_classes
    query_indices = topk_indices // num_classes
    mask_probs = mask_logits.sigmoid()
    target_h, target_w = target_hw
    predictions: List[Dict[str, object]] = []
    for score, query_idx, cls_idx in zip(scores_per_image.tolist(), query_indices.tolist(), labels_per_image.tolist()):
        if category_mapping is not None and 0 <= cls_idx < len(category_mapping):
            cls = int(category_mapping[cls_idx])
        else:
            cls = int(cls_idx)
        mask = mask_probs[query_idx].unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(
            mask,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        mask_np = mask_resized.detach().float().cpu().numpy()
        binary = (mask_np >= 0.5).astype(np.uint8)
        mask_area = int(binary.sum())
        if mask_area == 0:
            continue
        mask_confidence = float((mask_np * binary).sum() / (mask_area + 1e-6))
        final_score = float(score * mask_confidence)
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("ascii")
        bbox = mask_utils.toBbox(rle).tolist()
        predictions.append(
            {
                "segmentation": rle,
                "score": final_score,
                "category_id": cls,
                "bbox": bbox,
            }
        )
    return predictions


def run_coco_inference(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_queries: int,
) -> List[Dict[str, object]]:
    predictions: List[Dict[str, object]] = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    dataset = getattr(loader, "dataset", None)
    category_mapping = None
    if dataset is not None and hasattr(dataset, "contiguous_id_to_cat"):
        category_mapping = list(getattr(dataset, "contiguous_id_to_cat"))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, _masks, _indices, orig_sizes, *_rest, extras = batch
            images = images.to(device, non_blocking=True)
            with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(images)
            class_logits = outputs["class_queries_logits"]
            mask_logits = outputs["masks_queries_logits"]
            batch_size = images.shape[0]
            for b in range(batch_size):
                target_h = int(orig_sizes[b][0].item())
                target_w = int(orig_sizes[b][1].item())
                image_id = int(extras[b]["image_id"])
                coco_instances = _queries_to_coco_instances(
                    class_logits[b],
                    mask_logits[b],
                    (target_h, target_w),
                    max_queries=max_queries,
                    category_mapping=category_mapping,
                )
                for inst in coco_instances:
                    inst["image_id"] = image_id
                predictions.append(
                    {
                        "image_id": image_id,
                        "instances": coco_instances,
                    }
                )
    return predictions
