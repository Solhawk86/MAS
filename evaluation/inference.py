from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import amp
from PIL import Image


def _semantic_from_logits(class_logits: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask_classes = class_logits.softmax(dim=-1)[..., :-1]
    mask_probs = mask_logits.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", mask_classes, mask_probs)


def _resolve_hw_pair(value, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"期望长度为 2 的序列或整数，收到: {value!r}")


def _forward_semantic_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    depth_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with amp.autocast(autocast_device, dtype=autocast_dtype, enabled=device.type == "cuda"):
        outputs = model(images, depth_maps=depth_maps)
    return _semantic_from_logits(outputs["class_queries_logits"], outputs["masks_queries_logits"])


def _slide_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    depth_map: torch.Tensor | None = None,
) -> torch.Tensor:
    _, h_img, w_img = image.shape
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    preds = torch.zeros((1, num_classes, h_img, w_img), device=device, dtype=torch.float32)
    count = torch.zeros((1, 1, h_img, w_img), device=device, dtype=torch.float32)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop = image[:, y1:y2, x1:x2].unsqueeze(0)
            depth_crop = None
            if depth_map is not None:
                depth_crop = depth_map[:, y1:y2, x1:x2].unsqueeze(0)
            crop_logits = _forward_semantic_logits(
                model,
                crop,
                device=device,
                autocast_dtype=autocast_dtype,
                depth_maps=depth_crop,
            )
            crop_logits = F.interpolate(
                crop_logits,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )
            preds[:, :, y1:y2, x1:x2] += crop_logits
            count[:, :, y1:y2, x1:x2] += 1
    return preds / torch.clamp(count, min=1.0)


def _build_tta_views(
    image: torch.Tensor,
    eval_params: dict,
    depth_map: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None, bool]]:
    ratios = eval_params.get("tta_ratios") or [1.0]
    use_hflip = bool(eval_params.get("use_tta", False))
    views: list[tuple[torch.Tensor, torch.Tensor | None, bool]] = []
    for ratio in ratios:
        if isinstance(ratio, (int, float)) and ratio > 0:
            if abs(float(ratio) - 1.0) < 1e-6:
                scaled = image
            else:
                h = max(1, int(round(image.shape[-2] * float(ratio))))
                w = max(1, int(round(image.shape[-1] * float(ratio))))
                scaled = F.interpolate(
                    image.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if depth_map is not None:
                    scaled_depth = F.interpolate(
                        depth_map.unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    scaled_depth = None
            if depth_map is not None and abs(float(ratio) - 1.0) < 1e-6:
                scaled_depth = depth_map
        else:
            raise ValueError(f"非法 tta ratio: {ratio}")
        views.append((scaled, scaled_depth if depth_map is not None else None, False))
        if use_hflip:
            flipped_depth = (
                torch.flip(scaled_depth, dims=[-1])
                if depth_map is not None and scaled_depth is not None
                else None
            )
            views.append((torch.flip(scaled, dims=[-1]), flipped_depth, True))
    return views


def _predict_single_view(
    model: torch.nn.Module,
    view: torch.Tensor,
    *,
    eval_params: dict,
    num_classes: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    target_size: tuple[int, int],
    depth_map: torch.Tensor | None = None,
) -> torch.Tensor:
    inference_mode = str(eval_params.get("inference_mode", "slide")).lower() or "slide"
    crop_hw = _resolve_hw_pair(eval_params.get("crop_size") or eval_params.get("img_size"), default=view.shape[-2:])
    stride_hw = _resolve_hw_pair(
        eval_params.get("stride"),
        default=(max(1, crop_hw[0] * 2 // 3), max(1, crop_hw[1] * 2 // 3)),
    )
    if inference_mode == "slide":
        logits = _slide_inference(
            model,
            view,
            crop_size=crop_hw,
            stride=stride_hw,
            num_classes=num_classes,
            device=device,
            autocast_dtype=autocast_dtype,
            depth_map=depth_map,
        )
    else:
        resized = F.interpolate(
            view.unsqueeze(0),
            size=crop_hw,
            mode="bilinear",
            align_corners=False,
        )
        resized_depth = None
        if depth_map is not None:
            resized_depth = F.interpolate(
                depth_map.unsqueeze(0),
                size=crop_hw,
                mode="bilinear",
                align_corners=False,
            )
        logits = _forward_semantic_logits(
            model,
            resized,
            device=device,
            autocast_dtype=autocast_dtype,
            depth_maps=resized_depth,
        )
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def _run_full_image_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    eval_params: dict,
    num_classes: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    target_size: tuple[int, int],
    depth_map: torch.Tensor | None = None,
) -> torch.Tensor:
    views = _build_tta_views(image, eval_params, depth_map)
    view_count = len(views)
    aggregated: torch.Tensor | None = None
    for view, depth_view, flipped in views:
        logits = _predict_single_view(
            model,
            view,
            eval_params=eval_params,
            num_classes=num_classes,
            device=device,
            autocast_dtype=autocast_dtype,
            target_size=target_size,
            depth_map=depth_view,
        )
        if flipped:
            logits = torch.flip(logits, dims=[-1])
        aggregated = logits if aggregated is None else aggregated + logits
    if aggregated is None:
        aggregated = _predict_single_view(
            model,
            image,
            eval_params=eval_params,
            num_classes=num_classes,
            device=device,
            autocast_dtype=autocast_dtype,
            target_size=target_size,
            depth_map=depth_map,
        )
        view_count = 1
    return aggregated / float(view_count)


def _resolve_eval_inference_params(dataset_cfg: dict, eval_cfg: dict) -> dict:
    transforms_eval = dataset_cfg.get("transforms", {}).get("eval", {}) or {}
    resolved = {}
    resolved["inference_mode"] = eval_cfg.get("inference_mode") or transforms_eval.get("inference_mode", "slide")
    resolved["img_size"] = eval_cfg.get("img_size") or transforms_eval.get("img_size", 512)
    resolved["crop_size"] = eval_cfg.get("crop_size") or resolved["img_size"]
    resolved["stride"] = eval_cfg.get("stride") if eval_cfg.get("stride") is not None else transforms_eval.get("stride")
    resolved["use_tta"] = (
        eval_cfg.get("use_tta") if eval_cfg.get("use_tta") is not None else transforms_eval.get("use_tta", False)
    )
    resolved["tta_ratios"] = eval_cfg.get("tta_ratios") or transforms_eval.get("tta_ratios", [1.0])
    return resolved


def export_predictions(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    output_dir: Path | str,
    num_classes: int,
    eval_params: dict,
) -> None:
    """
    Run sliding-window inference over an entire loader and write grayscale mask PNGs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, _masks, sample_indices, orig_sizes, *rest = batch
            file_names = rest[0] if rest else None
            depth_maps = rest[1] if len(rest) > 1 and torch.is_tensor(rest[1]) else None
            images = images.to(device, non_blocking=True)
            if depth_maps is not None:
                depth_maps = depth_maps.to(device, non_blocking=True)
            batch_size = images.shape[0]
            for b in range(batch_size):
                target_h = int(orig_sizes[b][0].item())
                target_w = int(orig_sizes[b][1].item())
                if target_h <= 0 or target_w <= 0:
                    target_h, target_w = images.shape[-2:]
                logits = _run_full_image_inference(
                    model,
                    images[b],
                    depth_map=depth_maps[b] if depth_maps is not None else None,
                    eval_params=eval_params,
                    num_classes=num_classes,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    target_size=(target_h, target_w),
                )
                pred_mask = logits.argmax(dim=1).squeeze(0).float()
                max_value = max(1, int(num_classes) - 1)
                mask_tensor = torch.clamp(pred_mask, 0.0, float(max_value))
                if max_value > 0:
                    mask_tensor = mask_tensor / float(max_value)
                mask_tensor = (mask_tensor * 255).to(torch.uint8).cpu().numpy()
                if file_names:
                    stem = str(file_names[b])
                else:
                    if isinstance(sample_indices, torch.Tensor):
                        stem = f"sample_{int(sample_indices[b].item())}"
                    else:
                        stem = f"sample_{int(sample_indices[b])}"
                out_path = output_path / f"{stem}.png"
                Image.fromarray(mask_tensor, mode="L").save(out_path)
