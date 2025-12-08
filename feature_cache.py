from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist

from MAS.data.data import build_basic_dataloader
from tqdm.auto import tqdm


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return not _dist_is_initialized() or dist.get_rank() == 0


def _synchronize() -> None:
    if _dist_is_initialized():
        dist.barrier()


def cache_is_complete(cache_dir: Path, expected_len: int) -> bool:
    if not cache_dir.exists():
        return False
    return len(list(cache_dir.glob("*.pt"))) >= expected_len


def _prepare_feature_list(features: torch.Tensor | Sequence[torch.Tensor] | dict, feature_order: Optional[Sequence[str]]):
    if isinstance(features, dict):
        order = feature_order or sorted(features.keys())
        return [features[key] for key in order]
    if isinstance(features, (list, tuple)):
        return list(features)
    return [features]


def cache_features_for_dataset(
    split_name: str,
    dataset,
    *,
    cache_root: Path | str,
    feature_extractor: torch.nn.Module,
    feature_order: Optional[Sequence[str]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    reuse_existing: bool = True,
    store_images: bool = False,
    binarize_masks: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
):
    cache_root = Path(cache_root).expanduser()
    cache_dir = cache_root / split_name
    if _is_main_process():
        cache_dir.mkdir(parents=True, exist_ok=True)
    _synchronize()
    if reuse_existing and cache_is_complete(cache_dir, len(dataset)):
        if log_fn is not None and _is_main_process():
            log_fn(f"复用缓存 {cache_dir}（{len(list(cache_dir.glob('*.pt')))} 样本）")
        _synchronize()
        return cache_dir

    loader = build_basic_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        distributed=_dist_is_initialized(),
    )
    feature_extractor.eval()
    if _is_main_process():
        if _dist_is_initialized() and loader.sampler is not None:
            try:
                total_local = len(loader.sampler)  # type: ignore[arg-type]
            except TypeError:
                total_local = (len(dataset) + dist.get_world_size() - 1) // dist.get_world_size()
        else:
            total_local = len(dataset)
        progress_bar = tqdm(total=total_local, desc=f"缓存 {split_name}", unit="img")
    else:
        progress_bar = None
    processed = 0
    for images, masks, indices, *_ in loader:
        images_cpu = images
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            feats = feature_extractor(images)
        feature_list = _prepare_feature_list(feats, feature_order)
        feature_list = [feat.cpu() for feat in feature_list]
        masks_cpu = masks.cpu()
        for b_idx, sample_idx in enumerate(indices.tolist()):
            mask_tensor = masks_cpu[b_idx].to(torch.long)
            if binarize_masks:
                mask_tensor = (mask_tensor > 0).long()
            sample = {
                "features": [feat[b_idx].contiguous() for feat in feature_list],
                "mask": mask_tensor,
            }
            if store_images and isinstance(images_cpu, torch.Tensor):
                sample["image"] = images_cpu[b_idx].clone()
            torch.save(sample, cache_dir / f"{sample_idx:08d}.pt")
        processed += len(indices)
        if progress_bar is not None:
            progress_bar.update(len(indices))
    if progress_bar is not None:
        progress_bar.close()
    _synchronize()
    if log_fn is not None and _is_main_process():
        log_fn(f"已生成缓存 {cache_dir}（{len(list(cache_dir.glob('*.pt')))} 样本）")
    return cache_dir


__all__ = ["cache_features_for_dataset", "cache_is_complete"]
