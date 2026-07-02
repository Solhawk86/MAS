import random
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional, Callable, Any, Iterable, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

from data.dino_augmentations import DataAugmentationDINO
from data.augmentation import BatchWaveletSwapper
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator
from MAS.data.segmentation_transforms import (
    make_segmentation_eval_transforms,
    make_segmentation_train_transforms,
)

# NOTE: 图像+mask 数据集，用于下游分割任务；新增自监督数据pipeline。


def make_transform(resize_size: int | list[int] = 518):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def resize_with_patch_transform(
    mask_image,
    image_size: int = 518,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    mask 与特征 patch 对齐，保证预测 mask 与 GT 尺寸一致
    """
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(
        TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size))
    )


def make_enhance_transform(resize_size: int | list[int] = 518) -> Callable:
    """
    返回用于增强图像的 transform。
    默认包含颜色抖动 / 随机翻转，与 make_transform 保持相同的归一化。
    """
    augmentations = [
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return v2.Compose(augmentations)


class SegmentationFolderDataset(TorchDataset):
    """图像/掩码文件夹对，按文件名匹配。"""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transforms,
        *,
        keep_image_lists: bool = False,
        ignore_index: Optional[int] = None,
        depth_dir: Path | None = None,
        require_depth: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.require_depth = require_depth
        self.transforms = transforms
        self.keep_image_lists = keep_image_lists
        self.ignore_index = ignore_index
        image_map = self._scan(image_dir, {"jpg", "jpeg", "png"})
        mask_map = self._scan(mask_dir, {"png", "jpg", "jpeg"})
        common = sorted(image_map.keys() & mask_map.keys())
        if not common:
            raise ValueError(f"在 {image_dir} 与 {mask_dir} 中找不到匹配文件")
        self.image_paths = [image_map[k] for k in common]
        self.mask_paths = [mask_map[k] for k in common]
        self.depth_paths: list[Path | None] = [None] * len(common)
        self.depth_missing_stems: list[str] = []
        self.depth_matched_count = 0
        if depth_dir is not None:
            depth_map = self._scan(depth_dir, {"pt", "pth"})
            self.depth_paths = [depth_map.get(k) for k in common]
            self.depth_missing_stems = [
                k for k, path in zip(common, self.depth_paths) if path is None
            ]
            self.depth_matched_count = len(common) - len(self.depth_missing_stems)
            if require_depth and self.depth_missing_stems:
                preview = ", ".join(self.depth_missing_stems[:5])
                raise ValueError(
                    f"depth cache 缺失 {len(self.depth_missing_stems)} 个样本，"
                    f"depth_dir={depth_dir}，示例: {preview}"
                )

    @staticmethod
    def _scan(directory: Path, extensions: Iterable[str]) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower().lstrip(".") in extensions:
                result[path.stem] = path
        return result

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        orig_size = torch.tensor((mask.height, mask.width), dtype=torch.long)
        file_name = self.image_paths[idx].stem
        depth_tensor = None
        depth_valid = None
        if self.depth_dir is not None:
            depth_path = self.depth_paths[idx]
            if depth_path is None:
                if self.require_depth:
                    raise FileNotFoundError(f"样本 {file_name} 缺少 depth cache")
            else:
                depth_tensor = self._load_depth_tensor(depth_path)
                depth_valid = torch.isfinite(depth_tensor).to(torch.float32)
                depth_tensor = torch.nan_to_num(depth_tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if depth_tensor is None:
            image, mask = self.transforms(image, mask)
        else:
            image, mask, depth_tensor, depth_valid = self.transforms(
                image, mask, depth_tensor, depth_valid
            )
        image_tensor = self._prepare_image_tensor(image)
        mask_tensor = self._ensure_long_mask(mask)
        mask_tensor = normalize_binary_mask(mask_tensor, ignore_index=self.ignore_index)
        if depth_tensor is None:
            return image_tensor, mask_tensor, idx, orig_size, file_name
        depth_tensor = self._ensure_depth_tensor(depth_tensor)
        depth_valid = self._ensure_depth_tensor(depth_valid).clamp(0, 1)
        return image_tensor, mask_tensor, idx, orig_size, file_name, depth_tensor, depth_valid

    @staticmethod
    def _load_depth_tensor(path: Path) -> torch.Tensor:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        depth = sample.get("depth") if isinstance(sample, dict) else sample
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim == 3 and depth.shape[0] != 1:
            depth = depth[:1]
        if depth.ndim != 3:
            raise ValueError(f"depth cache {path} 形状非法: {tuple(depth.shape)}")
        return depth.float()

    def _prepare_image_tensor(self, image: Any) -> Any:
        if isinstance(image, list):
            tensors = [self._ensure_float_tensor(img) for img in image]
            return tensors if self.keep_image_lists else tensors[0]
        return self._ensure_float_tensor(image)

    @staticmethod
    def _ensure_float_tensor(image: Any) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        return image.float()

    @staticmethod
    def _ensure_long_mask(mask: Any) -> torch.Tensor:
        tensor = torch.as_tensor(mask) if not isinstance(mask, torch.Tensor) else mask
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.long()

    @staticmethod
    def _ensure_depth_tensor(depth: Any) -> torch.Tensor:
        tensor = torch.as_tensor(depth) if not isinstance(depth, torch.Tensor) else depth
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor[:1]
        return tensor.float()


def normalize_binary_mask(
    mask: torch.Tensor, ignore_index: Optional[int] = None
) -> torch.Tensor:
    """
    将二值 PNG 掩码（0/255 等）归一化为 0/1，避免 loss 计算时遗漏类别。
    多类别掩码保持原状。
    """
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.long()
    if mask.numel() == 0:
        return mask
    ignore_mask = torch.zeros_like(mask, dtype=torch.bool)
    if ignore_index is not None:
        ignore_mask = mask == ignore_index
    valid_pixels = mask[~ignore_mask]
    if valid_pixels.numel() == 0:
        return mask
    unique_vals = torch.unique(valid_pixels)
    if unique_vals.numel() <= 2:
        non_negative = bool(torch.all(unique_vals >= 0))
        has_foreground = bool(torch.any(valid_pixels > 0))
        if non_negative and has_foreground:
            mask = mask.clone()
            mask[~ignore_mask] = (valid_pixels > 0).long()
            if ignore_index is not None:
                mask[ignore_mask] = ignore_index
            return mask
    return mask


class FeatureCacheDataset(TorchDataset):
    """读取离线缓存的特征/mask 对。"""

    def __init__(self, cache_dir: Path, *, binarize_masks: bool = False):
        self.files = sorted(cache_dir.glob("*.pt"))
        if not self.files:
            raise ValueError(f"缓存目录 {cache_dir} 为空，请先生成特征")
        self.binarize_masks = binarize_masks

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        features = sample["features"]
        mask = sample["mask"]
        if self.binarize_masks:
            mask = normalize_binary_mask(mask)
        return features, mask


def _resolve_dataset_paths(dataset_cfg: dict) -> tuple[Path, Path, Path, Path]:
    return (
        Path(dataset_cfg["train_images"]).expanduser(),
        Path(dataset_cfg["train_masks"]).expanduser(),
        Path(dataset_cfg["val_images"]).expanduser(),
        Path(dataset_cfg["val_masks"]).expanduser(),
    )


def _build_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    distributed: bool,
    collate_fn: Optional[Callable] = None,
):
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_basic_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    distributed: bool = False,
) -> DataLoader:
    return _build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        distributed=distributed,
    )


def build_segmentation_datasets_from_cfg(dataset_cfg: dict):
    transforms_cfg = dataset_cfg.get("transforms", {})
    mean = transforms_cfg.get("mean", (0.485, 0.456, 0.406))
    std = transforms_cfg.get("std", (0.229, 0.224, 0.225))
    train_tf_cfg = transforms_cfg.get("train", {})
    eval_tf_cfg = transforms_cfg.get("eval", {})
    ignore_index = dataset_cfg.get("ignore_index")

    train_transforms = make_segmentation_train_transforms(
        img_size=train_tf_cfg.get("img_size", 256),
        random_img_size_ratio_range=train_tf_cfg.get(
            "random_img_size_ratio_range", [0.5, 2.0]
        ),
        crop_size=train_tf_cfg.get("crop_size", [256, 256]),
        flip_prob=train_tf_cfg.get("flip_prob", 0.5),
        mean=mean,
        std=std,
        ignore_index=ignore_index,
    )
    eval_img_size = eval_tf_cfg.get("img_size", 256)
    if isinstance(eval_img_size, int):
        eval_img_size = [eval_img_size, eval_img_size]
    eval_transforms = make_segmentation_eval_transforms(
        img_size=eval_img_size,
        inference_mode=eval_tf_cfg.get("inference_mode", "slide"),
        use_tta=eval_tf_cfg.get("use_tta", False),
        tta_ratios=eval_tf_cfg.get("tta_ratios", [1.0]),
        mean=mean,
        std=std,
    )

    train_images, train_masks, val_images, val_masks = _resolve_dataset_paths(
        dataset_cfg
    )
    train_depths = dataset_cfg.get("train_depths")
    val_depths = dataset_cfg.get("val_depths")
    train_depth_dir = Path(train_depths).expanduser() if train_depths else None
    val_depth_dir = Path(val_depths).expanduser() if val_depths else None
    train_dataset = SegmentationFolderDataset(
        train_images,
        train_masks,
        train_transforms,
        ignore_index=ignore_index,
        depth_dir=train_depth_dir,
        require_depth=train_depth_dir is not None,
    )
    val_dataset = SegmentationFolderDataset(
        val_images,
        val_masks,
        eval_transforms,
        ignore_index=ignore_index,
        depth_dir=val_depth_dir,
        require_depth=val_depth_dir is not None,
    )
    train_dataset = _maybe_apply_few_shot_subset(train_dataset, dataset_cfg)
    return train_dataset, val_dataset


def _parse_category_species_method(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    category = parts[1] if len(parts) > 1 else "UnknownCategory"
    species = parts[2] if len(parts) > 2 else "UnknownSpecies"
    method = parts[3] if len(parts) > 3 else "UnknownMethod"
    return category, species, method


def _balanced_few_shot_indices(
    dataset: SegmentationFolderDataset,
    *,
    num_samples: int,
    seed: int,
    strategy: str,
) -> tuple[list[int], list[str]]:
    if strategy != "category_species_method":
        raise ValueError(
            "few_shot.strategy 目前仅支持 'category_species_method'"
        )
    if num_samples not in {20, 50, 100, 200}:
        raise ValueError("few_shot.num_samples 仅支持 200/100/50/20")
    if num_samples <= 0:
        raise ValueError("few_shot.num_samples 必须为正整数")

    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[int]] = {}
    stems: list[str] = []
    for idx, image_path in enumerate(dataset.image_paths):
        stem = image_path.stem
        stems.append(stem)
        groups.setdefault(_parse_category_species_method(stem), []).append(idx)

    if num_samples >= len(stems):
        selected_indices = list(range(len(stems)))
        return selected_indices, [stems[idx] for idx in selected_indices]

    for indices in groups.values():
        rng.shuffle(indices)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    selected: list[int] = []
    selected_set: set[int] = set()

    while len(selected) < num_samples:
        progressed = False
        for key in group_keys:
            if len(selected) >= num_samples:
                break
            candidates = groups[key]
            while candidates and candidates[0] in selected_set:
                candidates.pop(0)
            if not candidates:
                continue
            idx = candidates.pop(0)
            selected.append(idx)
            selected_set.add(idx)
            progressed = True
        if not progressed:
            break

    if len(selected) < num_samples:
        remaining = [idx for idx in range(len(stems)) if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_samples - len(selected)])

    selected = selected[:num_samples]
    return selected, [stems[idx] for idx in selected]


def _maybe_apply_few_shot_subset(
    train_dataset: SegmentationFolderDataset,
    dataset_cfg: dict,
):
    few_shot_cfg = dataset_cfg.get("few_shot") or {}
    if not bool(few_shot_cfg.get("enabled", False)):
        return train_dataset

    num_samples = int(few_shot_cfg.get("num_samples", 0))
    seed = int(few_shot_cfg.get("seed", 0))
    strategy = str(few_shot_cfg.get("strategy", "category_species_method"))
    indices, selected_stems = _balanced_few_shot_indices(
        train_dataset,
        num_samples=num_samples,
        seed=seed,
        strategy=strategy,
    )
    subset = Subset(train_dataset, indices)
    subset.few_shot_enabled = True
    subset.few_shot_full_train_size = len(train_dataset)
    subset.few_shot_num_samples = len(indices)
    subset.few_shot_seed = seed
    subset.few_shot_strategy = strategy
    subset.few_shot_selected_indices = indices
    subset.few_shot_selected_stems = selected_stems
    subset.depth_dir = getattr(train_dataset, "depth_dir", None)
    subset.depth_matched_count = sum(
        1
        for idx in indices
        if getattr(train_dataset, "depth_paths", [None] * len(train_dataset))[idx]
        is not None
    )
    subset.depth_missing_stems = [
        train_dataset.image_paths[idx].stem
        for idx in indices
        if getattr(train_dataset, "depth_paths", [None] * len(train_dataset))[idx]
        is None
    ]
    return subset


def build_segmentation_image_dataloaders_from_cfg(
    dataset_cfg: dict,
    training_cfg: dict,
    device_type: str,
    *,
    distributed: bool = False,
):
    train_dataset, val_dataset = build_segmentation_datasets_from_cfg(dataset_cfg)

    pin_memory = device_type == "cuda"
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 2),
        num_workers=training_cfg.get("num_workers", 4),
        shuffle=training_cfg.get("shuffle_train", True),
        pin_memory=pin_memory,
        distributed=distributed,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=training_cfg.get("val_batch_size", 1),
        num_workers=max(1, training_cfg.get("num_workers", 4) // 2),
        shuffle=False,
        pin_memory=pin_memory,
        distributed=distributed,
    )
    return train_loader, val_loader


def build_cached_feature_dataloader(
    cache_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    distributed: bool = False,
    binarize_masks: bool = False,
):
    dataset = FeatureCacheDataset(cache_dir, binarize_masks=binarize_masks)
    return _build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        distributed=distributed,
    )


class Dataset(TorchDataset):
    """
    数据集加载类（图片 + mask）。

    Args:
        base_dir: 数据根目录，例如 "/home/mil201/hong/MAS/data/dataset"
        dataset_name: 子数据集名称，例如 "MAS3K/test"
        transform: 可选预处理函数，函数签名 transform(img, mask)
    """

    def __init__(
        self, base_dir: str, dataset_name: str, transform: Optional[Callable] = None
    ):
        self.img_dir = Path(base_dir) / dataset_name / "test/Image"
        self.mask_dir = Path(base_dir) / dataset_name / "test/Mask"

        if not self.img_dir.exists():
            raise ValueError(f"Image folder {self.img_dir} does not exist")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask folder {self.mask_dir} does not exist")

        self.img_paths = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ]
        )
        self.mask_paths = sorted(
            [p for p in self.mask_dir.iterdir() if p.suffix.lower() in [".png", ".jpg"]]
        )

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image and mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}"
            )

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 读取图片和 mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道 mask

        # 数据增强/预处理
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = make_transform()(img)

        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.to(torch.float32)

        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # [1,H,W]

        return img_tensor, mask_tensor, str(img_path)

    def get_size(self, idx: int):
        """
        获取数据集中第 idx 张图像的原始 (h, w)，返回 int。
        """
        img_path = self.img_paths[idx]
        with Image.open(img_path) as img:
            w, h = img.size
        return int(h), int(w)


class InfiniteDualDataloader:
    """单 DataLoader 版本，针对同一 batch 生成 (raw, enhance) 对。"""

    def __init__(
        self,
        dataset,
        enhance_transform: Callable,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        max_steps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.enhance_transform = enhance_transform
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )
        self.loader_iter = iter(self.loader)
        self.max_steps = max_steps
        self.step = 0

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.max_steps is not None and self.step >= self.max_steps:
            raise StopIteration

        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)

        imgs, masks, paths = batch
        raw_batch = (imgs, masks, paths)
        enhance_imgs = self._build_enhance_view(paths)
        enhance_batch = (enhance_imgs, masks, paths)

        self.step += 1
        return raw_batch, enhance_batch

    def __len__(self):
        return len(self.loader)

    def _build_enhance_view(self, paths):
        tensors = []
        for path in paths:
            with Image.open(path).convert("RGB") as img:
                tensors.append(self.enhance_transform(img))
        return torch.stack(tensors)


class ImagesOnlyDataset(TorchDataset):

    def __init__(
        self,
        base_dir: str,
        *,
        transform: Optional[Callable[[Image.Image], dict]] = None,
        recursive: bool = False,
        cache_images: bool = False,
        cache_max_images: int | None = None,
    ) -> None:
        self.root = Path(base_dir)
        if not self.root.exists():
            raise ValueError(f"Dataset root {self.root} does not exist")

        # transform 会返回多尺度裁剪与 teacher/student 视角，后续在 collate_fn 中打包
        self.transform = transform
        self.recursive = recursive
        self.cache_images = cache_images
        self.cache_max_images = cache_max_images
        self._image_cache: OrderedDict[int, Image.Image] = OrderedDict()
        self.img_paths = self._scan_images(self.root)
        if not self.img_paths:
            raise ValueError(f"No image files found under {self.root}")

    def _scan_images(self, directory: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        if self.recursive:
            iterator = directory.rglob("*")
        else:
            iterator = directory.iterdir()
        return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.img_paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path = self.img_paths[idx]
        if self.cache_images:
            cached = self._image_cache.get(idx)
            if cached is None:
                with Image.open(img_path) as img:
                    cached = img.convert("RGB").copy()
                self._image_cache[idx] = cached
                if (
                    self.cache_max_images is not None
                    and len(self._image_cache) > self.cache_max_images
                ):
                    self._image_cache.popitem(last=False)
            else:
                self._image_cache.move_to_end(idx)
            img = self._image_cache[idx].copy()
        else:
            with Image.open(img_path) as img_obj:
                img = img_obj.convert("RGB")

        if self.transform is None:
            raise RuntimeError(
                "ImagesOnlyDataset requires a transform that returns multi-view crops"
            )
        sample = self.transform(img)
        metadata = {"path": str(img_path)}
        return sample, metadata


def build_ssl_dataloader(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    ssl_cfg: dict,
    patch_size: int,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
    distributed: bool = False,
    num_replicas: int | None = None,
    rank: int | None = None,
) -> DataLoader:
    """Construct a DINOv3-style multi-crop loader for self-supervised pretraining."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    data_cfg = ssl_cfg.get("data", {}) if ssl_cfg else {}

    # 多尺度裁剪超参直接复用 DINOv3，默认读取 YAML 中的 ssl.data
    global_crops_size = data_cfg.get(
        "global_crops_size", ssl_cfg.get("global_crops_size", 224)
    )
    local_crops_size = data_cfg.get("local_crops_size", 96)
    local_crops_number = data_cfg.get("local_crops_number", 6)
    global_crops_scale = tuple(data_cfg.get("global_crops_scale", (0.4, 1.0)))
    local_crops_scale = tuple(data_cfg.get("local_crops_scale", (0.05, 0.4)))
    share_color_jitter = bool(data_cfg.get("share_color_jitter", False))
    horizontal_flips = bool(data_cfg.get("horizontal_flips", True))
    cache_images = bool(data_cfg.get("cache_images", False))
    cache_max_images = data_cfg.get("cache_max_images")
    if cache_max_images is not None:
        cache_max_images = int(cache_max_images)
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = (
        bool(data_cfg.get("persistent_workers", True)) if num_workers > 0 else False
    )
    prefetch_factor = (
        int(data_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None
    )

    augmentation = DataAugmentationDINO(
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        local_crops_number=local_crops_number,
        global_crops_size=global_crops_size,
        local_crops_size=local_crops_size,
        share_color_jitter=share_color_jitter,
        horizontal_flips=horizontal_flips,
    )

    dataset = ImagesOnlyDataset(
        base_dir=data_dir,
        transform=augmentation,
        cache_images=cache_images,
        recursive=bool(data_cfg.get("recursive", False)),
        cache_max_images=cache_max_images,
    )

    input_size = global_crops_size // patch_size
    # patch 级别遮挡生成器，用于 iBOT 遮蔽监督
    mask_generator = MaskingGenerator(
        input_size=(input_size, input_size),
        max_num_patches=int(0.5 * input_size * input_size),
    )

    mask_ratio_tuple = tuple(data_cfg.get("mask_ratio_min_max", (0.1, 0.5)))
    mask_probability = float(data_cfg.get("mask_probability", 0.7))
    random_circular_shift = bool(data_cfg.get("mask_random_circular_shift", False))

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=mask_ratio_tuple,
        mask_probability=mask_probability,
        dtype=dtype,
        n_tokens=input_size * input_size,
        mask_generator=mask_generator,
        random_circular_shift=random_circular_shift,
    )

    wavelet_cfg = ssl_cfg.get("wavelet_swap", {})
    wavelet_ratio = float(wavelet_cfg.get("ratio", 0.0))
    if wavelet_ratio > 0:
        clamp_min = wavelet_cfg.get("clamp_min", 0.0)
        clamp_max = wavelet_cfg.get("clamp_max", 1.0)
        clamp_tuple = (
            None if wavelet_cfg.get("clamp_disable", False) else (clamp_min, clamp_max)
        )
        local_swap_count = int(wavelet_cfg.get("local_swap_count"))
        swapper = BatchWaveletSwapper(
            swap_ratio=wavelet_ratio,
            swap_crop_index=int(wavelet_cfg.get("swap_crop_index", 0)),
            wavelet=wavelet_cfg.get("wavelet", "haar"),
            mode=wavelet_cfg.get("mode", "reflect"),
            clamp=clamp_tuple,
            levels=int(wavelet_cfg.get("levels", 1)),
            local_swap_count=local_swap_count,
            base_transform=augmentation,
        )

        base_collate_fn = collate_fn

        def collate_with_wavelet(samples_list):
            swapper(samples_list)
            return base_collate_fn(samples_list)

        collate_fn = collate_with_wavelet

    sampler = None
    if distributed:
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "persistent_workers": persistent_workers,
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "sampler": sampler,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)

    loader = DataLoader(**loader_kwargs)
    return loader
