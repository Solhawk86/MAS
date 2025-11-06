import random
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator

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
    '''
    mask 与特征 patch 对齐，保证预测 mask 与 GT 尺寸一致
    '''
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

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

class Dataset(TorchDataset):
    """
    数据集加载类（图片 + mask）。

    Args:
        base_dir: 数据根目录，例如 "/home/mil201/hong/MAS/data/dataset"
        dataset_name: 子数据集名称，例如 "MAS3K/test"
        transform: 可选预处理函数，函数签名 transform(img, mask)
    """

    def __init__(self, base_dir: str, dataset_name: str, transform: Optional[Callable] = None):
        self.img_dir = Path(base_dir) / dataset_name / "test/Image"
        self.mask_dir = Path(base_dir) / dataset_name / "test/Mask"

        if not self.img_dir.exists():
            raise ValueError(f"Image folder {self.img_dir} does not exist")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask folder {self.mask_dir} does not exist")

        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
        self.mask_paths = sorted([p for p in self.mask_dir.iterdir() if p.suffix.lower() in [".png", ".jpg"]])

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Image and mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}")

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
                if self.cache_max_images is not None and len(self._image_cache) > self.cache_max_images:
                    self._image_cache.popitem(last=False)
            else:
                self._image_cache.move_to_end(idx)
            img = self._image_cache[idx].copy()
        else:
            with Image.open(img_path) as img_obj:
                img = img_obj.convert("RGB")

        if self.transform is None:
            raise RuntimeError("ImagesOnlyDataset requires a transform that returns multi-view crops")
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
    global_crops_size = data_cfg.get("global_crops_size", ssl_cfg.get("global_crops_size", 224))
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
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) if num_workers > 0 else False
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None

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

    sampler = None
    if distributed:
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = torch.cuda.device_count() if torch.cuda.is_available() else 1
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
