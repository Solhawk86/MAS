import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import numpy as np
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

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

        return img_tensor, mask_tensor, img_path.name

    def get_size(self, idx: int):
        """
        获取数据集中第 idx 张图像的原始 (h, w)，返回 int。
        """
        img_path = self.img_paths[idx]
        with Image.open(img_path) as img:
            w, h = img.size
        return int(h), int(w)
    
class InfiniteDualDataloader:
    """
    支持两路输入的无限循环 DataLoader。
    每次返回 (img_raw, img_aug)，保证对应索引配对。
    """
    def __init__(
        self,
        raw_dataset,
        enhance_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        max_steps: Optional[int] = None,
    ):
        assert len(raw_dataset) == len(enhance_dataset), "两个数据集长度必须一致"
        self.raw_dataset = raw_dataset
        self.enhance_dataset = enhance_dataset

        # 创建单步 DataLoader，shuffle 可以控制每个 epoch 是否打乱
        self.raw_loader = DataLoader(
            raw_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True
        )
        self.enhance_loader = DataLoader(
            enhance_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True
        )

        self.raw_iter = iter(self.raw_loader)
        self.enhance_iter = iter(self.enhance_loader)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.step = 0

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.max_steps is not None and self.step >= self.max_steps:
            raise StopIteration

        try:
            raw_batch = next(self.raw_iter)
        except StopIteration:
            self.raw_iter = iter(self.raw_loader)
            raw_batch = next(self.raw_iter)

        try:
            enhance_batch = next(self.enhance_iter)
        except StopIteration:
            self.enhance_iter = iter(self.enhance_loader)
            enhance_batch = next(self.enhance_iter)

        self.step += 1
        return raw_batch, enhance_batch

    def __len__(self):
        return len(self.raw_loader)