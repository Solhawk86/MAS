import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2
try:
    import pywt  # type: ignore
except ImportError:  # pragma: no cover
    pywt = None



logger = logging.getLogger("mas.data.augmentation")

_TO_FLOAT_IMAGE = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

__all__ = [
    "swap_ll_components",
    "WaveletLLSwap",
    "run_wavelet_ll_swap_demo",
    "WaveletGlobalCropsWrapper",
    "BatchWaveletSwapper",
]


def _ensure_chw_tensor(image: Any) -> tuple[Tensor, torch.device]:
    """
    将任意输入转换为 float32 CHW tensor，记录原始设备。
    """
    if isinstance(image, torch.Tensor):
        device = image.device
        tensor = image.detach().clone()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1).contiguous()
        elif tensor.ndim != 3:
            raise ValueError(f"Expect image tensor with shape [C,H,W] or [H,W], got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
            if tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = tensor.to(torch.float32)
            if tensor.max().item() > 1.5:
                logger.debug("Input tensor has values >1.0, skipping normalization.")
        return tensor, device

    tensor = _TO_FLOAT_IMAGE(image)
    return tensor, torch.device("cpu")


def _reconstruct_channel(
    coeffs: Any,
    ref_hw: tuple[int, int],
    *,
    wavelet: str,
    mode: str,
) -> np.ndarray:
    """
    根据给定的小波系数（可多层）重建单个通道，并裁剪到参考尺寸。
    """
    channel = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
    if channel.shape != ref_hw:
        channel = channel[: ref_hw[0], : ref_hw[1]]
    return channel


def swap_ll_components(
    image_a: Tensor,
    image_b: Tensor,
    *,
    wavelet: str = "haar",
    mode: str = "reflect",
    clamp: tuple[float, float] | None = (0.0, 1.0),
    levels: int = 1,
    override_levels: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    对两张图片进行二维离散小波变换，并交换它们的 LL 低频分量。

    Args:
        image_a, image_b: 输入 tensor，要求均为 [C, H, W]，并且尺寸一致。
        wavelet: pywt 支持的小波名称，例如 'haar'、'db2'。
        mode: 边界填充模式，参考 pywt.dwt2。
        clamp: 若不为 None，则在返回前对结果进行裁剪，默认为 (0, 1)。
        levels: 小波分解层数，>=1。数值越大，交换越“深层”的低频语义。

    Returns:
        交换 LL 分量后的图像对 (image_a_swapped, image_b_swapped)。
    """
    if pywt is None:
        raise ImportError("PyWavelets is required for swap_ll_components. Install it via `pip install pywavelets`.")

    if image_a.shape != image_b.shape:
        raise ValueError(f"Both images must share the same shape. Got {tuple(image_a.shape)} vs {tuple(image_b.shape)}.")
    if image_a.ndim != 3:
        raise ValueError(f"swap_ll_components expects CHW tensors. Got {tuple(image_a.shape)}.")

    if levels < 1:
        raise ValueError("levels must be >= 1")

    h, w = image_a.shape[1], image_a.shape[2]
    numpy_a = image_a.detach().cpu().to(torch.float32).numpy()
    numpy_b = image_b.detach().cpu().to(torch.float32).numpy()

    max_level = pywt.dwtn_max_level((h, w), wavelet)
    if override_levels is not None:
        level_to_use = max(1, min(override_levels, max_level if max_level > 0 else 1))
    else:
        if max_level == 0:
            level_to_use = 1
        else:
            level_to_use = min(levels, max_level)

    swapped_a = []
    swapped_b = []
    for channel_a, channel_b in zip(numpy_a, numpy_b):
        coeffs_a = pywt.wavedec2(channel_a, wavelet=wavelet, mode=mode, level=level_to_use)
        coeffs_b = pywt.wavedec2(channel_b, wavelet=wavelet, mode=mode, level=level_to_use)

        new_coeffs_a = [coeffs_b[0]] + list(coeffs_a[1:])
        new_coeffs_b = [coeffs_a[0]] + list(coeffs_b[1:])

        recon_a = _reconstruct_channel(new_coeffs_a, (h, w), wavelet=wavelet, mode=mode)
        recon_b = _reconstruct_channel(new_coeffs_b, (h, w), wavelet=wavelet, mode=mode)
        swapped_a.append(recon_a)
        swapped_b.append(recon_b)

    swapped_a_tensor = torch.from_numpy(np.stack(swapped_a)).to(image_a.dtype)
    swapped_b_tensor = torch.from_numpy(np.stack(swapped_b)).to(image_b.dtype)

    if clamp is not None:
        low, high = clamp
        swapped_a_tensor = swapped_a_tensor.clamp(low, high)
        swapped_b_tensor = swapped_b_tensor.clamp(low, high)

    return swapped_a_tensor, swapped_b_tensor


class WaveletLLSwap:
    """
    PyTorch/torchvision 兼容的 transform，用于交换一对图像的低频 LL 分量。

    Example:
        >>> augment = WaveletLLSwap()
        >>> view1, view2 = augment(image_a, image_b)
    """

    def __init__(
        self,
        wavelet: str = "haar",
        mode: str = "reflect",
        clamp: tuple[float, float] | None = (0.0, 1.0),
        levels: int = 1,
        couple_details: bool = True,
    ):
        self.wavelet = wavelet
        self.mode = mode
        self.clamp = clamp
        self.levels = levels
        self.couple_details = couple_details

    def __call__(self, image_a: Any, image_b: Any, *, override_levels: int | None = None) -> tuple[Tensor, Tensor]:
        tensor_a, device_a = _ensure_chw_tensor(image_a)
        tensor_b, device_b = _ensure_chw_tensor(image_b)
        swapped_a, swapped_b = swap_ll_components(
            tensor_a,
            tensor_b,
            wavelet=self.wavelet,
            mode=self.mode,
            clamp=self.clamp,
            levels=self.levels,
            override_levels=override_levels,
        )
        return swapped_a.to(device_a), swapped_b.to(device_b)

    def max_decomposition_level(self, hw: tuple[int, int]) -> int:
        if pywt is None:
            raise ImportError("PyWavelets is required.")
        max_level = pywt.dwtn_max_level(hw, self.wavelet)
        return max_level if max_level > 0 else 1

class BatchWaveletSwapper:
    """
    在 collate 前的 samples_list 上按比例随机抽取样本，两两配对并对指定 global crop 做 LL 交换。
    仅影响 student 视角，teacher 视角保持不变。
    """

    def __init__(
        self,
        *,
        swap_ratio: float = 0.5,
        swap_crop_index: int = 0,
        local_swap_count: int = 0,
        local_swap_resize: int | None = 128,
        wavelet: str = "haar",
        mode: str = "reflect",
        clamp: tuple[float, float] | None = (0.0, 1.0),
        levels: int = 1,
        base_transform: Any | None = None,
    ):
        if not 0.0 <= swap_ratio <= 1.0:
            raise ValueError("swap_ratio must be in [0, 1].")
        if swap_crop_index < 0:
            raise ValueError("swap_crop_index must be >= 0.")
        self.swap_ratio = swap_ratio
        self.swap_crop_index = swap_crop_index
        self.local_swap_count = max(0, int(local_swap_count))
        self.local_swap_resize = int(local_swap_resize) if local_swap_resize is not None else None
        if base_transform is None:
            raise ValueError("BatchWaveletSwapper requires base_transform when enabling wavelet swaps.")
        self.swap_transform = WaveletLLSwap(
            wavelet=wavelet,
            mode=mode,
            clamp=clamp,
            levels=levels,
        )
        self.base_transform = base_transform
        self.last_swapped_pairs: list[tuple[int, int]] = []
        self.to_pil = v2.ToPILImage()
        self._resize_cache: dict[tuple[int, int], v2.Resize] = {}

    def __call__(self, samples_list: list[tuple[dict, Any]]) -> None:
        if pywt is None:
            raise ImportError("PyWavelets is required for BatchWaveletSwapper. 请先安装 pywt。")
        if not samples_list:
            return
        self.last_swapped_pairs.clear()
        num_samples = len(samples_list)
        if num_samples < 2 or self.swap_ratio <= 0.0:
            return

        target = int(num_samples * self.swap_ratio)
        if target < 2:
            return
        if target % 2 == 1:
            target -= 1
        if target < 2:
            return

        indices = list(range(num_samples))
        random.shuffle(indices)
        selected = indices[:target]
        if not selected:
            return

        for i in range(0, len(selected), 2):
            idx_a, idx_b = selected[i], selected[i + 1]
            sample_a = samples_list[idx_a][0]
            sample_b = samples_list[idx_b][0]
            self._swap_pair(sample_a, sample_b)
            self.last_swapped_pairs.append((idx_a, idx_b))

    def _swap_pair(self, sample_a: dict, sample_b: dict) -> None:
        swapped_globals, teacher_bases = self._swap_global_crop(sample_a, sample_b)
        teacher_base_a, teacher_base_b = teacher_bases
        swapped_locals = self._swap_local_crops(sample_a, sample_b)
        if not swapped_globals and swapped_locals == 0:
            return
        if self.base_transform is not None:
            self._rebuild_student_views(sample_a, swapped_locals, teacher_base=teacher_base_a)
            self._rebuild_student_views(sample_b, swapped_locals, teacher_base=teacher_base_b)
        else:
            crops_a = sample_a.get("global_crops")
            crops_b = sample_b.get("global_crops")
            if (
                isinstance(crops_a, list)
                and isinstance(crops_b, list)
                and len(crops_a) > self.swap_crop_index
                and len(crops_b) > self.swap_crop_index
            ):
                bases_a = sample_a.get("_global_bases")
                bases_b = sample_b.get("_global_bases")
                if (
                    isinstance(bases_a, list)
                    and isinstance(bases_b, list)
                    and len(bases_a) > self.swap_crop_index
                    and len(bases_b) > self.swap_crop_index
                ):
                    crops_a[self.swap_crop_index] = _ensure_chw_tensor(bases_a[self.swap_crop_index])[0]
                    crops_b[self.swap_crop_index] = _ensure_chw_tensor(bases_b[self.swap_crop_index])[0]

    def _swap_global_crop(self, sample_a: dict, sample_b: dict) -> tuple[bool, tuple[Any | None, Any | None]]:
        crops_a = sample_a.get("global_crops")
        crops_b = sample_b.get("global_crops")
        if (
            not isinstance(crops_a, list)
            or not isinstance(crops_b, list)
            or len(crops_a) <= self.swap_crop_index
            or len(crops_b) <= self.swap_crop_index
        ):
            return False, (None, None)
        bases_a = sample_a.get("_global_bases")
        bases_b = sample_b.get("_global_bases")
        if (
            not isinstance(bases_a, list)
            or not isinstance(bases_b, list)
            or len(bases_a) <= self.swap_crop_index
            or len(bases_b) <= self.swap_crop_index
        ):
            return False, (None, None)
        base_a = bases_a[self.swap_crop_index]
        base_b = bases_b[self.swap_crop_index]
        teacher_base_a = self._clone_base_object(base_a)
        teacher_base_b = self._clone_base_object(base_b)
        tensor_a, _ = _ensure_chw_tensor(base_a)
        tensor_b, _ = _ensure_chw_tensor(base_b)
        max_level_global = self.swap_transform.max_decomposition_level((tensor_a.shape[1], tensor_a.shape[2]))
        level_global = min(max_level_global, self.swap_transform.levels)
        swapped_a, swapped_b = self.swap_transform(tensor_a, tensor_b, override_levels=level_global)
        swapped_a_pil = self.to_pil(swapped_a.cpu())
        swapped_b_pil = self.to_pil(swapped_b.cpu())
        bases_a[self.swap_crop_index] = swapped_a_pil
        bases_b[self.swap_crop_index] = swapped_b_pil

        return True, (teacher_base_a, teacher_base_b)

    def _swap_local_crops(self, sample_a: dict, sample_b: dict) -> int:
        if self.local_swap_count <= 0:
            return 0
        bases_a = sample_a.get("_local_bases")
        bases_b = sample_b.get("_local_bases")
        if not isinstance(bases_a, list) or not isinstance(bases_b, list):
            return 0
        swap_limit = min(self.local_swap_count, len(bases_a), len(bases_b))
        if swap_limit <= 0:
            return 0

        swapped = 0
        for idx in range(swap_limit):
            base_a = bases_a[idx]
            base_b = bases_b[idx]
            tensor_a, _ = _ensure_chw_tensor(base_a)
            tensor_b, _ = _ensure_chw_tensor(base_b)
            original_hw = (tensor_a.shape[1], tensor_a.shape[2])
            proc_a = tensor_a
            proc_b = tensor_b
            if self.local_swap_resize is not None and (
                original_hw[0] != self.local_swap_resize or original_hw[1] != self.local_swap_resize
            ):
                proc_a = self._resize_tensor(proc_a, (self.local_swap_resize, self.local_swap_resize))
                proc_b = self._resize_tensor(proc_b, (self.local_swap_resize, self.local_swap_resize))
            level_local = self.swap_transform.max_decomposition_level((proc_a.shape[1], proc_a.shape[2]))
            swapped_a, swapped_b = self.swap_transform(proc_a, proc_b, override_levels=level_local)
            if (proc_a.shape[1], proc_a.shape[2]) != original_hw:
                swapped_a = self._resize_tensor(swapped_a, original_hw)
                swapped_b = self._resize_tensor(swapped_b, original_hw)
            bases_a[idx] = self.to_pil(swapped_a.cpu())
            bases_b[idx] = self.to_pil(swapped_b.cpu())
            swapped += 1
        return swapped

    def _rebuild_student_views(self, sample: dict, swapped_local_count: int, *, teacher_base: Any | None = None) -> None:
        self.base_transform.rebuild_sample(sample, skip_color_aug=True)
        if teacher_base is not None:
            self._restore_teacher_global(sample, teacher_base)
        if swapped_local_count <= 0:
            return
        self._restore_remaining_local_color_aug(sample, swapped_local_count)

    def _restore_teacher_global(self, sample: dict, teacher_base: Any) -> None:
        teacher_crops = sample.get("global_crops_teacher")
        if not isinstance(teacher_crops, list) or len(teacher_crops) <= self.swap_crop_index:
            return
        normalize = getattr(self.base_transform, "normalize", None)
        if normalize is None:
            return
        clone_base = getattr(self.base_transform, "_clone_base", None)
        if clone_base is None:
            clone_base = self._clone_base_object
        teacher_crops[self.swap_crop_index] = normalize(clone_base(teacher_base))

    def _restore_remaining_local_color_aug(self, sample: dict, swapped_local_count: int) -> None:
        local_transfo = getattr(self.base_transform, "local_transfo", None)
        if local_transfo is None:
            return
        local_bases = sample.get("_local_bases")
        local_crops = sample.get("local_crops")
        if not isinstance(local_bases, list) or not isinstance(local_crops, list):
            return
        total = min(len(local_bases), len(local_crops))
        if swapped_local_count >= total:
            return
        for idx in range(swapped_local_count, total):
            base = local_bases[idx]
            if base is None:
                continue
            local_crops[idx] = local_transfo(self._clone_base_object(base))

    def _resize_tensor(self, tensor: Tensor, size: tuple[int, int]) -> Tensor:
        resize = self._resize_cache.get(size)
        if resize is None:
            resize = v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
            self._resize_cache[size] = resize
        return resize(tensor)

    @staticmethod
    def _clone_base_object(base: Any) -> Any:
        if hasattr(base, "copy"):
            return base.copy()
        if torch.is_tensor(base):
            return base.clone()
        return base




def run_wavelet_ll_swap_demo(
    image_path_a: str | Path,
    image_path_b: str | Path,
    output_dir: str | Path = "wavelet_ll_swap_outputs",
    *,
    levels: int = 1,
    couple_details: bool = True,
) -> tuple[Path, Path]:
    """
    简易测试单元：读取两张图片，resize 到 512x512，执行 WaveletLLSwap，并将结果保存到本地。
    """
    if pywt is None:
        raise ImportError("PyWavelets is required for this demo. 请先安装 pywt。")

    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((512, 512), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    to_pil = v2.ToPILImage()

    image_a = preprocess(Image.open(image_path_a).convert("RGB"))
    image_b = preprocess(Image.open(image_path_b).convert("RGB"))

    swap_transform = WaveletLLSwap(levels=levels, couple_details=couple_details)
    swapped_a, swapped_b = swap_transform(image_a, image_b)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_a_path = output_dir / "swap_result_a.png"
    output_b_path = output_dir / "swap_result_b.png"

    to_pil(swapped_a.detach().cpu()).save(output_a_path)
    to_pil(swapped_b.detach().cpu()).save(output_b_path)

    logger.info("Wavelet LL swap demo outputs saved to %s and %s", output_a_path, output_b_path)
    return output_a_path, output_b_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wavelet LL swap transform demo.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/mydata/hong/wavelet_outputs",
        help="保存结果的目录（默认: wavelet_ll_swap_outputs）",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="小波分解层数，>=1",
    )
    parser.add_argument(
        "--decouple-details",
        action="store_true",
        help="仅交换 LL 系数，不同步同层级的细节系数（可能出现棋盘状边界）",
    )
    args = parser.parse_args()
    image1 = "/mnt/mydata/hong/RMAS_small/test/img/50.jpg"
    image2 = "/mnt/mydata/hong/RMAS/train/img/300.jpg"
    start_time = time.time()
    path_a, path_b = run_wavelet_ll_swap_demo(
        image_path_a=image1,
        image_path_b=image2,
        output_dir=args.output_dir,
        levels=args.levels,
        couple_details=not args.decouple_details,
    )
    print(f"运行时间: {time.time() - start_time:.3f}s")
    print(f"保存结果: {path_a}")
    print(f"保存结果: {path_b}")
