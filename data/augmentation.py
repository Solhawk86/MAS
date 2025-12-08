import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2
import pywt  # type: ignore



logger = logging.getLogger("mas.data.augmentation")

_TO_FLOAT_IMAGE = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

__all__ = ["swap_ll_components", "WaveletLLSwap", "run_wavelet_ll_swap_demo"]


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
    print(f"max_level: {max_level}")
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
    ):
        self.wavelet = wavelet
        self.mode = mode
        self.clamp = clamp
        self.levels = levels

    def __call__(self, image_a: Any, image_b: Any) -> tuple[Tensor, Tensor]:
        tensor_a, device_a = _ensure_chw_tensor(image_a)
        tensor_b, device_b = _ensure_chw_tensor(image_b)
        swapped_a, swapped_b = swap_ll_components(
            tensor_a,
            tensor_b,
            wavelet=self.wavelet,
            mode=self.mode,
            clamp=self.clamp,
            levels=self.levels,
        )
        return swapped_a.to(device_a), swapped_b.to(device_b)


def run_wavelet_ll_swap_demo(
    image_path_a: str | Path,
    image_path_b: str | Path,
    output_dir: str | Path = "wavelet_ll_swap_outputs",
    *,
    levels: int = 1,
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

    swap_transform = WaveletLLSwap(levels=levels)
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
        default=9,
        help="小波分解层数，>=1",
    )
    args = parser.parse_args()
    image1 = "/mnt/mydata/hong/RMAS/train/img/1.jpg"
    image2 = "/mnt/mydata/hong/RMAS/train/img/2.jpg"
    path_a, path_b = run_wavelet_ll_swap_demo(
        image_path_a=image1,
        image_path_b=image2,
        output_dir=args.output_dir,
        levels=args.levels,
    )
    print(f"保存结果: {path_a}")
    print(f"保存结果: {path_b}")
