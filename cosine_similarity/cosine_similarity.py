#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DINOv3 模型对指定 patch 与其它 patch 的相似度进行可视化。
在服务器端运行，无需交互式界面，直接将热力图存成图片。
所有参数均从配置文件读取。
"""

import sys
from pathlib import Path
import torch.nn as nn
import matplotlib
import torch.nn.functional as F
matplotlib.use("Agg")  # 强制使用非交互式后端
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAS.model.model import DINOv3SelfSupervised
from utils import load_config


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(image_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    读取图像并转换为 ImageNet 归一化后的张量，同时返回原图尺寸。
    """
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图像文件: {image_path}")

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(img), orig_size


def ensure_output_path(image_path: Path, output_dir: str | Path | None) -> Path:
    """根据配置生成热力图输出路径，若文件已存在则递增后缀避免覆盖。"""

    if output_dir is None:
        output_dir = Path.cwd() / "cosine_similarity" / "output"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{image_path.stem}_similarity"
    candidate = out_dir / f"{stem}.png"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{counter}.png"
        counter += 1
    return candidate

def visualize_patch_similarity(
    model: nn.Module,
    img: torch.Tensor,
    *,
    patch_coords: tuple[int, int] | None = None,
    layer: int | None = None,
    cmap: str = "viridis",
    normalize_mode: str = "global",
    show: bool = True,
):
    """
    将 ViT patch 特征之间的余弦相似度可视化为热力图。

    参数:
        model: 具备 `get_intermediate_layers` 接口的模型，或包含该接口的子模块（例如 `DualImageDINOv3`）。
        img: 单张图像，shape 为 [3, H, W]，需预先按 ImageNet 均值 / 方差归一化。
        patch_coords: 指定要展示的 patch 坐标 (i, j)，默认取特征图中心位置。
        layer: 指定使用的层索引，默认取最后一层 (n=1)。
        cmap: Matplotlib 中的 colormap 名称。
        show: 是否直接使用 Matplotlib 展示热力图。

    返回:
        viz: 归一化到 [0, 1] 的 RGB 可视化张量，形状 [h, w, h, w, 3]。
        cos: 归一化到 [0, 1] 的余弦相似度张量，形状 [h, w, h, w]。
              若 normalize_mode="local"，代表按局部最小最大重新缩放。
    """
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError("输入张量需为 [3, H, W] 的单张图像。")

    # 推断模型所在设备，优先使用 model.device，其次读取参数设备
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "backbone"):
        device = getattr(model.backbone, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    img = img.unsqueeze(0).to(device)

    # 适配 DualImageDINOv3 这类封装模型
    target_module = model
    if not hasattr(target_module, "get_intermediate_layers"):
        target_module = getattr(model, "backbone", None)
    if target_module is None or not hasattr(target_module, "get_intermediate_layers"):
        raise AttributeError("传入的模型缺少 get_intermediate_layers 方法。")

    layer_selector: int | list[int]
    if layer is None:
        layer_selector = 1
    else:
        layer_selector = [int(layer)]

    with torch.no_grad():
        feats = target_module.get_intermediate_layers(
            img,
            n=layer_selector,
            reshape=True,
        )[0]

    _, dim, h, w = feats.shape
    feats = feats.squeeze(0).movedim(-3, -1).flatten(0, 1)  # [hw, dim]
    feats = F.normalize(feats, p=2, dim=-1)  # L2 归一化便于计算余弦相似度

    cos = feats @ feats.transpose(0, 1)
    cos = cos.unflatten(1, (h, w)).unflatten(0, (h, w))
    cos = (cos + 1.0) / 2.0

    mode = normalize_mode.lower()
    if mode == "local":
        cos_min = cos.amin()
        cos_max = cos.amax()
        denom = torch.clamp(cos_max - cos_min, min=1e-6)
        cos = (cos - cos_min) / denom
    elif mode not in {"global", ""}:
        raise ValueError("normalize_mode 仅支持 'global' 或 'local'")

    viz = plt.get_cmap(cmap)(cos.detach().cpu().numpy())[..., :3]

    if patch_coords is None:
        patch_coords = (h // 2, w // 2)
    i, j = patch_coords
    if not (0 <= i < h and 0 <= j < w):
        raise ValueError(f"patch 坐标 {(i, j)} 超出特征网格范围 (h={h}, w={w})。")

    if show:
        plt.imshow(viz[i, j])
        plt.title(f"Similarity of patch {(i, j)} to all other patches")

    return viz, cos.cpu()


def build_mosaic_heatmap(heatmap: np.ndarray, patch_coords: tuple[int, int], patch_size: int) -> np.ndarray:
    """将 patch 级热力图放大成马赛克并高亮选中 patch。"""

    mosaic = np.repeat(np.repeat(np.clip(heatmap, 0.0, 1.0), patch_size, axis=0), patch_size, axis=1)
    red_color = np.array([1.0, 0.0, 0.0])
    i, j = patch_coords
    y0 = i * patch_size
    x0 = j * patch_size
    mosaic[y0 : y0 + patch_size, x0 : x0 + patch_size] = red_color
    return mosaic


def main() -> None:
    cfg = load_config()
    viz_cfg = cfg.get("cosine_viz", {})

    images_cfg = viz_cfg.get("images") or viz_cfg.get("image")
    if images_cfg is None:
        raise ValueError("cosine_viz.images 未配置，请在配置文件中指定至少一张图片路径。")
    if isinstance(images_cfg, (str, Path)):
        image_paths = [Path(images_cfg)]
    elif isinstance(images_cfg, list):
        image_paths = [Path(p) for p in images_cfg]
    else:
        raise TypeError("cosine_viz.images 需为字符串或字符串列表。")

    device = viz_cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_layers = viz_cfg.get(
        "backbone_out_layers",
        cfg.get("model", {}).get("backbone_out_layers", "LAST"),
    )

    train_cfg = cfg.get("train", {})
    lora_layers = train_cfg.get("lora_layers")
    if isinstance(lora_layers, str) and lora_layers.lower() in {"none", "null", ""}:
        lora_layers = None

    model = DINOv3SelfSupervised(
        cfg=cfg,
        device=device,
        lora_layers=lora_layers,
        lora_rank=train_cfg.get("lora_rank", 8),
        lora_alpha=float(train_cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.0)),
    )
    model.eval()

    finetuned_path = cfg.get("finetuned_weights")
    if finetuned_path:
        finetuned_path = Path(finetuned_path).expanduser()
        if not finetuned_path.exists():
            raise FileNotFoundError(f"找不到 finetuned_weights 文件: {finetuned_path}")
        state = torch.load(finetuned_path, map_location="cpu")
        if isinstance(state, dict):
            if "student" in state and isinstance(state["student"], dict) and "backbone" in state["student"]:
                backbone_state = state["student"]["backbone"]
            elif "backbone" in state:
                backbone_state = state["backbone"]
            else:
                backbone_state = state
        else:
            backbone_state = state

        missing, unexpected = model.student["backbone"].load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            print(
                "[WARN] 加载 finetuned_weights 时存在不匹配字段: "
                f"missing={missing}, unexpected={unexpected}"
            )

    feature_backbone = model.student["backbone"]
    feature_backbone.eval()

    layer_cfg = viz_cfg.get("layer")
    layer = None if layer_cfg is None else int(layer_cfg)
    patch_cfg = viz_cfg.get("patch_coords")
    if patch_cfg is not None:
        if len(patch_cfg) != 2:
            raise ValueError("cosine_viz.patch_coords 需提供两个整数索引。")
        patch_coords_cfg = (int(patch_cfg[0]), int(patch_cfg[1]))
    else:
        patch_coords_cfg = None

    cmap = viz_cfg.get("cmap", "viridis")
    output_dir = viz_cfg.get("output_dir")

    resize_to_original = bool(viz_cfg.get("resize_to_original", True))
    output_scale = viz_cfg.get("output_scale")
    patch_size = int(cfg.get("student", {}).get("patch_size", 16))
    normalize_mode = viz_cfg.get("normalize", "global")
    mosaic_enabled = int(viz_cfg.get("mosaic_heatmap", 1)) != 0

    for image_path in image_paths:
        image_path = image_path.expanduser()
        img_tensor, orig_size = load_image(image_path)
        output_path = ensure_output_path(image_path, output_dir)

        with torch.no_grad():
            viz, _ = visualize_patch_similarity(
                model=feature_backbone,
                img=img_tensor,
                patch_coords=patch_coords_cfg,
                layer=layer,
                cmap=cmap,
                normalize_mode=normalize_mode,
                show=False,
            )

        if patch_coords_cfg is None:
            patch_coords = (viz.shape[0] // 2, viz.shape[1] // 2)
        else:
            patch_coords = patch_coords_cfg
        i, j = patch_coords
        heatmap = viz[i, j]  # [h, w, 3]

        if mosaic_enabled:
            heatmap = build_mosaic_heatmap(heatmap, patch_coords, patch_size)

        heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_uint8)

        target_size = None
        if output_scale is not None:
            scale = float(output_scale)
            if scale > 0:
                target_size = (
                    max(1, int(round(heatmap_img.width * scale))),
                    max(1, int(round(heatmap_img.height * scale))),
                )
        elif resize_to_original:
            target_size = orig_size

        if target_size is not None and target_size != heatmap_img.size:
            heatmap_img = heatmap_img.resize(target_size, Image.BILINEAR)

        heatmap_img.save(output_path)
        print(f"已生成热力图: {output_path} (patch 坐标: {patch_coords})")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
