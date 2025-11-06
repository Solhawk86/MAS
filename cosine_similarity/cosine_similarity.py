#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DINOv3 模型对指定 patch 与其它 patch 的相似度进行可视化。
在服务器端运行，无需交互式界面，直接将热力图存成图片。
所有参数均从配置文件读取。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 强制使用非交互式后端
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import DualImageDINOv3, visualize_patch_similarity
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
    """
    根据配置生成热力图输出路径。
    """
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{image_path.stem}_similarity.png"
    return image_path.with_name(image_path.stem + "_similarity.png")


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

    model = DualImageDINOv3(
        cfg=cfg,
        device=device,
        backbone_out_layers=backbone_layers,
    )
    model.eval()

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

    for image_path in image_paths:
        image_path = image_path.expanduser()
        img_tensor, orig_size = load_image(image_path)
        output_path = ensure_output_path(image_path, output_dir)

        with torch.no_grad():
            viz, _ = visualize_patch_similarity(
                model=model,
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

        # 根据 patch 网格生成马赛克式热力图，并将选中 patch 着色为红色
        mosaic = np.repeat(np.repeat(np.clip(heatmap, 0.0, 1.0), patch_size, axis=0), patch_size, axis=1)
        red_color = np.array([1.0, 0.0, 0.0])
        y0 = i * patch_size
        x0 = j * patch_size
        mosaic[y0 : y0 + patch_size, x0 : x0 + patch_size] = red_color

        mosaic_uint8 = (np.clip(mosaic, 0.0, 1.0) * 255.0).astype(np.uint8)
        heatmap_img = Image.fromarray(mosaic_uint8)

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
