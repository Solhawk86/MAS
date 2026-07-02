#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache offline DINOv3 depth maps")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repo-dir", type=str, default="/data/hong/dinov3")
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default="/data/hong/dino_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--head-weights",
        type=str,
        default="/data/hong/dino_weights/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth",
    )
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _make_transform(size: int):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _scan_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _load_local_or_url_state_dict(original_loader):
    def _loader(url, *loader_args, **loader_kwargs):
        parsed = urlparse(str(url))
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
            print(f"[cache_dinov3_depth] loading local weights: {path}")
            return torch.load(path, map_location=loader_kwargs.get("map_location", "cpu"), weights_only=True)
        if parsed.scheme == "" and Path(str(url)).expanduser().is_file():
            path = Path(str(url)).expanduser()
            print(f"[cache_dinov3_depth] loading local weights: {path}")
            return torch.load(path, map_location=loader_kwargs.get("map_location", "cpu"), weights_only=True)
        return original_loader(url, *loader_args, **loader_kwargs)

    return _loader


def _keep_norm_layers_fp32(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.float()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = _dtype(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    # DINOv3's Depther constructor moves to CUDA when torch.cuda.is_available().
    # Build on CPU first so the 7B backbone never lands on a 24G card in fp32.
    original_cuda_available = torch.cuda.is_available
    original_state_dict_loader = torch.hub.load_state_dict_from_url
    torch.cuda.is_available = lambda: False
    torch.hub.load_state_dict_from_url = _load_local_or_url_state_dict(original_state_dict_loader)
    try:
        depther = torch.hub.load(
            args.repo_dir,
            "dinov3_vit7b16_dd",
            source="local",
            weights=args.head_weights,
            backbone_weights=args.backbone_weights,
            autocast_dtype=dtype,
        )
    finally:
        torch.hub.load_state_dict_from_url = original_state_dict_loader
        torch.cuda.is_available = original_cuda_available
    depther.eval()
    if device.type == "cuda" and dtype != torch.float32:
        depther.to(device=device, dtype=dtype)
        _keep_norm_layers_fp32(depther)
    else:
        depther.to(device=device)
    depther.autocast_ctx = partial(
        torch.autocast,
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type == "cuda" and dtype != torch.float32),
    )

    transform = _make_transform(args.img_size)
    images = _scan_images(image_dir)
    if args.limit > 0:
        images = images[: args.limit]
    print(
        f"[cache_dinov3_depth] images={len(images)} output={output_dir} device={device} dtype={dtype} img_size={args.img_size}"
    )

    with torch.inference_mode():
        for idx, path in enumerate(images, start=1):
            out_path = output_dir / f"{path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                continue
            image = Image.open(path).convert("RGB")
            orig_h, orig_w = image.height, image.width
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                depth = depther(tensor)
            depth = F.interpolate(
                depth.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).cpu()
            torch.save(
                {
                    "depth": depth,
                    "orig_size": (orig_h, orig_w),
                    "source_image": str(path),
                    "img_size": args.img_size,
                },
                out_path,
            )
            if idx % 25 == 0 or idx == len(images):
                print(f"[cache_dinov3_depth] {idx}/{len(images)} {out_path}")


if __name__ == "__main__":
    main()
