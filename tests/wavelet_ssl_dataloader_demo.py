import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torchvision.transforms import v2
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data import build_ssl_dataloader
from data.augmentation import BatchWaveletSwapper
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _load_yaml_cfg(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_dir(cfg: dict, override: str | None) -> str:
    if override:
        return str(Path(override).expanduser().resolve())
    input_cfg = cfg.get("input_images", {})
    paths = input_cfg.get("paths")
    if not paths:
        raise ValueError("配置文件中缺少 input_images.paths，请通过 --data-dir 指定。")
    if isinstance(paths, (list, tuple)):
        base_dir = paths[0]
    else:
        base_dir = paths
    return str(Path(base_dir).expanduser().resolve())


def build_wavelet_dataloader(
    cfg: dict,
    *,
    batch_size: int,
    num_workers: int,
    data_dir: str,
) -> torch.utils.data.DataLoader:
    ssl_cfg = cfg.get("ssl", {})
    patch_size = cfg.get("student", {}).get("patch_size", 16)
    dataloader = build_ssl_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        ssl_cfg=ssl_cfg,
        patch_size=patch_size,
        dtype=torch.float32,
        seed=None,
        distributed=False,
    )
    return dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Wavelet-augmented SSL dataloader smoke test.")
    parser.add_argument("--config", type=str, default="config_eval.yaml", help="配置文件路径")
    parser.add_argument("--data-dir", type=str, default= "/mnt/mydata/hong/RMAS/train/img",help="训练数据目录，默认读取配置 input_images.paths")
    parser.add_argument("--batch-size", type=int, default=4, help="测试 batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers 数量")
    parser.add_argument("--wavelet", type=str, default="haar", help="小波类型")
    parser.add_argument("--mode", type=str, default="reflect", help="pywt padding 模式")
    parser.add_argument("--levels", type=int, default=10, help="小波分解层数")
    parser.add_argument("--clamp-min", type=float, default=0.0, help="clamp 下界")
    parser.add_argument("--clamp-max", type=float, default=1.0, help="clamp 上界")
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="禁用 clamp，允许结果不受 [min, max] 限制",
    )
    parser.add_argument("--swap-ratio", type=float, default=0.5, help="参与交换的样本比例，取值 [0,1]")
    parser.add_argument("--swap-crop-index", type=int, default=0, help="参与交换的 global crop 索引（0 或 1）")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/mydata/hong/wavelet_batch_outputs",
        help="保存整批图像的目录",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml_cfg(cfg_path)
    data_dir = _resolve_data_dir(cfg, args.data_dir)

    clamp_tuple = None if args.no_clamp else (args.clamp_min, args.clamp_max)
    dataloader = build_wavelet_dataloader(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=data_dir,
    )

    dataset = dataloader.dataset
    base_transform = getattr(dataset, "transform", None)
    swapper = BatchWaveletSwapper(
        swap_ratio=args.swap_ratio,
        swap_crop_index=args.swap_crop_index,
        wavelet=args.wavelet,
        mode=args.mode,
        clamp=clamp_tuple,
        local_swap_count=4,
        levels=args.levels,
        base_transform=base_transform,
    )
    base_collate: Callable[[list], dict] = dataloader.collate_fn

    def collate_with_wavelet(samples_list):
        sample_paths: list[str] = []
        swapped_pairs: list[tuple[str, str]] = []
        per_sample_views: list[dict[str, Any]] = []
        for idx, (_, metadata) in enumerate(samples_list):
            if isinstance(metadata, dict):
                sample_paths.append(str(metadata.get("path", f"sample_{idx}")))
            else:
                sample_paths.append(f"sample_{idx}")
        if samples_list:
            first_sample = samples_list[0][0]
            num_global = len(first_sample.get("global_crops", []))
            num_local = len(first_sample.get("local_crops", []))
        else:
            num_global = 0
            num_local = 0
        swapper(samples_list)
        for i, (sample, metadata) in enumerate(samples_list):
            per_sample_views.append(
                {
                    "path": sample_paths[i],
                    "global": [crop.detach().clone() for crop in sample.get("global_crops", [])],
                    "local": [crop.detach().clone() for crop in sample.get("local_crops", [])],
                    "teacher": [crop.detach().clone() for crop in sample.get("global_crops_teacher", [])],
                }
            )
        for idx_a, idx_b in getattr(swapper, "last_swapped_pairs", []):
            name_a = sample_paths[idx_a] if idx_a < len(sample_paths) else f"sample_{idx_a}"
            name_b = sample_paths[idx_b] if idx_b < len(sample_paths) else f"sample_{idx_b}"
            swapped_pairs.append((name_a, name_b))
        batch_dict = base_collate(samples_list)
        batch_dict["sample_paths"] = sample_paths
        batch_dict["num_global_views"] = num_global
        batch_dict["num_local_views"] = num_local
        batch_dict["wavelet_swapped_pairs"] = swapped_pairs
        batch_dict["per_sample_views"] = per_sample_views
        return batch_dict

    dataloader.collate_fn = collate_with_wavelet

    batch = next(iter(dataloader))
    print("Batch keys:", list(batch.keys()))
    print("collated_global_crops:", tuple(batch["collated_global_crops"].shape))
    print("collated_local_crops:", tuple(batch["collated_local_crops"].shape))
    print("collated_masks:", tuple(batch["collated_masks"].shape))
    if "collated_gram_teacher_crops" in batch:
        print("collated_gram_teacher_crops:", tuple(batch["collated_gram_teacher_crops"].shape))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = batch.pop("sample_paths", [])
    num_global = batch.pop("num_global_views", 0)
    num_local = batch.pop("num_local_views", 0)
    swapped_pairs = batch.get("wavelet_swapped_pairs", [])
    per_sample_views = batch.pop("per_sample_views", [])
    global_names = _build_view_filenames(sample_paths, "global", num_global)
    local_names = _build_view_filenames(sample_paths, "local", num_local)
    _save_batch_images(batch["collated_global_crops"], output_dir / "global_crops", global_names)
    _save_batch_images(batch["collated_local_crops"], output_dir / "local_crops", local_names)
    _save_original_full_images(sample_paths, output_dir / "original_images")
    _save_per_sample_views(per_sample_views, output_dir / "per_sample_crops")
    _save_swap_pairs(swapped_pairs, output_dir / "swap_pairs.txt")
    print(f"已将完整 batch 的 global/local crops 及原始图像保存到 {output_dir}")


def _save_batch_images(tensor: torch.Tensor, out_dir: Path, names: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = v2.ToPILImage()
    imgs = _unnormalize_for_visualization(tensor.detach().cpu())
    for idx, img in enumerate(imgs):
        filename = names[idx] if names and idx < len(names) else f"{idx:04d}.png"
        pil_img = to_pil(img.clamp(0.0, 1.0))
        pil_img.save(out_dir / filename)


def _unnormalize_for_visualization(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expect tensor shape [N,C,H,W], got {tuple(tensor.shape)}")
    device = tensor.device
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)
    return tensor * std + mean


def _build_view_filenames(sample_paths: list[str], prefix: str, num_views: int) -> list[str]:
    names: list[str] = []
    if num_views <= 0 or not sample_paths:
        return names
    stems = [_safe_stem(path) for path in sample_paths]
    for view_idx in range(num_views):
        for stem in stems:
            names.append(f"{stem}_{prefix}{view_idx}.png")
    return names


def _safe_stem(path_str: str | None) -> str:
    if not path_str:
        return "unknown"
    try:
        return Path(path_str).stem
    except OSError:
        return Path("unknown").stem


def _save_original_full_images(sample_paths: list[str], out_dir: Path) -> None:
    if not sample_paths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, path_str in enumerate(sample_paths):
        if not path_str or path_str == "None":
            continue
        try:
            file_path = Path(path_str)
            if not file_path.exists():
                continue
            with Image.open(path_str) as img:
                stem = file_path.stem or f"sample_{idx}"
                suffix = file_path.suffix or ".png"
                dest = out_dir / f"{stem}_original{suffix}"
                img.convert("RGB").save(dest)
        except Exception as exc:
            print(f"[WARN] 无法保存原图 {path_str}: {exc}")


def _save_swap_pairs(pairs: list[tuple[str, str]], file_path: Path) -> None:
    if not pairs:
        return
    lines = [f"{Path(a).stem} <-> {Path(b).stem}\n" for a, b in pairs]
    file_path.write_text("".join(lines), encoding="utf-8")


def _save_per_sample_views(per_sample_views: list[dict[str, Any]], out_dir: Path) -> None:
    if not per_sample_views:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, view in enumerate(per_sample_views):
        stem = _safe_stem(view.get("path"))
        sample_dir = out_dir / f"{idx:02d}_{stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_view_group(view.get("global"), sample_dir / "student_global")
        _save_view_group(view.get("local"), sample_dir / "student_local")
        _save_view_group(view.get("teacher"), sample_dir / "teacher_global")


def _save_view_group(tensors: list[torch.Tensor] | None, out_dir: Path) -> None:
    if not tensors:
        return
    stacked = torch.stack([t.detach().cpu() for t in tensors])
    _save_batch_images(stacked, out_dir)


if __name__ == "__main__":
    main()
