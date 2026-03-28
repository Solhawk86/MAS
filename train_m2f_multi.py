#!/usr/bin/env python3
"""循环运行 train_m2f.py，可按多个数据集配置生成临时配置并依次训练/评估。"""
from __future__ import annotations

import argparse
import copy
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CFG = CURRENT_DIR / "config_m2f_segmentation.yaml"
DEFAULT_MULTI_CFG = CURRENT_DIR / "config_m2f_segmentation_multi.yaml"
TRAIN_SCRIPT = CURRENT_DIR / "train_m2f.py"
DEFAULT_TEMP_DIR = CURRENT_DIR / ".multi_configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="循环调用 train_m2f.py，可按多数据集配置批量训练/评估。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_MULTI_CFG),
        help=(
            "YAML 配置路径。若包含 datasets 字段，则按多数据集配置运行；"
            "否则按单配置重复训练。"
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="训练重复次数（多数据集模式下为整轮重复次数）。",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=None,
        help="可选训练命令前缀，例如 'torchrun --nproc_per_node=4'。",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="透传给 train_m2f.py 的 --resume 参数。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不真正执行。",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=str(DEFAULT_TEMP_DIR),
        help="多数据集模式下生成临时配置文件的目录。",
    )
    return parser.parse_args()


def _build_command(args: argparse.Namespace, cfg_path: Path) -> list[str]:
    if args.runner:
        cmd = shlex.split(args.runner)
    else:
        cmd = [sys.executable]
    cmd.append(str(TRAIN_SCRIPT))
    cmd.extend(["--config", str(cfg_path)])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    return cmd


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置内容必须是字典结构: {path}")
    return data


def _is_multi_config(cfg: dict) -> bool:
    return "datasets" in cfg


def _sanitize_name(name: str) -> str:
    if not name:
        return "dataset"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "dataset"


def _resolve_base_config(multi_cfg: dict, multi_cfg_path: Path) -> dict:
    base_spec = multi_cfg.get("base_config")
    if base_spec is None:
        base_path = DEFAULT_BASE_CFG
        return _load_yaml(base_path)
    if isinstance(base_spec, dict):
        return copy.deepcopy(base_spec)
    if not isinstance(base_spec, str):
        raise ValueError("base_config 需为路径字符串或字典。")
    base_path = Path(base_spec).expanduser()
    if not base_path.is_absolute():
        base_path = multi_cfg_path.parent / base_path
    if not base_path.exists():
        raise FileNotFoundError(f"找不到 base_config: {base_path}")
    return _load_yaml(base_path)


def _apply_dataset(base_cfg: dict, dataset_cfg: dict) -> dict:
    merged = copy.deepcopy(base_cfg)
    if "dataset" not in merged or not isinstance(merged["dataset"], dict):
        merged["dataset"] = {}
    for key in ("train_images", "train_masks", "val_images", "val_masks"):
        if key not in dataset_cfg:
            raise ValueError(f"dataset 配置缺少必需字段: {key}")
        merged["dataset"][key] = dataset_cfg[key]
    if "evaluation" not in merged or not isinstance(merged["evaluation"], dict):
        merged["evaluation"] = {}
    merged["evaluation"]["gt_dir"] = dataset_cfg.get("gt_dir", dataset_cfg["val_masks"])
    return merged


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs 需要为正整数。")

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    if _is_multi_config(cfg):
        datasets = cfg.get("datasets") or []
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("多数据集配置缺少 datasets 列表。")
        base_cfg = _resolve_base_config(cfg, cfg_path)
        temp_dir = Path(args.temp_dir).expanduser()
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[multi-train] 多数据集配置: {cfg_path}")
        print(f"[multi-train] base_config: {cfg.get('base_config', DEFAULT_BASE_CFG)}")
        print(f"[multi-train] 生成临时配置目录: {temp_dir}")

        for run_idx in range(1, args.num_runs + 1):
            if args.num_runs > 1:
                print(f"\n[multi-train] Round {run_idx}/{args.num_runs}")
            for ds_idx, ds_cfg in enumerate(datasets, start=1):
                if not isinstance(ds_cfg, dict):
                    raise ValueError("datasets 列表元素必须是字典。")
                name = _sanitize_name(str(ds_cfg.get("name", f"dataset_{ds_idx}")))
                merged_cfg = _apply_dataset(base_cfg, ds_cfg)
                out_cfg_path = temp_dir / f"{ds_idx:02d}_{name}.yaml"
                with open(out_cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=False)

                cmd = _build_command(args, out_cfg_path)
                print(f"\n[multi-train] Dataset {ds_idx}/{len(datasets)}: {name}")
                print(f"[multi-train] 命令: {' '.join(cmd)}")
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
        return

    print(f"[multi-train] 单配置模式: {cfg_path}")
    for idx in range(1, args.num_runs + 1):
        cmd = _build_command(args, cfg_path)
        print(f"\n[multi-train] Run {idx}/{args.num_runs}")
        print(f"[multi-train] 命令: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
