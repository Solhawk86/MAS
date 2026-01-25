#!/usr/bin/env python3
"""循环运行 train_m2f.py，不改动配置，仅重复训练."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = CURRENT_DIR / "config_m2f_segmentation.yaml"
TRAIN_SCRIPT = CURRENT_DIR / "train_m2f.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="循环调用 train_m2f.py，多次训练。")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CFG),
        help="YAML 配置路径（默认 MAS/config_m2f_segmentation.yaml）。",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="训练重复次数。",
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


def main() -> None:
    args = parse_args()
    base_cfg_path = Path(args.config).expanduser().resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {base_cfg_path}")
    if args.num_runs < 1:
        raise ValueError("--num-runs 需要为正整数。")

    print(f"[multi-train] 基础配置: {base_cfg_path}")

    for idx in range(1, args.num_runs + 1):
        cmd = _build_command(args, base_cfg_path)
        print(f"\n[multi-train] Run {idx}/{args.num_runs}")
        print(f"[multi-train] 命令: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
