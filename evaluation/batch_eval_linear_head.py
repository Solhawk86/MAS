#!/usr/bin/env python3
# 批量评估微调权重
"""Batch evaluation utility for MAS segmentation eval."""

import argparse
import copy
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_eval.yaml"
DEFAULT_EVAL = SCRIPT_DIR / "evaluation/eval_linear_head.py"
WEIGHT_PATTERN = re.compile(r"student_backbone_step_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 MAS 的 eval_linear_head.py 评估一组权重"
    )
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="包含 student_backbone_step_*.pth 的目录",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="基础配置文件 (默认: %(default)s)",
    )
    parser.add_argument(
        "--seg-config",
        default=None,
        help="可选的 segmentation eval 配置文件，传递给 eval.py",
    )
    parser.add_argument(
        "--eval-script",
        default=str(DEFAULT_EVAL),
        help="eval.py 路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="运行 eval.py 的命令，可包含额外参数（例如 'torchrun --nproc_per_node=4'）",
    )
    return parser.parse_args()


def load_base_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置 {cfg_path} 不是一个有效的 YAML 字典")
    return cfg


def collect_weight_files(weights_dir: Path) -> List[Path]:
    matches: List[Tuple[int, Path]] = []
    for candidate in weights_dir.glob("student_backbone_step_*.pth"):
        match = WEIGHT_PATTERN.match(candidate.name)
        if not match:
            continue
        matches.append((int(match.group(1)), candidate.resolve()))
    matches.sort(key=lambda item: item[0])
    return [path for _, path in matches]


def resolve_feature_cache_dir(cfg: dict, eval_workdir: Path) -> Path:
    seg_section = cfg.get("segmentation_eval", {}) or {}
    cache_dir = seg_section.get("feature_cache_dir", "./feature_cache")
    cache_path = Path(str(cache_dir)).expanduser()
    if not cache_path.is_absolute():
        cache_path = (eval_workdir / cache_path).resolve()
    return cache_path


def clear_feature_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def write_temp_config(base_cfg: dict, weight_path: Path) -> Path:
    cfg_copy = copy.deepcopy(base_cfg)
    cfg_copy["finetuned_weights"] = str(weight_path)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_eval.yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(cfg_copy, tmp, allow_unicode=True, sort_keys=False)
    finally:
        tmp.close()
    return Path(tmp.name)


def _parse_python_bin(python_bin: str) -> List[str]:
    parts = shlex.split(python_bin)
    if not parts:
        raise ValueError("--python-bin 不能为空")
    return parts


def run_eval_command(
    python_bin: str,
    eval_script: Path,
    config_path: Path,
    seg_config: Optional[Path],
) -> None:
    cmd: List[str] = _parse_python_bin(python_bin)
    cmd.extend([str(eval_script), "--config", str(config_path)])
    if seg_config is not None:
        cmd.extend(["--seg-config", str(seg_config)])
    subprocess.run(cmd, check=True, cwd=str(eval_script.parent))


def validate_paths(
    eval_script: Path,
    weights_dir: Path,
    config_path: Path,
    seg_config: Optional[str],
) -> Optional[Path]:
    if not eval_script.is_file():
        raise FileNotFoundError(f"找不到 eval.py: {eval_script}")
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    if not weights_dir.is_dir():
        raise NotADirectoryError(f"权重目录不存在: {weights_dir}")
    if seg_config is None:
        return None
    seg_cfg_path = Path(seg_config).expanduser().resolve()
    if not seg_cfg_path.is_file():
        raise FileNotFoundError(f"找不到 seg-config 文件: {seg_cfg_path}")
    return seg_cfg_path


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    seg_config_path = validate_paths(
        eval_script, weights_dir, config_path, args.seg_config
    )

    base_cfg = load_base_config(config_path)
    cache_dir = resolve_feature_cache_dir(base_cfg, eval_script.parent)

    weight_files = collect_weight_files(weights_dir)
    if not weight_files:
        raise FileNotFoundError(
            f"在 {weights_dir} 下找不到 student_backbone_step_*.pth 文件"
        )

    for weight_path in weight_files:
        print(f"\n=== Evaluating {weight_path.name} ===", flush=True)
        clear_feature_cache(cache_dir)
        temp_config = write_temp_config(base_cfg, weight_path)
        try:
            run_eval_command(args.python_bin, eval_script, temp_config, seg_config_path)
        finally:
            temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
