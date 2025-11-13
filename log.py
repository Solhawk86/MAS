from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils import create_tensorboard_writer, tensorboard_log_scalars


@dataclass(frozen=True)
class RunWorkspace:
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    log_file: Path


class TrainingLogger:
    """集中管理训练运行的日志、TensorBoard 以及目录结构。"""

    def __init__(self, workspace: RunWorkspace, logger: logging.Logger | None, tb_writer: SummaryWriter | None) -> None:
        self.workspace = workspace
        self._logger = logger
        self._tb_writer = tb_writer

    @property
    def checkpoint_dir(self) -> Path:
        return self.workspace.checkpoints_dir

    def log_config(self, cfg: dict[str, Any]) -> None:
        """将完整配置写入日志文件，便于复现。"""
        if self._logger is None:
            return
        cfg_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        self._logger.info("=== 训练配置 ===\n%s", cfg_text)

    def log_scalars(self, step: int, *scalar_groups: dict[str, float]) -> None:
        if self._tb_writer is None:
            return
        for scalars in scalar_groups:
            if scalars:
                tensorboard_log_scalars(self._tb_writer, step, scalars)

    def log_progress(
        self,
        *,
        step: int,
        max_steps: int,
        loss_val: float,
        lr_val: float,
        teacher_temp: float,
        momentum: float,
        running_metrics: dict[str, float],
    ) -> None:
        metrics_pairs = sorted(running_metrics.items())
        metrics_str = " ".join(f"{name}:{value:.4f}" for name, value in metrics_pairs)
        message = (
            f"[Step {step}/{max_steps}] loss={loss_val:.4f} lr={lr_val:.6e} "
            f"temp={teacher_temp:.4f} mom={momentum:.5f} {metrics_str}"
        ).strip()
        self.log_message(message)

    def log_message(self, message: str) -> None:
        print(message)
        if self._logger is not None:
            self._logger.info(message)

    def log_debug(self, message: str) -> None:
        if self._logger is not None:
            self._logger.debug(message)

    def log_module_grad_status(
        self,
        module: nn.Module | None,
        *,
        label: str = "",
        only_trainable: bool = False,
    ) -> None:
        if self._logger is None or module is None:
            return
        try:
            params = module.named_parameters()
        except AttributeError:
            return
        prefix = f"{label}." if label else ""
        for name, param in params:
            if only_trainable and not param.requires_grad:
                continue
            self._logger.debug(f"grad={param.requires_grad} {prefix}{name}")

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        if self._logger is not None:
            for handler in list(self._logger.handlers):
                handler.close()
                self._logger.removeHandler(handler)
            self._logger = None


def init_training_logger(
    *,
    rank: int,
    distributed: bool,
    tensorboard_enabled: bool,
    tensorboard_subdir: str | None,
    checkpoint_subdir: str | None,
    weights_root: str | Path | None = None,
) -> TrainingLogger:
    workspace = _build_workspace(
        rank=rank,
        distributed=distributed,
        tensorboard_subdir=tensorboard_subdir,
        checkpoint_subdir=checkpoint_subdir,
        weights_root=weights_root,
    )
    logger = _configure_file_logger(workspace.log_file) if rank == 0 else None
    tb_writer = create_tensorboard_writer(workspace.tensorboard_dir, rank, tensorboard_enabled)
    return TrainingLogger(workspace, logger, tb_writer)


def _build_workspace(
    *,
    rank: int,
    distributed: bool,
    tensorboard_subdir: str | None,
    checkpoint_subdir: str | None,
    weights_root: str | Path | None,
) -> RunWorkspace:
    base_dir = Path(weights_root) if weights_root is not None else Path(__file__).resolve().parent / "weights"
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir_str = ""
    if rank == 0:
        run_dir = _next_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_str = str(run_dir)

    if _should_broadcast(distributed):
        buffer = [run_dir_str]
        dist.broadcast_object_list(buffer, src=0)
        run_dir_str = buffer[0]

    if not run_dir_str:
        raise RuntimeError("无法确定训练运行目录")

    run_dir = Path(run_dir_str)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = _subdir_path(run_dir, checkpoint_subdir, "checkpoints")
    tensorboard_dir = _subdir_path(run_dir, tensorboard_subdir, "tensorboard")
    log_file = run_dir / "train.log"

    return RunWorkspace(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tensorboard_dir=tensorboard_dir,
        log_file=log_file,
    )


def _next_run_dir(base_dir: Path) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    candidate = base_dir / date_str
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{date_str}_{suffix:02d}"
        suffix += 1
    return candidate


def _subdir_path(root: Path, requested: str | None, default_name: str) -> Path:
    name = (requested or default_name).strip()
    if not name:
        name = default_name
    sub_path = Path(name)
    if sub_path.is_absolute():
        sub_path = Path(sub_path.name)
    cleaned_parts = [part for part in sub_path.parts if part not in ("..", ".", "")]
    if not cleaned_parts:
        cleaned_parts = [default_name]
    relative_path = Path(*cleaned_parts)
    target = root / relative_path
    target.mkdir(parents=True, exist_ok=True)
    return target


def _configure_file_logger(log_file: Path) -> logging.Logger:
    logger_name = f"mas_train_{log_file.parent.name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _should_broadcast(distributed: bool) -> bool:
    return distributed and dist.is_available() and dist.is_initialized()


__all__ = [
    "RunWorkspace",
    "TrainingLogger",
    "init_training_logger",
]
