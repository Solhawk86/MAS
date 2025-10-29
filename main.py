from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

from data.data import ImagesOnlyDataset, build_ssl_dataloader
from loss import SelfSupervisedLoss
from model import DINOv3SelfSupervised
from cosine_scheduler import CosineScheduler

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
from utils import launch_distributed, load_config


def _resolve_base_dir(input_cfg: dict) -> str:
    """
    从配置中解析数据集路径
    """
    paths = input_cfg.get("paths", "") if input_cfg else ""
    if isinstance(paths, (list, tuple)):
        return paths[0] if paths else ""
    return paths


def _build_teacher_temp_schedule(cfg: dict, total_iters: int) -> np.ndarray:
    """
    生成 warmup 阶段线性增加的温度。
    剩余步骤保持 final 温度。
    确保数组长度等于 total_iters。
    返回 numpy 数组。
    """
    base = float(cfg.get("base", 0.04))
    final = float(cfg.get("final", base))
    warmup = int(cfg.get("warmup_iters", 0))
    warmup = max(0, min(warmup, total_iters))

    if total_iters <= 0:
        return np.array([], dtype=np.float32)

    if warmup > 0:
        # 线性warmup
        warmup_vals = np.linspace(base, final, warmup, endpoint=False, dtype=np.float32)
        remaining = total_iters - warmup
        if remaining > 0:
            rest = np.full((remaining,), final, dtype=np.float32)
            schedule = np.concatenate([warmup_vals, rest])
        else:
            schedule = warmup_vals
    else:
        schedule = np.full((total_iters,), final, dtype=np.float32)

    if schedule.size < total_iters:
        pad = np.full((total_iters - schedule.size,), final, dtype=np.float32)
        schedule = np.concatenate([schedule, pad])
    return schedule


def _ensure_dir(path: str | Path) -> None:
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def _broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for param in module.parameters():
        dist.broadcast(param.data, src)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src)


def _broadcast_loss_state(loss_module: SelfSupervisedLoss, src: int = 0) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    dist.broadcast(loss_module.dino_loss.center, src)
    dist.broadcast(loss_module.ibot_patch_loss.center, src)


def _all_reduce_tensor(tensor: torch.Tensor, divisor: int) -> torch.Tensor:
    if divisor <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= divisor
    return tensor


def train_worker(rank: int, world_size: int, cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    train_cfg = cfg.get("train", {})
    ssl_cfg = cfg.get("ssl", {})

    seed = int(train_cfg.get("seed", 0) or 0)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    global_batch = int(train_cfg.get("batchsize", 64))
    if world_size > 1 and global_batch % world_size != 0:
        raise ValueError(f"Global batch size {global_batch} 不能被进程数 {world_size} 整除")
    batch_size = global_batch // world_size if world_size > 1 else global_batch

    num_workers = int(train_cfg.get("num_workers", 4))
    max_steps = int(train_cfg.get("steps", train_cfg.get("max_steps", 50000)))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")

    # LoRA
    lora_layers = train_cfg.get("lora_layers", 4)
    lora_rank = train_cfg.get("lora_rank", 8)
    lora_alpha = float(train_cfg.get("lora_alpha", 16.0))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.0))

    # 调度器超参
    lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    warmup_start_lr = float(train_cfg.get("warmup_start_lr", lr * 0.1 if warmup_steps > 0 else lr))

    teacher_temp_schedule = _build_teacher_temp_schedule(ssl_cfg.get("dino", {}).get("teacher_temp", {}), max_steps)
    ema_cfg = ssl_cfg.get("ema", {})
    momentum_schedule = CosineScheduler(
        base_value=float(ema_cfg.get("base", 0.996)),
        final_value=float(ema_cfg.get("final", 1.0)),
        total_iters=max_steps,
    )
    lr_schedule = CosineScheduler(
        base_value=lr,
        final_value=min_lr,
        total_iters=max_steps,
        warmup_iters=warmup_steps,
        start_warmup_value=warmup_start_lr,
    )

    input_cfg = cfg.get("input_images", {})
    base_dir = _resolve_base_dir(input_cfg)
    dataset_name = input_cfg.get("dataset_name", "")
    image_subdir = input_cfg.get("image_subdir")
    if not base_dir or not dataset_name:
        raise ValueError("配置中的 input_images.paths 或 dataset_name 未设置有效路径。")

    patch_size = cfg.get("student", {}).get("patch_size", 16)

    dataset = ImagesOnlyDataset(
        base_dir=base_dir,
        dataset_name=dataset_name,
        image_subdir=image_subdir,
        transform=None,
    )

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    dataloader = build_ssl_dataloader(
        base_dir=base_dir,
        dataset_name=dataset_name,
        image_subdir=image_subdir,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        ssl_cfg=ssl_cfg,
        patch_size=patch_size,
        dtype=torch.float32,
        seed=seed + rank,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    exhaustive_nan = int(train_cfg.get("max_nan_steps", 3))

    if sampler is not None:
        sampler.set_epoch(0)

    model = DINOv3SelfSupervised(
        cfg=cfg,
        device=str(device),
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        ssl_cfg=ssl_cfg,
    ).to(device)
    # loss = DINO + iBOT + KOLEO 
    # 暂未加入Gram
    criterion = SelfSupervisedLoss(
        dino_cfg=ssl_cfg.get("dino", {}),
        ibot_cfg=ssl_cfg.get("ibot", {}),
        koleo_cfg=ssl_cfg.get("koleo", {}),
    ).to(device)

    # DINOv3 的 DINOLoss / iBOTPatchLoss 会把 center 缓冲区初始化成 NaN，
    # 官方训练在 SSLMetaArch.init_weights() 里专门调用 init_weights() 把它们清零。
    # 如果不做这一步，第一次 softmax 时就会因为减去 NaN 中心导致整个 loss 变成 NaN。
    # 调用 init_weights() 保证中心正确初始化，后续 update_center 才可以按动量更新，从而避免 Nan 
    if rank == 0:
        criterion.dino_loss.init_weights()
        criterion.ibot_patch_loss.init_weights()

    trainable_params = model.student_trainable_parameters()
    if not trainable_params:
        raise RuntimeError("未找到可训练的学生参数 (LoRA + Head)")

    if world_size > 1:
        _broadcast_module_state(model, src=0)
        _broadcast_loss_state(criterion, src=0)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    data_iter = iter(dataloader)
    running_metrics: dict[str, float] = {}
    start_step = 1
    epoch = 0
    is_rank_zero = rank == 0

    if is_rank_zero:
        _ensure_dir(checkpoint_dir)

    consecutive_nan_count = 0

    for step in range(start_step, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        lr_this_step = float(lr_schedule[step - 1])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        teacher_temp = float(teacher_temp_schedule[min(step - 1, len(teacher_temp_schedule) - 1)])
        momentum = float(momentum_schedule[step - 1])

        global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
        masks = batch["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = batch["mask_indices_list"].to(device)
        masks_weight = batch["masks_weight"].to(device)
        n_masked_patches = torch.clamp(batch["n_masked_patches"].to(device), min=1)

        n_global_crops = 2
        B = global_crops.shape[0] // n_global_crops
        n_local_crops = local_crops.shape[0] // B
        
        # 按 (crop, batch, C, H, W) 重新组织张量，便于喂入 DINO 主干
        global_crops = global_crops.view(n_global_crops, B, *global_crops.shape[1:])
        local_crops = local_crops.view(n_local_crops, B, *local_crops.shape[1:])
        masks = masks.view(n_global_crops * B, -1)
        mask_indices_list = mask_indices_list.long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            teacher_global = model.get_teacher_output(
                global_crops,
                mask_indices_list=mask_indices_list,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches,
                dino_loss=criterion.dino_loss,
                ibot_loss=criterion.ibot_patch_loss,
            )
            # if torch.isnan(teacher_global["cls_centered"]).any():
            #     raise RuntimeError(f"NaN detected in teacher cls_centered on rank {rank} at step {step}")
            student_global, student_local = model.get_student_output(
                global_crops=global_crops,
                local_crops=local_crops,
                masks=masks,
                mask_indices_list=mask_indices_list,
            )

            loss, loss_terms = criterion(
                teacher_global=teacher_global,
                student_global=student_global,
                student_local=student_local,
                masks=masks,
                mask_indices_list=mask_indices_list,
                masks_weight=masks_weight,
            )
            loss_finite = torch.isfinite(loss)
            if world_size > 1:
                finite_tensor = torch.tensor(
                    1 if loss_finite else 0, device=device, dtype=torch.int32
                )
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
                loss_finite = bool(finite_tensor.item())

            if not loss_finite:
                consecutive_nan_count += 1
                if is_rank_zero:
                    print(
                        f"[warn] step {step} encountered non-finite loss (count {consecutive_nan_count}); skipping batch"
                    )
                if consecutive_nan_count > exhaustive_nan:
                    raise RuntimeError("Too many consecutive NaN/Inf losses, aborting training")
                optimizer.zero_grad(set_to_none=True)
                continue

        scaler.scale(loss).backward()
        consecutive_nan_count = 0

        need_unscale = grad_clip > 0 or world_size > 1
        if need_unscale:
            scaler.unscale_(optimizer)

        if world_size > 1:
            for param in trainable_params:
                if param.grad is not None:
                    _all_reduce_tensor(param.grad, world_size)

        if grad_clip > 0:
            clip_grad_norm_(trainable_params, grad_clip)

        scaler.step(optimizer)
        scaler.update()

        model.update_teacher(momentum)
        criterion.update_centers(
            teacher_cls_after_head=teacher_global["cls_after_head"].flatten(0, 1),
            teacher_masked_patch_after_head=teacher_global["masked_patch_after_head"],
        )

        loss_detached = loss.detach()
        loss_detached = _all_reduce_tensor(loss_detached, world_size)
        loss_val = float(loss_detached.cpu())

        loss_terms_detached: dict[str, torch.Tensor] = {}
        for key, value in loss_terms.items():
            tensor_detached = value.detach()
            tensor_detached = _all_reduce_tensor(tensor_detached, world_size)
            loss_terms_detached[key] = tensor_detached

        if is_rank_zero:
            running_metrics.setdefault("loss", loss_val)
            running_metrics["loss"] = running_metrics["loss"] * 0.9 + loss_val * 0.1
            for key, value in loss_terms_detached.items():
                val = float(value.cpu())
                running_metrics[key] = running_metrics.get(key, val) * 0.9 + val * 0.1

            if step % log_interval == 0 or step == 1:
                metrics_str = " ".join(f"{k}:{v:.4f}" for k, v in sorted(running_metrics.items()))
                print(
                    f"[Rank {rank}] [Step {step}/{max_steps}] loss={loss_val:.4f} lr={lr_this_step:.6e} "
                    f"temp={teacher_temp:.4f} mom={momentum:.5f} {metrics_str}"
                )

            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                ckpt_path = Path(checkpoint_dir) / f"step_{step:06d}.pth"
                torch.save(
                    {
                        "step": step,
                        "teacher": model.teacher_state_dict(),
                        "student": model.student_state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    ckpt_path,
                )

    if is_rank_zero:
        final_ckpt = Path(checkpoint_dir) / "final_teacher.pth"
        torch.save(
            {
                "teacher": model.teacher_state_dict(),
                "student": model.student_state_dict(),
            },
            final_ckpt,
        )
        print(f"训练结束，教师权重已保存至 {final_ckpt}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "config_eval.yaml"
    cfg = load_config(str(cfg_path))
    devices = cfg.get("train", {}).get("devices", "auto")

    launch_distributed(
        train_worker,
        devices=devices,
        args=(str(cfg_path),),
    )


if __name__ == "__main__":
    main()
