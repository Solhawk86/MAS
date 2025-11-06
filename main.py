from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP


from data.data import build_ssl_dataloader
from loss import SelfSupervisedLoss
from model import DINOv3SelfSupervised
from utils import (
    _build_teacher_temp_schedule,
    _distributed_mean,
    _ensure_dir,
    _ensure_torch_fsdp_compat,
    _setup_device_and_distributed,
    load_config,
)

# NOTE: 训练脚本遵循 DINOv3 自监督范式，仅更新 LoRA + 头部参数。四卡 3090 建议调整 batchsize.

_ensure_torch_fsdp_compat()

from dinov3.train.cosine_lr_scheduler import CosineScheduler  # noqa: E402


def train() -> None:
    cfg_path = Path(__file__).resolve().parent / "config_eval.yaml"
    cfg = load_config(cfg_path)
    device, rank, world_size, local_rank, distributed = _setup_device_and_distributed()
    is_main_process = rank == 0

    train_cfg = cfg.get("train", {})
    ssl_cfg = cfg.get("ssl", {})
    #1、训练参数
    batch_size = int(train_cfg.get("batchsize", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    max_steps = int(train_cfg.get("steps", train_cfg.get("max_steps", 50000)))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))

    #2、配置 LoRA 微调参数，仅训练 LoRA 层和head。
    lora_layers = train_cfg.get("lora_layers", 4)
    lora_rank = train_cfg.get("lora_rank", 8)
    lora_alpha = float(train_cfg.get("lora_alpha", 16.0))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.0))
    #3、调度器
    lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    warmup_start_lr = float(train_cfg.get("warmup_start_lr", lr * 0.1 if warmup_steps > 0 else lr))

    teacher_temp_schedule = _build_teacher_temp_schedule(ssl_cfg.get("dino", {}).get("teacher_temp", {}), max_steps)
    if teacher_temp_schedule.size > 0:
        teacher_temp_schedule = np.maximum(teacher_temp_schedule, 1e-4)
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
    #4、加载图像
    input_cfg = cfg.get("input_images", {})
    paths = input_cfg.get("paths")
    if not paths:
        raise ValueError("配置中的 input_images.paths 未设置有效路径。")
    if isinstance(paths, (list, tuple)):
        base_dir = paths[0]
    else:
        base_dir = paths
    base_dir = str(Path(base_dir).expanduser().resolve())

    base_seed = train_cfg.get("seed")
    seed_value = int(base_seed) + rank if base_seed is not None else None

    patch_size = cfg.get("student", {}).get("patch_size", 16)
    dataloader = build_ssl_dataloader(
        data_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        ssl_cfg=ssl_cfg,
        patch_size=patch_size,
        dtype=torch.float32,
        seed=seed_value,
        distributed=distributed,
        num_replicas=world_size,
        rank=rank,
    )

    model = DINOv3SelfSupervised(
        cfg=cfg,
        device=str(device),
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        ssl_cfg=ssl_cfg,
    ).to(device)
    if distributed:
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        model.student["backbone"] = DDP(model.student["backbone"], **ddp_kwargs)
        model.student["dino_head"] = DDP(model.student["dino_head"], **ddp_kwargs)
        model.student["ibot_head"] = DDP(model.student["ibot_head"], **ddp_kwargs)
    # loss = DINO + iBOT + KOLEO 
    # 暂未加入Gram
    criterion = SelfSupervisedLoss(
        dino_cfg=ssl_cfg.get("dino", {}),
        ibot_cfg=ssl_cfg.get("ibot", {}),
        koleo_cfg=ssl_cfg.get("koleo", {}),
    ).to(device)

    # 仅收集 LoRA 与学生头部参数，保持主干冻结
    trainable_params = model.student_trainable_parameters()
    if not trainable_params:
        raise RuntimeError("未找到可训练的学生参数 (LoRA + Head)")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    data_iter = iter(dataloader)
    current_epoch = 0
    running_metrics: dict[str, float] = {}
    update_step = 0

    _ensure_dir(checkpoint_dir)
    
    while update_step < max_steps:
        optimizer.zero_grad(set_to_none=True)

        lr_this_step = float(lr_schedule[update_step])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        if len(teacher_temp_schedule) > 0:
            teacher_temp = float(teacher_temp_schedule[min(update_step, len(teacher_temp_schedule) - 1)])
        else:
            teacher_temp = 0.04
        teacher_temp = max(teacher_temp, 1e-4)
        momentum = float(momentum_schedule[update_step])

        micro_losses: list[torch.Tensor] = []
        micro_terms: list[dict[str, torch.Tensor]] = []
        cls_center_buffer: list[torch.Tensor] = []
        mask_center_buffer: list[torch.Tensor] = []

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                current_epoch += 1
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(current_epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
            local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
            masks = batch["collated_masks"].to(device, non_blocking=True)
            mask_indices_list = batch["mask_indices_list"].to(device)
            masks_weight = batch["masks_weight"].to(device)
            n_masked_patches = batch["n_masked_patches"].to(device)

            n_global_crops = 2
            B = global_crops.shape[0] // n_global_crops
            n_local_crops = local_crops.shape[0] // B

            global_crops = global_crops.view(n_global_crops, B, *global_crops.shape[1:])
            local_crops = local_crops.view(n_local_crops, B, *local_crops.shape[1:])
            masks = masks.view(n_global_crops * B, -1)
            mask_indices_list = mask_indices_list.long()

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                teacher_global = model.get_teacher_output(
                    global_crops,
                    mask_indices_list=mask_indices_list,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches,
                    dino_loss=criterion.dino_loss,
                    ibot_loss=criterion.ibot_patch_loss,
                )
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

            (loss / grad_accum_steps).backward()
            micro_losses.append(loss.detach())
            micro_terms.append({k: v.detach() for k, v in loss_terms.items()})
            cls_center_buffer.append(teacher_global["cls_after_head"].flatten(0, 1).detach())
            mask_center = teacher_global["masked_patch_after_head"]
            if mask_center.numel() > 0:
                mask_center_buffer.append(mask_center.detach())

        if grad_clip > 0:
            clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()

        update_step += 1

        model.update_teacher(momentum)
        teacher_cls_for_center = torch.cat(cls_center_buffer, dim=0)
        if mask_center_buffer:
            teacher_mask_for_center = torch.cat(mask_center_buffer, dim=0)
        else:
            teacher_mask_for_center = torch.empty_like(cls_center_buffer[:0])
        criterion.update_centers(
            teacher_cls_after_head=teacher_cls_for_center,
            teacher_masked_patch_after_head=teacher_mask_for_center,
        )

        loss_mean = _distributed_mean(torch.stack(micro_losses).mean(), world_size)
        reduced_terms: dict[str, torch.Tensor] = {}
        if micro_terms:
            term_keys = micro_terms[0].keys()
            for key in term_keys:
                stacked = torch.stack([term[key] for term in micro_terms])
                reduced_terms[key] = _distributed_mean(stacked.mean(), world_size)
        loss_val = float(loss_mean.cpu())

        if is_main_process:
            running_metrics.setdefault("loss", loss_val)
            running_metrics["loss"] = running_metrics["loss"] * 0.9 + loss_val * 0.1
            for key, tensor_val in reduced_terms.items():
                val = float(tensor_val.cpu())
                running_metrics[key] = running_metrics.get(key, val) * 0.9 + val * 0.1

            if update_step % log_interval == 0 or update_step == 1:
                metrics_str = " ".join(f"{k}:{v:.4f}" for k, v in sorted(running_metrics.items()))
                print(
                    f"[Step {update_step}/{max_steps}] loss={loss_val:.4f} lr={lr_this_step:.6e} "
                    f"temp={teacher_temp:.4f} mom={momentum:.5f} {metrics_str}"
                )

            if checkpoint_interval > 0 and update_step % checkpoint_interval == 0:
                ckpt_path = Path(checkpoint_dir) / f"step_{update_step:06d}.pth"
                torch.save(
                    {
                        "step": update_step,
                        "teacher": model.teacher_state_dict(),
                        "student": model.student_state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )

    if distributed:
        dist.barrier()

    if is_main_process:
        final_ckpt = Path(checkpoint_dir) / "final_teacher.pth"
        torch.save(
            {
                "teacher": model.teacher_state_dict(),
                "student": model.student_state_dict(),
            },
            final_ckpt,
        )
        print(f"训练结束，教师权重已保存至 {final_ckpt}")


if __name__ == "__main__":
    train()
