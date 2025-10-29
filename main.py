import os
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


def _ensure_torch_fsdp_compat() -> None:
    try:
        import torch.distributed.fsdp as torch_fsdp  # type: ignore
    except (ImportError, ModuleNotFoundError, AttributeError):
        return
    if not hasattr(torch_fsdp, "register_fsdp_forward_method"):
        def _register_fsdp_forward_method(*args, **kwargs):
            return None

        torch_fsdp.register_fsdp_forward_method = _register_fsdp_forward_method  # type: ignore[attr-defined]

    module_name = "torch.distributed.fsdp._fully_shard"
    submodule_name = f"{module_name}._fsdp_state"
    if submodule_name not in sys.modules:
        try:
            from torch.distributed._composable.fsdp import _fsdp_state as composable_fsdp_state  # type: ignore[attr-defined]
        except (ImportError, ModuleNotFoundError, AttributeError):
            return

        fully_shard_pkg = sys.modules.get(module_name)
        if fully_shard_pkg is None:
            fully_shard_pkg = types.ModuleType(module_name)
            fully_shard_pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules[module_name] = fully_shard_pkg
            setattr(torch_fsdp, "_fully_shard", fully_shard_pkg)

        fsdp_state_module = types.ModuleType(submodule_name)
        for attr in ("FSDPState", "FSDPStateContext"):
            if hasattr(composable_fsdp_state, attr):
                setattr(fsdp_state_module, attr, getattr(composable_fsdp_state, attr))

        sys.modules[submodule_name] = fsdp_state_module
        setattr(fully_shard_pkg, "_fsdp_state", fsdp_state_module)


_ensure_torch_fsdp_compat()


def _ensure_torch_distributed_initialized(device: torch.device) -> None:
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" and torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)


from data.data import build_ssl_dataloader
from loss import SelfSupervisedLoss
from model import DINOv3SelfSupervised
from utils import load_config
from dinov3.train.cosine_lr_scheduler import CosineScheduler

# NOTE: 训练脚本遵循 DINOv3 自监督范式，仅更新 LoRA + 头部参数。四卡 3090 建议调整 batchsize.


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
            # 后续步骤采用final作为恒定温度
            rest = np.full((remaining,), final, dtype=np.float32)
            schedule = np.concatenate([warmup_vals, rest])
        else:
            schedule = warmup_vals
    else:
        schedule = np.full((total_iters,), final, dtype=np.float32)

    if schedule.size < total_iters:
        # 补齐长度，防止前面拼接导致长度不足 total_iters
        pad = np.full((total_iters - schedule.size,), final, dtype=np.float32)
        schedule = np.concatenate([schedule, pad])
    return schedule


def _ensure_dir(path: str | Path) -> None:
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def train() -> None:
    cfg_path = Path(__file__).resolve().parent / "config_eval.yaml"
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_torch_distributed_initialized(device)

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
    base_dir = _resolve_base_dir(input_cfg)
    dataset_name = input_cfg.get("dataset_name", "")
    image_subdir = input_cfg.get("image_subdir")
    if not base_dir or not dataset_name:
        raise ValueError("配置中的 input_images.paths 或 dataset_name 未设置有效路径。")

    patch_size = cfg.get("student", {}).get("patch_size", 16)
    dataloader = build_ssl_dataloader(
        base_dir=base_dir,
        dataset_name=dataset_name,
        image_subdir=image_subdir,
        batch_size=batch_size,
        num_workers=num_workers,
        ssl_cfg=ssl_cfg,
        patch_size=patch_size,
        dtype=torch.float32,
        seed=train_cfg.get("seed"),
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
    scaler = GradScaler(enabled=device.type == "cuda")

    data_iter = iter(dataloader)
    running_metrics: dict[str, float] = {}
    start_step = 1

    _ensure_dir(checkpoint_dir)
    
    # train
    for step in range(start_step, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
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
        n_masked_patches = batch["n_masked_patches"].to(device)

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

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable_params, grad_clip)

        scaler.step(optimizer)
        scaler.update()

        model.update_teacher(momentum)
        criterion.update_centers(
            teacher_cls_after_head=teacher_global["cls_after_head"].flatten(0, 1),
            teacher_masked_patch_after_head=teacher_global["masked_patch_after_head"],
        )

        loss_val = float(loss.detach().cpu())
        running_metrics.setdefault("loss", loss_val)
        running_metrics["loss"] = running_metrics["loss"] * 0.9 + loss_val * 0.1
        for key, value in loss_terms.items():
            val = float(value.detach().cpu())
            running_metrics[key] = running_metrics.get(key, val) * 0.9 + val * 0.1

        if step % log_interval == 0 or step == 1:
            metrics_str = " ".join(f"{k}:{v:.4f}" for k, v in sorted(running_metrics.items()))
            print(
                f"[Step {step}/{max_steps}] loss={loss_val:.4f} lr={lr_this_step:.6e} "
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
