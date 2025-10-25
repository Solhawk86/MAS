import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from MAS.utils import load_config
from model import DualImageDINOv3
from data.data import Dataset, make_transform, make_enhance_transform, InfiniteDualDataloader
from loss import self_supervised_loss
from MAS.Dino_LoRA import LoRA_qkv


def _collect_lora_parameters(model: DualImageDINOv3) -> list[torch.nn.Parameter]:
    """只返回 LoRA_qkv 的可训练参数，用于构建优化器。"""
    params: list[torch.nn.Parameter] = []
    for module in model.backbone.modules():
        if isinstance(module, LoRA_qkv):
            params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def _resolve_base_dir(input_cfg: dict) -> str:
    paths = input_cfg.get("paths", "") if input_cfg else ""
    if isinstance(paths, (list, tuple)):
        return paths[0] if paths else ""
    return paths


def train():
    cfg = load_config("MAS/config_eval.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    batch_size = train_cfg.get("batchsize", 8)
    num_workers = train_cfg.get("num_workers", 4)
    max_steps = train_cfg.get("steps", train_cfg.get("max_steps", 50000))
    lora_layers = train_cfg.get("lora_layers", 4)
    lora_rank = train_cfg.get("lora_rank", 8)
    lora_alpha = train_cfg.get("lora_alpha", 16.0)
    lora_dropout = train_cfg.get("lora_dropout", 0.0)
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    eta_min = train_cfg.get("min_lr", 0.0)
    image_size = train_cfg.get("image_size", cfg.get("crops", {}).get("global_crops_size", 518))
    log_interval = train_cfg.get("log_interval", 50)

    input_cfg = cfg.get("input_images", {})
    base_dir = _resolve_base_dir(input_cfg)
    dataset_name = input_cfg.get("dataset_name", "")
    if not base_dir or not dataset_name:
        raise ValueError("配置中的 input_images.paths 或 dataset_name 未设置有效路径。")

    raw_transform = make_transform(image_size)
    enhance_transform = make_enhance_transform(image_size)

    raw_dataset = Dataset(base_dir=base_dir, dataset_name=dataset_name, transform=raw_transform)
    enhance_dataset = Dataset(base_dir=base_dir, dataset_name=dataset_name, transform=enhance_transform)

    # 构建无限数据流，确保微调阶段可以长时间迭代
    dual_loader = InfiniteDualDataloader(
        raw_dataset,
        enhance_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_steps=max_steps,
    )

    # 启用 LoRA 并让模型默认处于可训练模式（原 backbone 权重仍被冻结）
    backbone_out_layers = model_cfg.get("backbone_out_layers", "FOUR_EVEN_INTERVALS")

    model = DualImageDINOv3(
        cfg=cfg,
        device=device,
        backbone_out_layers=backbone_out_layers,
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_mode=True,
    ).to(device)
    model.train()

    # 仅优化 LoRA 权重，避免动到预训练 Backbone
    lora_params = _collect_lora_parameters(model)
    if not lora_params:
        raise RuntimeError("未找到可训练的 LoRA 参数，请确认已正确注入。")

    optimizer = AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=eta_min)

    for step, (raw_batch, enhance_batch) in enumerate(dual_loader, start=1):
        # DataLoader 返回 (img, mask, name)，这里只取图像部分
        raw_imgs = raw_batch[0].to(device, non_blocking=True)
        enhance_imgs = enhance_batch[0].to(device, non_blocking=True)

        # 前向得到多层特征列表，取最后一层构建自监督损失
        feat1, feat2 = model(raw_imgs, enhance_imgs, train_mode=True)
        feat1_last = feat1[-1]
        feat2_last = feat2[-1]

        loss = self_supervised_loss(feat1_last, feat2_last)

        # 典型的优化流程：梯度清零 -> 反向传播 -> 参数更新 -> 调度学习率
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == 1 or step == max_steps:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Step {step}/{max_steps}] loss={loss.item():.6f} lr={current_lr:.6e}")

    print("训练结束。")


if __name__ == "__main__":
    train()
