# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.loss.dino_clstoken_loss import DINOLoss
from dinov3.loss.ibot_patch_loss import iBOTPatchLoss
from dinov3.loss.koleo_loss import KoLeoLoss

def l1_loss(pred, target):
    return F.l1_loss(pred, target)


class SelfSupervisedLoss(nn.Module):
    """Combine DINO, iBOT and KoLeo losses for student-teacher training."""

    def __init__(self, *, dino_cfg: dict, ibot_cfg: dict, koleo_cfg: dict):
        super().__init__()
        dino_out_dim = dino_cfg.get("head_n_prototypes", 8192)
        ibot_out_dim = ibot_cfg.get("head_n_prototypes", 8192)

        self.dino_loss = DINOLoss(
            out_dim=dino_out_dim,
            student_temp=dino_cfg.get("student_temp", 0.1),
            center_momentum=dino_cfg.get("center_momentum", 0.9),
        )
        self.ibot_patch_loss = iBOTPatchLoss(
            patch_out_dim=ibot_out_dim,
            student_temp=ibot_cfg.get("student_temp", 0.1),
            center_momentum=ibot_cfg.get("center_momentum", 0.9),
        )
        self.koleo_loss = KoLeoLoss()

        self.dino_loss_weight = float(dino_cfg.get("loss_weight", 1.0))
        self.dino_global_ignore_diagonal = bool(dino_cfg.get("global_ignore_diagonal", False))
        self.koleo_loss_weight = float(koleo_cfg.get("loss_weight", 0.0))
        self.ibot_loss_weight = float(ibot_cfg.get("loss_weight", 1.0))

    def forward(
        self,
        *,
        teacher_global: dict[str, torch.Tensor],
        student_global: dict[str, torch.Tensor],
        student_local: dict[str, torch.Tensor],
        masks: torch.Tensor,
        mask_indices_list: torch.Tensor,
        masks_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # n_global_crops=2，n_local_crops 通常为6，形状与 DINOv3 主干保持一致
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict: dict[str, torch.Tensor] = {} # 额外loss接口，在日志或 tensorboard 中显示，不影响梯度回传

        device = student_global["cls_after_head"].device
        loss_total = torch.zeros((), device=device)

        # 复用官方实现的权重系数，平衡 global/local 匹配次数
        # 如果简单把所有匹配加权平均，因为 local crops 数量多，local loss 会占总 loss 的比例过大，使全局特征训练不足。
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1)
            if self.dino_global_ignore_diagonal
            else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)

        # Local crops vs. global teacher
        dino_local_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local"] = dino_local_loss.detach()
        loss_total = loss_total + self.dino_loss_weight * dino_local_scale * dino_local_loss

        # Global crops vs. global teacher
        dino_global_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global"] = dino_global_loss.detach()
        loss_total = loss_total + self.dino_loss_weight * dino_global_scale * dino_global_loss

        # KoLeo regularisation on student global CLS tokens
        if self.koleo_loss_weight > 0:
            koleo = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
            loss_dict["koleo"] = koleo.detach()
            loss_total = loss_total + self.koleo_loss_weight * n_global_crops * koleo

        # iBOT masked patch prediction
        if self.ibot_loss_weight > 0 and mask_indices_list.numel() > 0:
            ibot = self.ibot_patch_loss.forward_masked(
                student_global["masked_patch_after_head"],
                teacher_global["masked_patch_centered"],
                student_masks_flat=masks,
                n_masked_patches=mask_indices_list.shape[0],
                masks_weight=masks_weight,
            )
            loss_dict["ibot"] = ibot.detach()
            loss_total = loss_total + self.ibot_loss_weight * ibot

        return loss_total, loss_dict

    @torch.no_grad()
    def update_centers(
        self,
        teacher_cls_after_head: torch.Tensor,
        teacher_masked_patch_after_head: torch.Tensor,
    ) -> None:
        # EMA 更新中心向量，确保教师输出稳定
        self.dino_loss.update_center(teacher_cls_after_head)
        if teacher_masked_patch_after_head.numel() > 0:
            self.ibot_patch_loss.update_center(teacher_masked_patch_after_head)
