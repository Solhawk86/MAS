# loss.py
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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
        self.dino_global_ignore_diagonal = bool(
            dino_cfg.get("global_ignore_diagonal", False)
        )
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
        loss_dict: dict[str, torch.Tensor] = (
            {}
        )  # 额外loss接口，在日志或 tensorboard 中显示，不影响梯度回传

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
        loss_total = (
            loss_total + self.dino_loss_weight * dino_local_scale * dino_local_loss
        )

        # Global crops vs. global teacher
        dino_global_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global"] = dino_global_loss.detach()
        loss_total = (
            loss_total + self.dino_loss_weight * dino_global_scale * dino_global_loss
        )

        # KoLeo regularisation on student global CLS tokens
        if self.koleo_loss_weight > 0:
            koleo = (
                sum(self.koleo_loss(x) for x in student_global["cls_pre_head"])
                / n_global_crops
            )
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


class LaplacianClarity(nn.Module):
    """
    Compute clarity map as depth/degradation proxy
    """

    def __init__(self):
        super().__init__()
        kernel = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.register_buffer("kernel", kernel)

    def forward(self, img):
        """
        img: [B, 3, H, W], range [0,1]
        return: [B, 1, H, W]
        """
        gray = img.mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, self.kernel, padding=1)
        clarity = lap.abs()

        # normalize per image
        clarity = clarity / (clarity.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return clarity


class QueryDepthConsistencyLoss(nn.Module):

    def __init__(self, num_queries, hidden_dim, div_loss_weight, num_depth=3):
        """
        num_queries: 查询数量
        hidden_dim: depth latent 维度
        num_depth: depth embedding 组数（near/mid/far）
        div_loss_weight: diversity loss 权重
        """
        super().__init__()
        self.depth_proj = nn.Linear(hidden_dim, 1)
        self.num_depth = num_depth
        self.div_loss_weight = div_loss_weight

    def forward(self, depth_latent, masks, clarity_map, depth_ids):
        """
        depth_latent: [num_queries, hidden_dim] 每个 query 的 latent
        masks:        [B, num_queries, H, W] mask logits
        clarity_map:  [B, 1, H, W] 退化深度 map (高值清晰)
        depth_ids:    [num_queries] 每个 query 属于哪组 depth (0~num_depth-1)
        """
        B, Q, H, W = masks.shape

        if clarity_map.shape[-2:] != (H, W):
            clarity_map = F.interpolate(
                clarity_map,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if depth_latent.dim() == 3:
            depth_latent = depth_latent.mean(dim=1)

        #  将 depth latent 投影为标量 depth
        query_depth = self.depth_proj(depth_latent).squeeze(-1)  # [Q]

        masks_detach = masks.detach()  # 不反传至主分割头
        clarity = clarity_map  # [B,1,H,W]

        num = (masks_detach * clarity).flatten(2).sum(dim=2)
        den = masks_detach.flatten(2).sum(dim=2).clamp_min(1e-6)
        query_region_depth = (num / den).mean(dim=0)

        # query depth consistency loss
        loss_depth = F.l1_loss(query_depth, query_region_depth)

        # 计算三组 query 的平均退化深度
        depth_means: list[torch.Tensor] = []
        zero = torch.zeros((), device=query_region_depth.device)
        for k in range(self.num_depth):
            idx = depth_ids == k
            depth_means.append(query_region_depth[idx].mean() if idx.any() else zero)
        E_avg_depth = torch.stack(depth_means, dim=0)

        # diversity loss: 两两组平均退化深度差异
        loss_div = 0
        for i in range(self.num_depth):
            for j in range(i + 1, self.num_depth):
                loss_div += (E_avg_depth[i] - E_avg_depth[j]) ** 2

        # 总 loss
        loss = loss_depth - self.div_loss_weight * loss_div

        return loss


def _demo_laplacian_clarity(image_path: str, output_path: str) -> None:
    """将输入图像通过 LaplacianClarity 生成 0-255 的可视化 PNG。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(Path(image_path)).convert("RGB")
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    clarity_map = LaplacianClarity().to(device)(img_tensor)
    clarity_np = clarity_map.squeeze(0).squeeze(0).detach().cpu().numpy()
    clarity_np = clarity_np / (clarity_np.max() + 1e-8)
    clarity_img = (clarity_np * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(clarity_img, mode="L").save(Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaplacianClarity 可视化测试")
    parser.add_argument("input", type=str, help="输入图像路径")
    parser.add_argument("output", type=str, help="输出 PNG 路径")
    args = parser.parse_args()
    _demo_laplacian_clarity(args.input, args.output)
