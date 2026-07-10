from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


def _to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, c, h, w = x.shape
    head_dim = c // num_heads
    return x.view(b, num_heads, head_dim, h * w)


def _from_heads(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, heads, c, _ = x.shape
    return x.view(b, heads * c, h, w)


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        weight = self.weight.view(1, -1, 1, 1)
        if self.bias is None:
            return x * weight
        return x * weight + self.bias.view(1, -1, 1, 1)


class AttentionBase(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q, k, v = self.qkv2(self.qkv1(x)).chunk(3, dim=1)
        q = F.normalize(_to_heads(q, self.num_heads), dim=-1)
        k = F.normalize(_to_heads(k, self.num_heads), dim=-1)
        v = _to_heads(v, self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ v
        return self.proj(_from_heads(out, h, w))


class Mlp(nn.Module):
    def __init__(self, in_features: int, ffn_expansion_factor: float = 2.0, bias: bool = False) -> None:
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 8, ffn_expansion_factor: float = 1.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, bias=True)
        self.attn = AttentionBase(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim, bias=True)
        self.mlp = Mlp(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp: int, oup: int, expand_ratio: int) -> None:
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.block = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetailNode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shuffle_conv = nn.Conv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self.shuffle_conv(torch.cat((z1, z2), dim=1)).chunk(2, dim=1)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(torch.clamp(self.theta_rho(z2), min=-5.0, max=5.0)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DetailNode() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1, z2 = x.chunk(2, dim=1)
        for layer in self.layers:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(_to_heads(q, self.num_heads), dim=-1)
        k = F.normalize(_to_heads(k, self.num_heads), dim=-1)
        v = _to_heads(v, self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v
        return self.project_out(_from_heads(out, h, w))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, bias=True)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, bias=True)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RestormerEncoder(nn.Module):
    def __init__(
        self,
        inp_channels: int = 1,
        dim: int = 64,
        num_blocks: Sequence[int] = (4, 4),
        heads: Sequence[int] = (8, 8, 8),
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
                for _ in range(int(num_blocks[0]))
            ]
        )
        self.base_feature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detail_feature = DetailFeatureExtraction()

    def forward(self, inp_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.encoder_level1(self.patch_embed(inp_img))
        return self.base_feature(feature), self.detail_feature(feature), feature


class RestormerDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        dim: int = 64,
        num_blocks: Sequence[int] = (4, 4),
        heads: Sequence[int] = (8, 8, 8),
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.reduce_channel = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(dim, heads[1], ffn_expansion_factor, bias)
                for _ in range(int(num_blocks[1]))
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(
        self,
        inp_img: torch.Tensor | None,
        base_feature: torch.Tensor,
        detail_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_feature = self.reduce_channel(torch.cat((base_feature, detail_feature), dim=1))
        decoded = self.encoder_level2(fused_feature)
        decoded = self.output(decoded)
        if inp_img is not None:
            decoded = decoded + inp_img
        return torch.sigmoid(decoded), fused_feature


class _PriorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Sobelxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernelx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        kernely = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.register_buffer("weightx", kernelx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("weighty", kernely.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weightx = self.weightx.to(device=x.device, dtype=x.dtype)
        weighty = self.weighty.to(device=x.device, dtype=x.dtype)
        return torch.abs(F.conv2d(x, weightx, padding=1)) + torch.abs(F.conv2d(x, weighty, padding=1))


def _cc(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    n, c, _, _ = img1.shape
    x = img1.float().reshape(n, c, -1)
    y = img2.float().reshape(n, c, -1)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    denom = eps + torch.sqrt(torch.sum(x**2, dim=-1)) * torch.sqrt(torch.sum(y**2, dim=-1))
    return torch.clamp(torch.sum(x * y, dim=-1) / denom, -1.0, 1.0).mean()


class CDDFuseDepthPriorAdd(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        feature_order: Sequence[str],
        image_mean: Sequence[float] | None,
        image_std: Sequence[float] | None,
        prior_channels: int = 64,
        gamma_init: float = 0.1,
        fusion_loss_weight: float = 1.0,
        decomp_loss_weight: float = 2.0,
        fusion_size: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_order = list(feature_order)
        self.gamma_init = float(gamma_init)
        self.fusion_loss_weight = float(fusion_loss_weight)
        self.decomp_loss_weight = float(decomp_loss_weight)
        self.fusion_size = int(fusion_size)

        self.register_buffer("image_mean", self._as_image_buffer(image_mean), persistent=False)
        self.register_buffer("image_std", self._as_image_buffer(image_std), persistent=False)
        self._uses_255_stats = bool(
            image_mean
            and image_std
            and (
                any(abs(float(v)) > 1.0 for v in image_mean)
                or any(abs(float(v)) > 1.0 for v in image_std)
            )
        )

        c = int(prior_channels)
        self.encoder = RestormerEncoder(inp_channels=1, dim=c)
        self.decoder = RestormerDecoder(out_channels=1, dim=c)
        self.decoder.encoder_level2.requires_grad_(False)
        self.decoder.output.requires_grad_(False)
        self.base_fuse = BaseFeatureExtraction(dim=c, num_heads=8)
        self.detail_fuse = DetailFeatureExtraction(num_layers=1)

        self.pyramid = nn.ModuleList(
            [
                _PriorBlock(c, c, stride=4),
                _PriorBlock(c, c * 2, stride=2),
                _PriorBlock(c * 2, c * 4, stride=2),
                _PriorBlock(c * 4, c * 4, stride=2),
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(c, self.embed_dim, kernel_size=1),
                nn.Conv2d(c * 2, self.embed_dim, kernel_size=1),
                nn.Conv2d(c * 4, self.embed_dim, kernel_size=1),
                nn.Conv2d(c * 4, self.embed_dim, kernel_size=1),
            ]
        )
        self.gamma = nn.Parameter(torch.full((len(self.feature_order),), self.gamma_init))
        self.sobel = Sobelxy()
        self.last_losses: dict[str, torch.Tensor] = {}
        self.last_raw_losses: dict[str, torch.Tensor] = {}

    @staticmethod
    def _as_image_buffer(values: Sequence[float] | None) -> torch.Tensor:
        if values is None:
            return torch.empty(0)
        return torch.tensor([float(v) for v in values], dtype=torch.float32).view(1, -1, 1, 1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        depth_maps: torch.Tensor | None,
        features_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if depth_maps is None:
            raise ValueError("use_cddfuse_depth_prior_add=True 时 forward 必须提供 depth_maps")
        if not features_dict:
            return dict(features_dict)

        image01 = self._restore_image01(pixel_values)
        gray = self._rgb_to_gray(image01)
        depth = self._normalize_depth(depth_maps, target_size=gray.shape[-2:])
        gray, depth = self._resize_fusion_inputs(gray, depth)

        vis_base, vis_detail, _ = self.encoder(gray)
        dep_base, dep_detail, _ = self.encoder(depth)
        fused_base = self.base_fuse(vis_base + dep_base)
        fused_detail = self.detail_fuse(vis_detail + dep_detail)
        fused_img, fused_feature = self.decoder(gray, fused_base, fused_detail)

        prior_pyramid = self._build_prior_pyramid(fused_feature)
        fused = dict(features_dict)
        prior_abs_values: list[torch.Tensor] = []
        target_abs_values: list[torch.Tensor] = []
        contribution_abs_values: list[torch.Tensor] = []
        for level_idx, key in enumerate(self.feature_order):
            target = features_dict.get(key)
            if target is None:
                continue
            prior = self.projections[level_idx](prior_pyramid[level_idx])
            if prior.shape[-2:] != target.shape[-2:]:
                prior = F.interpolate(prior, size=target.shape[-2:], mode="bilinear", align_corners=False)
            gamma = self.gamma[level_idx].to(device=target.device, dtype=target.dtype)
            prior = prior.to(dtype=target.dtype)
            contribution = gamma * prior
            fused[key] = target + contribution
            prior_abs_values.append(prior.float().abs().flatten(1).mean(dim=1))
            target_abs_values.append(target.detach().float().abs().flatten(1).mean(dim=1))
            contribution_abs_values.append(contribution.float().abs().flatten(1).mean(dim=1))

        self._update_losses(
            gray,
            depth,
            fused_img,
            vis_base,
            dep_base,
            vis_detail,
            dep_detail,
            prior_abs_values=prior_abs_values,
            target_abs_values=target_abs_values,
            contribution_abs_values=contribution_abs_values,
        )
        return fused

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        return {}

    def raw_losses(self) -> dict[str, torch.Tensor]:
        return dict(self.last_raw_losses)

    def _restore_image01(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image = pixel_values.float()
        if self.image_mean.numel() and self.image_std.numel():
            mean = self.image_mean.to(device=image.device, dtype=image.dtype)
            std = self.image_std.to(device=image.device, dtype=image.dtype)
            image = image * std + mean
        if self._uses_255_stats:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    @staticmethod
    def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return image
        return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    @staticmethod
    def _normalize_depth(depth_maps: torch.Tensor, *, target_size: tuple[int, int]) -> torch.Tensor:
        depth = depth_maps.float()
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4:
            raise ValueError(f"depth_maps 形状应为 [B,1,H,W]，实际为 {tuple(depth.shape)}")
        if depth.shape[1] != 1:
            depth = depth[:, :1]
        if depth.shape[-2:] != target_size:
            depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)
        flat = depth.flatten(1)
        dmin = flat.min(dim=1).values.view(-1, 1, 1, 1)
        dmax = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return ((depth - dmin) / (dmax - dmin).clamp_min(1e-6)).clamp(0.0, 1.0)

    def _resize_fusion_inputs(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_size <= 0:
            return gray, depth
        h, w = gray.shape[-2:]
        if max(h, w) <= self.fusion_size:
            return gray, depth
        scale = float(self.fusion_size) / float(max(h, w))
        target = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
        gray = F.interpolate(gray, size=target, mode="bilinear", align_corners=False)
        depth = F.interpolate(depth, size=target, mode="bilinear", align_corners=False)
        return gray, depth

    def _build_prior_pyramid(self, fused_feature: torch.Tensor) -> list[torch.Tensor]:
        p1 = self.pyramid[0](fused_feature)
        p2 = self.pyramid[1](p1)
        p3 = self.pyramid[2](p2)
        p4 = self.pyramid[3](p3)
        return [p1, p2, p3, p4]

    def _update_losses(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
        fused_img: torch.Tensor,
        vis_base: torch.Tensor,
        dep_base: torch.Tensor,
        vis_detail: torch.Tensor,
        dep_detail: torch.Tensor,
        prior_abs_values: list[torch.Tensor],
        target_abs_values: list[torch.Tensor],
        contribution_abs_values: list[torch.Tensor],
    ) -> None:
        target_intensity = torch.max(gray, depth)
        loss_in = F.l1_loss(target_intensity, fused_img)
        joint_grad = torch.max(self.sobel(gray), self.sobel(depth))
        loss_grad = F.l1_loss(joint_grad, self.sobel(fused_img))
        fusion_loss = loss_in + 10.0 * loss_grad
        base_cc = _cc(vis_base, dep_base)
        detail_cc = _cc(vis_detail, dep_detail)
        decomp_loss = detail_cc.pow(2) / (1.01 + base_cc)
        self.last_losses = {}
        self.last_raw_losses = {
            "loss_cddfuse_fusion": fusion_loss.detach(),
            "loss_cddfuse_decomp": decomp_loss.detach(),
            "loss_cddfuse_total": (fusion_loss + decomp_loss).detach(),
            "loss_cddfuse_fusion_raw": fusion_loss.detach(),
            "loss_cddfuse_decomp_raw": decomp_loss.detach(),
            "loss_cddfuse_in": loss_in.detach(),
            "loss_cddfuse_grad": loss_grad.detach(),
            "cddfuse/base_cc": base_cc.detach(),
            "cddfuse/detail_cc": detail_cc.detach(),
            "cddfuse/gray_mean": gray.detach().mean(),
            "cddfuse/depth_mean": depth.detach().mean(),
            "cddfuse/fused_img_mean": fused_img.detach().mean(),
            "cddfuse/fused_img_std": fused_img.detach().std(unbiased=False),
            "cddfuse/gamma_mean": self.gamma.detach().mean(),
        }
        if prior_abs_values and target_abs_values and contribution_abs_values:
            prior_abs = torch.cat(prior_abs_values).mean()
            target_abs = torch.cat(target_abs_values).mean()
            contribution_abs = torch.cat(contribution_abs_values).mean()
            self.last_raw_losses.update(
                {
                    "cddfuse/prior_abs": prior_abs.detach(),
                    "cddfuse/target_abs": target_abs.detach(),
                    "cddfuse/prior_strength": (prior_abs / (target_abs + 1e-6)).detach(),
                    "cddfuse/contribution_abs": contribution_abs.detach(),
                    "cddfuse/contribution_strength": (contribution_abs / (target_abs + 1e-6)).detach(),
                }
            )
        for idx, value in enumerate(self.gamma.detach()):
            self.last_raw_losses[f"cddfuse/gamma_level_{idx + 1}"] = value
