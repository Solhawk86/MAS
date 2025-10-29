import math
import torch
from torch import nn

# 注意：本模块提供一个用于 ViT 自注意力 qkv 的 LoRA 包装器，只对 q 与 v 添加可训练低秩更新。
# 用法：将 dinov3.layers.attention.SelfAttention.qkv 替换为本包装器实例。


class LoRA_qkv(nn.Module):
    def __init__(self, base_qkv: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.qkv = base_qkv  # 原始 Linear/LinearKMaskedBias: in_features -> 3*C
        self.rank = rank
        self.alpha = alpha
        self.scaling = rank / alpha
        self.dropout = nn.Dropout(dropout)

        # 冻结原始 qkv 权重
        for p in self.qkv.parameters():
            p.requires_grad = False

        # 缓存 Linear 属性，避免外部再访问 self.qkv 时缺失
        self.in_features = self.qkv.in_features
        self.out_features = self.qkv.out_features
        in_features = self.in_features
        out_per_part = self.out_features // 3  # C（即每个 q/k/v 的维度）

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype

        # 为 q 和 v 分别构建 LoRA(A/B)
        self.lora_a_q = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b_q = nn.Linear(rank, out_per_part, bias=False, device=device, dtype=dtype)
        self.lora_a_v = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b_v = nn.Linear(rank, out_per_part, bias=False, device=device, dtype=dtype)

        # 初始化 LoRA 权重
        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

        self.out_per_part = out_per_part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础 qkv 变换
        base_out = self.qkv(x)  # [B, N, 3*C]

        # LoRA 路径，仅作用于 q 和 v 段
        # y = W x + (r/alpha) * B(Ax)
        lora_q = self.lora_b_q(self.lora_a_q(self.dropout(x))) * self.scaling  # [B, N, C]
        lora_v = self.lora_b_v(self.lora_a_v(self.dropout(x))) * self.scaling  # [B, N, C]

        C = self.out_per_part
        # q 段: [:C]，k 段: [C:2C]，v 段: [2C:3C]
        base_out[:, :, :C] = base_out[:, :, :C] + lora_q
        base_out[:, :, 2 * C :] = base_out[:, :, 2 * C :] + lora_v
        return base_out


def inject_lora_qv_into_vit(backbone: nn.Module, layers, rank: int, alpha: float, dropout: float = 0.0):
    """
    将 LoRA_qkv 注入到给定 ViT backbone 的指定 block 层，仅作用于每层自注意力的 q 与 v。

    参数:
        backbone: DINOv3 的 ViT 模型（DinoVisionTransformer）
        layers: 需要注入的 block 索引集合，支持：
            - int: 取最后 n 层
            - list/tuple/set[int]: 指定具体层号
            - "all": 全部层
        rank, alpha, dropout: LoRA 超参数
    """
    if isinstance(layers, str) and layers.lower() == "all":
        target_indices = list(range(getattr(backbone, "n_blocks", len(backbone.blocks))))
    elif isinstance(layers, int):
        n_blocks = getattr(backbone, "n_blocks", len(backbone.blocks))
        target_indices = list(range(max(0, n_blocks - layers), n_blocks))
    else:
        target_indices = list(layers)

    for i, blk in enumerate(backbone.blocks):
        if i not in target_indices:
            continue
        if not hasattr(blk, "attn") or not hasattr(blk.attn, "qkv"):
            continue
        blk.attn.qkv = LoRA_qkv(blk.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)


def test_dual_dinov3_lora():
    """
    简单测试：构建 DualImageDINOv3，加载权重，注入 LoRA，并检查输出特征形状。
    注意：依赖 config_eval.yaml 中的 pretrained_weights 路径有效。
    """
    import torch
    from MAS.model import DualImageDINOv3
    from MAS.utils import load_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("MAS/config_eval.yaml")

    # 例：在最后 4 层注入 LoRA，rank=8，alpha=16
    model = DualImageDINOv3(
        cfg=cfg,
        device=device,
        backbone_out_layers="FOUR_EVEN_INTERVALS",
        lora_layers=4,  # 最后 4 层
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)
    model.eval()

    img = torch.randn(2, 3, 518, 518, device=device)
    img_aug = torch.randn(2, 3, 518, 518, device=device)
    feat1, feat2 = model(img, img_aug)
    print(f"feat1 layers: {len(feat1)}; feat2 layers: {len(feat2)}")
    for i, f in enumerate(feat1, 1):
        if isinstance(f, tuple):
            print(f"L{i} patch {f[0].shape} cls {f[1].shape}")
        else:
            print(f"L{i} {f.shape}")

    return feat1, feat2
