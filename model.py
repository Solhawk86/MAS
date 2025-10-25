import time
import yaml
import torch
import torch.nn as nn
import sys
sys.path.append('/home/yangyan/hong/dinov3')
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Union
from utils import *
from dinov3.eval.segmentation.models import BackboneLayersSet, ModelWithIntermediateLayers
from Dino_LoRA import inject_lora_qv_into_vit, LoRA_qkv
from dinov3.models import build_model_for_eval  


class DualImageDINOv3(nn.Module):
    """
    模型类：DINOv3 ViT 作为 backbone，接收两个输入 (原图 + 增强图)，输出两者特征。

    参数:
        cfg: 可以是 dict / YAML 路径 / OmegaConf（包含 `student` 与 `crops` 配置）
        backbone_pth: 预训练权重路径，未提供时从 cfg['pretrained_weights'] 中读取
        backbone_type: 可选，覆盖 cfg['student']['arch']
        backbone_out_layers: 中间层索引选择
        device: 设备类型，例如 'cuda', 'cuda:0', 'cpu'
    """

    def __init__(
        self,
        cfg,
        backbone_pth: str | None = None,
        backbone_type: str | None = None,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        device: str = "cuda",
        # LoRA 注入相关可选参数
        lora_layers: int | list[int] | str | None = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        train_mode: bool | None = None,
    ):
        super(DualImageDINOv3, self).__init__()
        self.device = torch.device(device)

        # 构建并规范化 DINOv3 配置
        processed_cfg_dict = prepare_dinov3_cfg(cfg, backbone_type=backbone_type)
        self.cfg = dict_to_namespace(processed_cfg_dict)

        # 加载预训练权重路径
        pretrained_weights = backbone_pth or processed_cfg_dict.get("pretrained_weights")
        if not pretrained_weights:
            raise ValueError(
                "未在 cfg['pretrained_weights'] 中找到权重路径，也未显式传入 `backbone_pth`。"
            )

        # 加载 DINOv3 backbone（teacher 模型用于特征提取）
        self.backbone = build_model_for_eval(
            config=self.cfg,
            pretrained_weights=pretrained_weights,
        )
        self.backbone.eval()
        self.backbone.to(self.device)
        self.backbone.requires_grad_(False)

        # 在冻结原模型参数后，对指定层的 q 和 v 注入 LoRA
        self._lora_enabled = lora_layers is not None
        if self._lora_enabled:
            inject_lora_qv_into_vit(
                self.backbone,
                layers=lora_layers,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # 记录默认的训练模式：若显式传入则使用传入值，否则在启用 LoRA 时默认允许梯度
        self._train_mode_default = self._lora_enabled if train_mode is None else train_mode

        # 封装成可提取中间层输出的模型
        autocast_device_type = "cuda" if self.device.type == "cuda" else self.device.type
        self._backbone_indices = self._resolve_backbone_out_layers(self.backbone, backbone_out_layers)
        self._autocast_ctx = partial(
            torch.autocast,
            device_type=autocast_device_type,
            enabled=self.device.type == "cuda",
            dtype=torch.float32,
        )
        self.feature_extractor = ModelWithIntermediateLayers(   #官方实现，返回输入的feature
            self.backbone,
            n=self._backbone_indices,
            autocast_ctx=self._autocast_ctx,
            reshape=False,
            return_class_token=False,
        )

        # LoRA 层需要保持可训练（默认 requires_grad=True）。显式确认其状态，防止意外被冻结
        if self._lora_enabled:
            for module in self.backbone.modules():
                if isinstance(module, LoRA_qkv):
                    for param in module.parameters():
                        param.requires_grad = True

    def forward(self, img1, img2, *, train_mode: bool | None = None):
        """
        输入：
            img1: 原始图像，shape [B, 3, H, W]
            img2: 增强后图像，shape [B, 3, H, W]
        输出：
            feat1, feat2: 两个图像的特征列表（通常包含4层）
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        mode = self._train_mode_default if train_mode is None else train_mode

        if mode:
            feat1 = self._forward_with_grad(img1)
            feat2 = self._forward_with_grad(img2)
        else:
            feat1 = self._forward_inference(img1)
            feat2 = self._forward_inference(img2)
        return feat1, feat2

    def _forward_with_grad(self, images: torch.Tensor):
        '''
        训练模式下获取中间层特征，允许梯度在 LoRA 中回传
        层配置参考_resolve_backbone_out_layers
        '''
        with self._autocast_ctx():
            features = self.backbone.get_intermediate_layers(
                images,
                n=self._backbone_indices,
                reshape=False,
                return_class_token=False,
            )
        return features

    def _forward_inference(self, images: torch.Tensor):
        with torch.no_grad():
            return self.feature_extractor(images)

    @staticmethod
    def _resolve_backbone_out_layers(model: nn.Module, backbone_out_layers):
        """
        将层配置转换成具体的层索引或数量。支持：
        - BackboneLayersSet 枚举值
        - 枚举名称字符串
        - 具体的整数列表 / 元组
        - 单个整数（取最后 n 层）
        """
        if isinstance(backbone_out_layers, BackboneLayersSet):
            layerset = backbone_out_layers
        elif isinstance(backbone_out_layers, str):
            try:
                layerset = BackboneLayersSet[backbone_out_layers]
            except KeyError:
                raise ValueError(f"无法识别的 backbone_out_layers: {backbone_out_layers}") from None
        else:
            layerset = None

        if layerset is not None:
            return DualImageDINOv3._layerset_to_indices(model, layerset)

        if isinstance(backbone_out_layers, int):
            n_blocks = getattr(model, "n_blocks", None)
            if n_blocks is None:
                raise ValueError("无法根据整数推断层索引，因为模型缺少 n_blocks 属性")
            if backbone_out_layers <= 0:
                raise ValueError("backbone_out_layers 必须是正整数")
            indices = list(range(max(0, n_blocks - backbone_out_layers), n_blocks))
            return indices

        if isinstance(backbone_out_layers, Iterable):
            indices = list(backbone_out_layers)
            if not indices:
                raise ValueError("backbone_out_layers 序列不能为空")
            return indices

        raise TypeError(
            "backbone_out_layers 类型不支持，请传入 BackboneLayersSet 枚举、枚举名、整数或整数序列"
        )

    @staticmethod
    def _layerset_to_indices(model: nn.Module, layerset: BackboneLayersSet) -> list[int]:
        n_blocks = getattr(model, "n_blocks", None)
        if n_blocks is None:
            raise ValueError("模型缺少 n_blocks 属性，无法确定中间层索引")

        if layerset == BackboneLayersSet.LAST:
            indices = [n_blocks - 1]
        elif layerset == BackboneLayersSet.FOUR_LAST:
            indices = list(range(max(0, n_blocks - 4), n_blocks))
        elif layerset == BackboneLayersSet.FOUR_EVEN_INTERVALS:
            if n_blocks == 24:
                indices = [4, 11, 17, 23] # used in paper
            else:
                step = max(1, n_blocks // 4)
                indices = [min(n_blocks - 1, step * i - 1) for i in range(1, 5)]
        else:
            raise ValueError(f"暂不支持的 BackboneLayersSet: {layerset}")
        if any(idx >= n_blocks or idx < 0 for idx in indices):
            raise ValueError(f"生成的层索引 {indices} 超出模型 block 范围 (0, {n_blocks - 1})")
        return indices
    
def test():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    cfg = load_config('config_eval.yaml')

    model = DualImageDINOv3(cfg=cfg, device=device).to(device)
    model.eval()
    print(next(model.backbone.parameters()).device)
    
    img = torch.randn(2, 3, 518, 518)
    img_aug = torch.randn(2, 3, 518, 518)

    feat1, feat2 = model(img, img_aug)
    print(f"原图特征层数: {len(feat1)}")
    for i, feat in enumerate(feat1, start=1):
        if isinstance(feat, tuple):
            patch_feat, cls_feat = feat
            print(f"层 {i}: patch特征 {patch_feat.shape}, cls特征 {cls_feat.shape}")
        else:
            print(f"层 {i}: {feat.shape}")

    print(f"增强图特征层数: {len(feat2)}")
    for i, feat in enumerate(feat2, start=1):
        if isinstance(feat, tuple):
            patch_feat, cls_feat = feat
            print(f"层 {i}: patch特征 {patch_feat.shape}, cls特征 {cls_feat.shape}")
        else:
            print(f"层 {i}: {feat.shape}")

if __name__ == "__main__":
    test()
