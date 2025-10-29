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
from dinov3.layers.dino_head import DINOHead
from Dino_LoRA import inject_lora_qv_into_vit, LoRA_qkv
from dinov3.models import build_model_for_eval  

# NOTE: DualImageDINOv3 保留供分割任务调用；DINOv3SelfSupervised 则负责自监督 LoRA 训练。


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


class DINOv3SelfSupervised(nn.Module):
    """Student-teacher wrapper tailored for LoRA-only self-supervised pretraining."""

    def __init__(
        self,
        cfg,
        *,
        device: str = "cuda",
        lora_layers: int | list[int] | str | None = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        ssl_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        processed_cfg_dict = prepare_dinov3_cfg(cfg)
        self.cfg = dict_to_namespace(processed_cfg_dict)
        self.ssl_cfg = ssl_cfg or {}

        pretrained_weights = processed_cfg_dict.get("pretrained_weights")
        if not pretrained_weights:
            raise ValueError("自监督预训练需要配置 pretrained_weights 以初始化教师模型")

        # 通过预训练权重构建教师模型，并复制给学生
        teacher_backbone = build_model_for_eval(
            config=self.cfg,
            pretrained_weights=pretrained_weights,
        )
        teacher_backbone.to(self.device)
        teacher_backbone.eval()

        student_backbone = deepcopy(teacher_backbone)
        student_backbone.to(self.device)
        student_backbone.train()
        
        # 注入LoRA
        self._lora_enabled = lora_layers is not None
        if self._lora_enabled:
            inject_lora_qv_into_vit(
                student_backbone,
                layers=lora_layers,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            inject_lora_qv_into_vit(
                teacher_backbone,
                layers=lora_layers,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        # 注意：teacher/backbone 仅为了结构对齐而注入 LoRA，参数由 EMA 驱动

        student_backbone.requires_grad_(False)
        teacher_backbone.requires_grad_(False)

        if self._lora_enabled:
            self._set_lora_trainable(student_backbone)

        dino_head_cfg = self.ssl_cfg.get("dino", {})
        ibot_head_cfg = self.ssl_cfg.get("ibot", {})

        embed_dim = getattr(student_backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("无法在 backbone 上找到 embed_dim")

        # 构建 Head
        student_dino_head = self._build_head(embed_dim, dino_head_cfg).to(self.device)
        student_ibot_head = self._build_head(embed_dim, ibot_head_cfg).to(self.device)
        teacher_dino_head = self._build_head(embed_dim, dino_head_cfg).to(self.device)
        teacher_ibot_head = self._build_head(embed_dim, ibot_head_cfg).to(self.device)

        teacher_dino_head.load_state_dict(student_dino_head.state_dict())
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        teacher_dino_head.requires_grad_(False)
        teacher_ibot_head.requires_grad_(False)
        teacher_dino_head.eval()
        teacher_ibot_head.eval()

        # student/backbone 搭配 LoRA，可训练；teacher 全量冻结，仅通过 EMA 同步
        self.student = nn.ModuleDict(
            {
                "backbone": student_backbone,
                "dino_head": student_dino_head,
                "ibot_head": student_ibot_head,
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": teacher_backbone,
                "dino_head": teacher_dino_head,
                "ibot_head": teacher_ibot_head,
            }
        )
        self.student.train()
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    @staticmethod
    def _build_head(embed_dim: int, head_cfg: Mapping[str, Any]) -> DINOHead:
        out_dim = head_cfg.get("head_n_prototypes", 8192)
        hidden_dim = head_cfg.get("head_hidden_dim", 2048)
        bottleneck_dim = head_cfg.get("head_bottleneck_dim", 256)
        nlayers = head_cfg.get("head_nlayers", 3)
        use_bn = head_cfg.get("use_bn", False)
        mlp_bias = head_cfg.get("mlp_bias", True)
        # DINOv3 代码中不再使用 norm_last_layer，这里保持兼容但忽略该参数
        return DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            nlayers=nlayers,
            use_bn=use_bn,
            mlp_bias=mlp_bias,
        )

    @staticmethod
    def _set_lora_trainable(backbone: nn.Module) -> None:
        for module in backbone.modules():
            if isinstance(module, LoRA_qkv):
                for param in module.parameters():
                    param.requires_grad = True

    def student_trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for name, module in self.student.items():
            if name == "backbone":
                params.extend(p for p in module.parameters() if p.requires_grad)
            else:
                params.extend(module.parameters())
        return params

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        teacher_params = []
        student_params = []
        for module_name in self.student.keys():
            student_mod = self.student[module_name]
            teacher_mod = self.teacher[module_name]
            for sp, tp in zip(student_mod.parameters(), teacher_mod.parameters()):
                if not tp.data.is_floating_point():
                    continue
                teacher_params.append(tp)
                student_params.append(sp)
        # 按 EMA momentum 同步教师参数，避免显存回传
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)

    def get_teacher_output(
        self,
        images: torch.Tensor,
        *,
        mask_indices_list: torch.Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: torch.Tensor,
        dino_loss,
        ibot_loss,
    ) -> dict[str, torch.Tensor]:
        n_crops, B, _, H, W = images.shape
        flat_images = images.flatten(0, 1)

        with torch.no_grad():
            backbone_out = self.teacher["backbone"](flat_images, is_training=True)
        # 官方封装输出字典
        # output.append(
        #         {
        #             "x_norm_clstoken": x_norm_cls_reg[:, 0],
        #             "x_storage_tokens": x_norm_cls_reg[:, 1:],
        #             "x_norm_patchtokens": x_norm_patch,
        #             "x_prenorm": x,
        #             "masks": masks,
        #         }
        cls = backbone_out["x_norm_clstoken"]
        reg = backbone_out["x_storage_tokens"]
        patch = backbone_out["x_norm_patchtokens"]

        # 只在被遮挡的 patch 上执行 iBOT head，降低计算量
        masked_patch_pre_head = torch.index_select(patch.flatten(0, 1), 0, mask_indices_list)
        masked_patch_after_head = self.teacher["ibot_head"](masked_patch_pre_head)

        cls_after_head = self.teacher["dino_head"](cls)
        cls_centered = dino_loss.sinkhorn_knopp_teacher(cls_after_head, teacher_temp=teacher_temp)  #温度归一化
        cls_centered = cls_centered.unflatten(0, (n_crops, B))
        masked_patch_centered = ibot_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )

        return {
            "cls_pre_head": cls.unflatten(0, (n_crops, B)),
            "reg_pre_head": reg.unflatten(0, (n_crops, B)),
            "patch_pre_head": patch.unflatten(0, (n_crops, B)),
            "cls_after_head": cls_after_head.unflatten(0, (n_crops, B)),
            "cls_centered": cls_centered,
            "masked_patch_centered": masked_patch_centered,
            "masked_patch_after_head": masked_patch_after_head,
        }

    def get_student_output(
        self,
        *,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        masks: torch.Tensor,
        mask_indices_list: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        n_global_crops, B, _, H, W = global_crops.shape
        n_local_crops, _, _, h_local, w_local = local_crops.shape

        global_flat = global_crops.flatten(0, 1)
        local_flat = local_crops.flatten(0, 1)

        global_out, local_out = self.student["backbone"](
            [global_flat, local_flat],
            masks=[masks, None],
            is_training=True,
        )

        g_cls = global_out["x_norm_clstoken"]
        g_reg = global_out["x_storage_tokens"]
        g_patch = global_out["x_norm_patchtokens"]
        l_cls = local_out["x_norm_clstoken"]
        l_reg = local_out["x_storage_tokens"]
        l_patch = local_out["x_norm_patchtokens"]

        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), 0, mask_indices_list)
        masked_patch_after_head = self.student["ibot_head"](masked_patches_pre_head)

        buffer = [g_cls, l_cls]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)
        buffer = self.student["dino_head"](buffer)
        g_cls_after_head, l_cls_after_head = torch.split(buffer, sizes, dim=0)

        student_global = {
            "cls_pre_head": g_cls.unflatten(0, (n_global_crops, B)),
            "reg_pre_head": g_reg.unflatten(0, (n_global_crops, B)),
            "patch_pre_head": g_patch.unflatten(0, (n_global_crops, B)),
            "cls_after_head": g_cls_after_head.unflatten(0, (n_global_crops, B)),
            "masked_patch_after_head": masked_patch_after_head,
            "masked_patch_pre_head": masked_patches_pre_head,
        }
        student_local = {
            "cls_pre_head": l_cls.unflatten(0, (n_local_crops, B)),
            "reg_pre_head": l_reg.unflatten(0, (n_local_crops, B)),
            "patch_pre_head": l_patch.unflatten(0, (n_local_crops, B)),
            "cls_after_head": l_cls_after_head.unflatten(0, (n_local_crops, B)),
        }
        return student_global, student_local

    def teacher_state_dict(self) -> dict[str, Any]:
        return {k: v.state_dict() for k, v in self.teacher.items()}

    def student_state_dict(self) -> dict[str, Any]:
        return {k: v.state_dict() for k, v in self.student.items()}

def test():
    torch.cuda.set_device(2)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    cfg = load_config('config_eval.yaml')

    model = DualImageDINOv3(cfg=cfg, device=device).to(device)
    model.eval()
    print(next(model.backbone.parameters()).device)
    
    img = torch.randn(2, 3, 518, 518)
    img_aug = torch.randn(2, 3, 518, 518)

    feat1, feat2 = model(img, img_aug)
    
    print("保持进程，按 Ctrl+C 退出...")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("退出测试，释放显存")
        
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
