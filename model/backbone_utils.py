from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from MAS.model.Dino_LoRA import inject_lora_qv_into_vit
from dinov3.models import build_model_for_eval
from utils import dict_to_namespace, prepare_dinov3_cfg
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")


def _extract_backbone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if "student" in state and isinstance(state["student"], dict) and "backbone" in state["student"]:
        return state["student"]["backbone"]
    if "backbone" in state and isinstance(state["backbone"], dict):
        return state["backbone"]
    return state


def build_backbone(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    processed = prepare_dinov3_cfg(cfg)
    ns_cfg = dict_to_namespace(processed)
    pretrained = processed.get("pretrained_weights")
    backbone = build_model_for_eval(config=ns_cfg, pretrained_weights=pretrained)
    backbone.to(device)
    backbone.eval()

    lora_layers = cfg.get("train", {}).get("lora_layers")
    if lora_layers not in (None, 0, []):
        inject_lora_qv_into_vit(
            backbone,
            layers=lora_layers,
            rank=int(cfg.get("train", {}).get("lora_rank", 8)),
            alpha=float(cfg.get("train", {}).get("lora_alpha", 16.0)),
            dropout=float(cfg.get("train", {}).get("lora_dropout", 0.0)),
        )

    finetuned_path = cfg.get("finetuned_weights")
    if finetuned_path:
        state = torch.load(Path(finetuned_path).expanduser(), map_location="cpu")
        backbone_state = _extract_backbone_state(state)
        backbone.load_state_dict(backbone_state, strict=False)
    backbone.requires_grad_(False)
    return backbone


__all__ = ["build_backbone"]
