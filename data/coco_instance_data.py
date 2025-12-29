from __future__ import annotations

"""COCO instance dataset utilities for Mask2Former training.

This module adapts the data/evaluation helpers from
``/home/yangyan/hong/Diveseg/mask2former`` into a lightweight PyTorch-friendly
implementation that can be imported by ``MAS/train_m2f.py`` later on.
It covers three tasks:

1.  Parsing COCO-style JSON annotations (images + segmentations) and building
    PyTorch ``Dataset``/``DataLoader`` instances.
2.  Exporting Mask2Former predictions to standard COCO evaluation format,
    computing mAP / AP50 / AP75 via ``pycocotools``.
3.  Visualising per-image predictions as PNG overlays, similar to
    ``InstanceSegEvaluator.save_visualizations`` in DiveSeg.

The dataset keeps both semantic masks (class-per-pixel) and the per-instance
maps required by Mask2Former.  The semantic map allows the current
``train_m2f`` implementation to keep running, while the per-instance tensors
(``instance_masks`` + ``instance_class_labels``) are ready for the follow-up
multi-class instance heads.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dinov3.eval.segmentation.transforms import (
    make_segmentation_eval_transforms,
    make_segmentation_train_transforms,
)

from .data import _build_dataloader


class CocoInstanceDataset(Dataset):
    """Load images + instance segmentations from a COCO JSON file.

    Args:
        image_root: Directory containing all RGB images.
        annotation_file: Path to the COCO ``instances_*.json`` file.
        transforms: Callable that accepts ``(PIL.Image, np.ndarray)`` and
            returns the transformed pair. Defaults to the segmentation
            transforms already used inside MAS.
        ignore_index: Label value reserved for unlabeled pixels.
    """

    def __init__(
        self,
        image_root: Path | str,
        annotation_file: Path | str,
        transforms: Optional[Callable] = None,
        *,
        ignore_index: int = 255,
    ) -> None:
        self.image_root = Path(image_root)
        self.coco = COCO(str(annotation_file))
        self.img_ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms
        self.ignore_index = ignore_index

        categories = self.coco.loadCats(self.coco.getCatIds())
        # map original COCO category ids to contiguous ids used by the head
        self.cat_id_to_contiguous: Dict[int, int] = {}
        self.contiguous_id_to_cat: List[int] = []
        self.class_names: List[str] = []
        for idx, cat in enumerate(sorted(categories, key=lambda c: c["id"])):
            self.cat_id_to_contiguous[cat["id"]] = idx
            self.contiguous_id_to_cat.append(cat["id"])
            self.class_names.append(cat["name"])

        # cache for quick metadata lookup
        self._images_by_id: Dict[int, Dict[str, Any]] = {
            img_id: self.coco.imgs[img_id] for img_id in self.img_ids
        }

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.img_ids)

    def image_path(self, image_id: int) -> Path:
        info = self._images_by_id[image_id]
        return self.image_root / info["file_name"]

    def _build_label_tensor(
        self,
        anns: Sequence[Dict[str, Any]],
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, List[int]]:
        semantic = np.full((height, width), self.ignore_index, dtype=np.int64)
        mask_channels: List[np.ndarray] = []
        class_ids: List[int] = []

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            category_id = self.cat_id_to_contiguous.get(ann["category_id"], None)
            if category_id is None:
                continue
            mask = self.coco.annToMask(ann)
            if mask is None:
                continue
            mask = mask.astype(bool)
            semantic[mask] = category_id

            channel = np.full((height, width), self.ignore_index, dtype=np.int64)
            channel[mask] = self.ignore_index + len(mask_channels) + 1
            mask_channels.append(channel)
            class_ids.append(category_id)

        stacked = semantic[..., None]
        if mask_channels:
            stacked = np.concatenate([stacked, np.stack(mask_channels, axis=-1)], axis=-1)
        return stacked, class_ids

    def __getitem__(self, index: int):  # type: ignore[override]
        image_id = self.img_ids[index]
        img_info = self._images_by_id[image_id]
        file_name = img_info["file_name"]
        image = Image.open(self.image_root / file_name).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        label_np, mask_class_ids = self._build_label_tensor(anns, img_info["height"], img_info["width"])

        if self.transforms is not None:
            image, label_np = self.transforms(image, label_np)

        image_tensor = self._ensure_image_tensor(image)
        label_tensor = self._ensure_label_tensor(label_np)

        semantic_tensor = label_tensor[0].long()
        instance_layers = label_tensor[1:]

        instance_masks, class_labels = self._extract_instances(
            instance_layers,
            mask_class_ids,
            (semantic_tensor.shape[-2], semantic_tensor.shape[-1]),
        )

        extras = {
            "instance_masks": instance_masks,
            "instance_class_labels": class_labels,
            "image_id": image_id,
        }
        orig_size = torch.tensor((int(img_info["height"]), int(img_info["width"])), dtype=torch.long)
        return image_tensor, semantic_tensor, index, orig_size, file_name, extras

    @staticmethod
    def _ensure_image_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, list):
            # MAS 数据 pipeline 只使用第一张增强图
            image = image[0]
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.from_numpy(np.asarray(image))
        if tensor.ndim == 3 and tensor.shape[0] not in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()

    @staticmethod
    def _ensure_label_tensor(label: Any) -> torch.Tensor:
        if isinstance(label, torch.Tensor):
            tensor = label
        else:
            tensor = torch.as_tensor(label)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _extract_instances(
        self,
        instance_layers: torch.Tensor,
        class_ids: Sequence[int],
        spatial_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks: List[torch.Tensor] = []
        labels: List[int] = []

        for layer, cls in zip(instance_layers, class_ids):
            binary = (layer != self.ignore_index).float()
            if binary.max() <= 0:
                continue
            masks.append(binary)
            labels.append(cls)

        if masks:
            mask_tensor = torch.stack(masks, dim=0)
            class_tensor = torch.as_tensor(labels, dtype=torch.long)
        else:
            height, width = spatial_size
            mask_tensor = torch.zeros((0, height, width), dtype=torch.float32)
            class_tensor = torch.zeros((0,), dtype=torch.long)
        return mask_tensor, class_tensor


def coco_instance_collate_fn(batch):
    """自定义 collate：保持实例掩码列表的变长形状。"""

    images, semantic_masks, indices, orig_sizes, file_names, extras = zip(*batch)
    image_tensor = torch.stack(images, dim=0)
    semantic_tensor = torch.stack(semantic_masks, dim=0)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    orig_tensor = torch.stack(orig_sizes, dim=0)
    file_name_list = list(file_names)
    extras_list = list(extras)
    return image_tensor, semantic_tensor, index_tensor, orig_tensor, file_name_list, extras_list


def build_coco_instance_datasets_from_cfg(dataset_cfg: dict):
    """Create train/val ``CocoInstanceDataset`` from YAML config.

    Expected keys under ``dataset``:
        train_images, train_annotations, val_images, val_annotations,
        num_classes (optional), ignore_index (optional), transforms.*
    """

    transforms_cfg = dataset_cfg.get("transforms", {})
    mean = transforms_cfg.get("mean", (0.485, 0.456, 0.406))
    std = transforms_cfg.get("std", (0.229, 0.224, 0.225))
    train_tf_cfg = transforms_cfg.get("train", {})
    eval_tf_cfg = transforms_cfg.get("eval", {})

    train_tf = make_segmentation_train_transforms(
        img_size=train_tf_cfg.get("img_size", 512),
        random_img_size_ratio_range=train_tf_cfg.get("random_img_size_ratio_range", [0.5, 2.0]),
        crop_size=train_tf_cfg.get("crop_size", [512, 512]),
        flip_prob=train_tf_cfg.get("flip_prob", 0.5),
        mean=mean,
        std=std,
    )

    eval_img_size = eval_tf_cfg.get("img_size", 512)
    if isinstance(eval_img_size, int):
        eval_img_size = [eval_img_size, eval_img_size]
    eval_tf = make_segmentation_eval_transforms(
        img_size=eval_img_size,
        inference_mode=eval_tf_cfg.get("inference_mode", "slide"),
        use_tta=eval_tf_cfg.get("use_tta", False),
        tta_ratios=eval_tf_cfg.get("tta_ratios", [1.0]),
        mean=mean,
        std=std,
    )

    train_dataset = CocoInstanceDataset(
        dataset_cfg["train_images"],
        dataset_cfg["train_annotations"],
        transforms=train_tf,
        ignore_index=int(dataset_cfg.get("ignore_index", 255)),
    )
    val_dataset = CocoInstanceDataset(
        dataset_cfg["val_images"],
        dataset_cfg["val_annotations"],
        transforms=eval_tf,
        ignore_index=int(dataset_cfg.get("ignore_index", 255)),
    )
    return train_dataset, val_dataset


def build_coco_instance_dataloaders_from_cfg(
    dataset_cfg: dict,
    training_cfg: dict,
    device_type: str,
    *,
    distributed: bool = False,
):
    train_dataset, val_dataset = build_coco_instance_datasets_from_cfg(dataset_cfg)
    pin_memory = device_type == "cuda"
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 4),
        num_workers=training_cfg.get("num_workers", 8),
        shuffle=True,
        pin_memory=pin_memory,
        distributed=distributed,
        collate_fn=coco_instance_collate_fn,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=training_cfg.get("val_batch_size", 1),
        num_workers=max(1, training_cfg.get("num_workers", 8) // 2),
        shuffle=False,
        pin_memory=pin_memory,
        distributed=distributed,
        collate_fn=coco_instance_collate_fn,
    )
    return train_loader, val_loader


def coco_evaluate_instances(
    prediction_dicts: Sequence[Dict[str, Any]],
    annotation_file: str | Path,
    *,
    output_dir: Optional[Path | str] = None,
    iou_type: str = "segm",
) -> Dict[str, float]:
    """Run COCOeval on a list of prediction dictionaries.

    ``prediction_dicts`` should match the Detectron2 / Mask2Former format:
        {"image_id": int, "instances": [ {"segmentation": RLE, "score": float,
          "category_id": int, "bbox": [x, y, w, h] } ] }
    """

    coco_gt = COCO(str(annotation_file))
    coco_results: List[Dict[str, Any]] = list(itertools.chain(*[p["instances"] for p in prediction_dicts]))
    if not coco_results:
        raise ValueError("Empty predictions: cannot run COCOeval")
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    metrics = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
    }
    if output_dir:
        output_path = Path(output_dir) / "coco_instances_results.json"
        output_path.write_text(json.dumps(coco_results))
    return metrics


def visualize_instance_predictions(
    dataset: CocoInstanceDataset,
    prediction_dicts: Sequence[Dict[str, Any]],
    output_dir: Path | str,
    *,
    score_thresh: float = 0.5,
) -> None:
    """Save colourised PNG overlays for predicted masks.

    Colours are deterministically generated per class id, mirroring the
    behaviour of ``InstanceSegEvaluator.save_visualizations``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = _build_class_palette(len(dataset.class_names))

    for prediction in prediction_dicts:
        image_id = prediction["image_id"]
        image_path = dataset.image_path(image_id)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        overlay = image.copy()
        instances = prediction.get("instances", [])
        for inst in instances:
            if inst.get("score", 1.0) < score_thresh:
                continue
            segm = inst.get("segmentation")
            if segm is None:
                continue
            mask = mask_utils.decode(segm).astype(bool)
            if mask.shape[:2] != overlay.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = palette[inst.get("category_id", 0) % len(palette)]
            overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
        save_path = output_dir / f"vis_{image_id}.png"
        cv2.imwrite(str(save_path), overlay)


def _build_class_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    hsv = np.linspace(0, 179, max(1, num_classes), dtype=np.uint8)
    palette = []
    for h in hsv:
        color = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        palette.append(tuple(int(c) for c in color))
    return palette
