# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        self._resize_global_pre = resize_global
        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

        self.global_transfo1_light = v2.Compose([self._resize_global_pre, self.normalize])
        self.global_transfo2_light = v2.Compose([self._resize_global_pre, self.normalize])
        self.local_transfo_light = v2.Compose([self.normalize])

    def __call__(self, image, *, skip_color_aug: bool = False):
        bases = self._sample_bases(image)
        return self._finalize_from_bases(bases, skip_color_aug=skip_color_aug)

    def generate(self, image, *, skip_color_aug: bool = False):
        return self.__call__(image, skip_color_aug=skip_color_aug)

    def rebuild_sample(self, sample: dict, *, skip_color_aug: bool) -> dict:
        bases = {
            "global_bases": [self._clone_base(b) for b in sample.get("_global_bases", [])],
        }
        if self.local_crops_subset_of_global_crops:
            bases["local_subset_sources"] = sample.get("_local_subset_sources")
        else:
            bases["local_bases"] = [self._clone_base(b) for b in sample.get("_local_bases", [])]

        offsets = sample.get("_stored_offsets")
        rebuilt = self._finalize_from_bases(bases, skip_color_aug=skip_color_aug, preset_offsets=offsets)
        sample.update(rebuilt)
        return sample

    def _sample_bases(self, image):
        im1_base = self.geometric_augmentation_global(image)
        im2_base = self.geometric_augmentation_global(image)
        bases = {
            "global_bases": [self._clone_base(im1_base), self._clone_base(im2_base)],
        }
        if self.local_crops_subset_of_global_crops:
            sources = [0] * (self.local_crops_number // 2) + [1] * (self.local_crops_number // 2)
            bases["local_subset_sources"] = sources
        else:
            local_bases = [self.geometric_augmentation_local(image) for _ in range(self.local_crops_number)]
            bases["local_bases"] = [self._clone_base(lb) for lb in local_bases]
        return bases

    def _finalize_from_bases(self, bases, *, skip_color_aug: bool, preset_offsets=None):
        output = {"weak_flag": True}
        global_bases = bases["global_bases"]

        transfo1 = self.global_transfo1_light if skip_color_aug else self.global_transfo1
        transfo2 = self.global_transfo2_light if skip_color_aug else self.global_transfo2

        im1_base = global_bases[0]
        im2_base = global_bases[1]

        global_crop_1_transf = transfo1(self._clone_base(im1_base))
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        global_crop_2_transf = transfo2(self._clone_base(im2_base))
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        if self.teacher_no_color_jitter or skip_color_aug:
            output["global_crops_teacher"] = [
                self.normalize(self._clone_base(im1_base)),
                self.normalize(self._clone_base(im2_base)),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions or skip_color_aug:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(self._clone_base(im1_base)))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(self._clone_base(im2_base)))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        if self.local_crops_subset_of_global_crops:
            local_sources = bases.get("local_subset_sources") or []
            if not local_sources:
                local_sources = [0] * (self.local_crops_number // 2) + [1] * (self.local_crops_number // 2)
            offsets = preset_offsets if preset_offsets is not None else []
            local_crops = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for idx, src in enumerate(local_sources):
                if preset_offsets is None:
                    rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                    offsets.append((rx, ry))
                else:
                    rx, ry = offsets[idx]
                crop = output["global_crops"][src][:, rx : rx + ls, ry : ry + ls]
                local_crops.append(crop)
            output["local_crops"] = local_crops
            output["offsets"] = offsets
            output["_local_subset_sources"] = local_sources
            output["_stored_offsets"] = offsets
            output["_local_bases"] = []
        else:
            local_transfo = self.local_transfo_light if skip_color_aug else self.local_transfo
            local_bases = bases.get("local_bases", [])
            stored_local_bases = []
            local_crops = []
            for base in local_bases:
                base_clone = self._clone_base(base)
                stored_local_bases.append(self._clone_base(base))
                local_crops.append(local_transfo(base_clone))
            output["local_crops"] = local_crops
            output["offsets"] = ()
            output["_local_bases"] = stored_local_bases
            output["_local_subset_sources"] = None
            output["_stored_offsets"] = ()

        output["_global_bases"] = [self._clone_base(b) for b in global_bases]
        output["_skip_color_aug"] = skip_color_aug
        return output

    @staticmethod
    def _clone_base(base):
        if hasattr(base, "copy"):
            return base.copy()
        if torch.is_tensor(base):
            return base.clone()
        return deepcopy(base)
