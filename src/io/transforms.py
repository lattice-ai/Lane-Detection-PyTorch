# -*- coding: utf-8 -*-
"""Utility Classes for various transforms"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image


class Compose:
    """
    Custom Compose Object for Custom Transforms
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, mask: Image.Image, bbx=None):
        if bbx is None:
            for transform in self.transforms:
                img, mask = transform(img, mask)
            return img, mask
        for transform in self.transforms:
            img, mask, bbx = transform(img, mask, bbx)
        return img, mask, bbx


class FreeScaleMask:
    """
    Scales the Mask based on the given size
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, mask: Image.Image) -> Image.Image:
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class RandomRotate:
    """
    Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle: int) -> None:
        self.angle = angle

    def __call__(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        assert label is None or image.size == label.size

        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label


class MaskToTensor:
    """
    Converts Mask to a Tensor
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class RandomLROffsetLABEL:
    """
    Randomly Offset from Left to Right (based on width)
    """

    def __init__(self, max_offset: int) -> None:
        self.max_offset = max_offset

    def __call__(
        self, img: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        offset = np.random.randint(-self.max_offset, self.max_offset)
        width, _ = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0 : width - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0 : width - real_offset, :] = img[:, real_offset:, :]
            img[:, width - real_offset :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0 : width - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0 : width - offset] = label[:, offset:]
            label[:, width - offset :] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomUDoffsetLABEL:
    """
    Randomly Offset from Up to Down (based on height)
    """

    def __init__(self, max_offset: int) -> None:
        self.max_offset = max_offset

    def __call__(
        self, img: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        offset = np.random.randint(-self.max_offset, self.max_offset)
        _, height = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0 : height - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0 : height - real_offset, :, :] = img[real_offset:, :, :]
            img[height - real_offset :, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0 : height - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0 : height - offset, :] = label[offset:, :]
            label[height - offset :, :] = 0
        return Image.fromarray(img), Image.fromarray(label)
