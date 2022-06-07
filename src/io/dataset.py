# -*- coding: utf-8 -*-
"""Utility Classes for creating PyTorch Dataset Objects"""
from __future__ import annotations

import os
from typing import Any, Callable, Tuple

import numpy as np
import torch
from PIL import Image

from src.utils import find_start_pos, tusimple_row_anchor


class TUSimpleTestDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for testing on the TUSimple Dataset
    """

    def __init__(self, path: str, list_path: str, img_transform: Any = None) -> None:
        super().__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, encoding="utf-8") as list_file:
            self.list = list_file.readlines()

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        name: str = self.list[index].split()[0]
        img_path: str = os.path.join(self.path, name)
        img: Image.Image = Image.open(img_path)

        # Apply Image Transforms if provided
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self) -> int:
        return len(self.list)


# pylint: disable=R0902
class TUSimpleDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Training on the TUSimple Dataset
    """

    # pylint: disable=R0902
    def __init__(
        self,
        path: str,
        list_path: str,
        img_transform: Any = None,
        target_transform: Any = None,
        simu_transform: Any = None,
        griding_num: int = 50,
        load_name: bool = False,
        row_anchor: list = tusimple_row_anchor,
        use_aux: bool = False,
        segment_transform: Any = None,
        num_lanes: int = 4,
    ) -> None:
        super().__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, encoding="utf-8") as list_file:
            self.list = list_file.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()  # type: ignore

    def __getitem__(self, index: int):
        list_val = self.list[index]
        l_info = list_val.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == "/":
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors

        width, _ = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, width)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        # Apply Image Transforms if provided
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self) -> int:
        return len(self.list)

    def _grid_pts(self, pts: np.ndarray, num_cols: int, width: int) -> np.ndarray:
        # pts : numlane,n,2
        num_lane, location, _ = pts.shape
        col_sample: np.ndarray = np.linspace(0, width - 1, num_cols)

        to_pts = np.zeros((location, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [
                    int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols
                    for pt in pti
                ]
            )

        points_array = to_pts.astype(int)
        return points_array

    # pylint: disable=R0914
    def _get_index(self, label):
        width, height = label.size

        # The Row Anchors provided are wrt to an image height of 288
        # Incase the height isn't 288, we scale the anchors
        if height != 288:
            scale_fn: Callable[[int], int] = lambda x: int((x * 1.0 / 288) * height)
            scaled_row_anchors: list = list(map(scale_fn, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(scaled_row_anchors), 2))
        for i, row_anchor in enumerate(scaled_row_anchors):
            label_r = np.asarray(label)[int(round(row_anchor))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = row_anchor
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = row_anchor
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is
                # already the last y-coordinate of all rows
                # this means this lane has reached the bottom
                # boundary of the image so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2 :, :]
            fitted_polynomials_coeffs = np.polyfit(
                valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1
            )
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(fitted_polynomials_coeffs, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > width - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted

        return all_idx_cp
