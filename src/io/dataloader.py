# -*- coding: utf-8 -*-
"""Utility Classes for getting dataloaders"""
from __future__ import annotations

import os
from typing import Iterable

import torch
from torchvision import transforms

import src.io.transforms as CustomTransforms
from src.io.dataset import TUSimpleDataset, TUSimpleTestDataset
from src.utils import tusimple_row_anchor


def tusimple_train_dataloader(
    batch_size: int, data_root: str, griding_num: int, use_aux: bool, num_lanes: int
) -> Iterable:
    """
    Get Training Dataloader for the TUSimple Dataset

    :param batch_size: Batch Size for the Dataloader
    :type batch_size: int
    :param data_root: Path to the root dir of the dataset
    :type data_root: str
    :param griding_num: Number of grid cells
    :type griding_num: int
    :param use_aux: Whether to use the auxiliary branch
    :type use_aux: bool
    :param num_lanes: Number of Lanes
    :type num_lanes: int
    :return: A Pytorch Dataloader for training on the TUSimple Dataset
    :rtype: Iterable
    """
    target_transform = transforms.Compose(
        [
            CustomTransforms.FreeScaleMask((288, 800)),
            CustomTransforms.MaskToTensor(),
        ]
    )
    segment_transform = transforms.Compose(
        [
            CustomTransforms.FreeScaleMask((36, 100)),
            CustomTransforms.MaskToTensor(),
        ]
    )
    img_transform = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    simu_transform = CustomTransforms.Compose(
        [
            CustomTransforms.RandomRotate(6),
            CustomTransforms.RandomUDoffsetLABEL(100),
            CustomTransforms.RandomLROffsetLABEL(200),
        ]
    )

    train_dataset = TUSimpleDataset(
        path=data_root,
        list_path=os.path.join(data_root, "train_gt.txt"),
        img_transform=img_transform,
        target_transform=target_transform,
        simu_transform=simu_transform,
        griding_num=griding_num,
        row_anchor=tusimple_row_anchor,
        segment_transform=segment_transform,
        use_aux=use_aux,
        num_lanes=num_lanes,
    )

    sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    return train_loader


def tusimple_test_dataloader(batch_size: int, data_root: str) -> Iterable:
    """
    Get Test Dataloader for the TUSimple Dataset

    :param batch_size: Batch Size for the Dataloader
    :type batch_size: int
    :param data_root: Path to the root dir of the dataset
    :type data_root: str
    :return: A Pytorch Dataloader for testing on the TUSimple Dataset
    :rtype: Iterable
    """
    img_transforms = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_dataset = TUSimpleTestDataset(
        path=data_root,
        list_path=os.path.join(data_root, "test.txt"),
        img_transform=img_transforms,
    )

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )
    return loader
