# -*- coding: utf-8 -*-
"""Basic Metrics and related utilities"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch


def converter(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convertes the given numpy array or torch Tensor to a flattened numpy array

    :param data: A numpy ndarray or torch Tensor
    :type data: Union[torch.Tensor, np.ndarray]
    :return: Flattened numpy array
    :rtype: numpy ndarray
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()  # type: ignore


def fast_hist(
    label_pred: np.ndarray, label_true: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Creates a square matrix to store the overlap calculated using bincount

    :param label_pred: Predicted Labels
    :type label_pred: np.ndarray
    :param label_true: Ground Truth Labels
    :type label_true: np.ndarray
    :param num_classes: Number of Classes
    :type num_classes: int
    :return: A Square Matrix with overlap information
    :rtype: np.ndarray
    """
    # Get Overlaps using bincount
    hist = np.bincount(
        num_classes * label_true.astype(int) + label_pred, minlength=num_classes**2
    )
    # Convert to Square Matrix
    hist = hist.reshape(num_classes, num_classes)
    return hist


class IoU:
    """
    Barebones Implement of the Intersection over Union(IoU) Metric
    also known as the Jaccard Index
    """

    def __init__(self, class_num: int) -> None:
        self.class_num = class_num
        self.hist: np.ndarray = np.zeros((self.class_num, self.class_num))

    def update(self, predict: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the Histogram Matrix containing the Overlap information

        :param predict: Predicted Values
        :type predict: torch.Tensor
        :param target: True Values
        :type target: torch.Tensor
        """
        # Convert to flattened numpy arrays
        predict, target = (converter(predict), converter(target))  # type: ignore

        self.hist += fast_hist(predict, target, self.class_num)  # type: ignore

    def reset(self) -> None:
        """Resets the Matrix to Zero"""
        self.hist = np.zeros((self.class_num, self.class_num))

    def get_miou(self) -> np.float64:
        """Returns the IoU Value

        :return: IoU Metric Value
        :rtype: np.float64
        """
        miou = np.diag(self.hist) / (
            np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) - np.diag(self.hist)
        )
        miou = np.nanmean(miou)
        return miou

    def get_acc(self) -> np.float64:
        """Returns the Accuracy

        :return: Accuracy Metric Value
        :rtype: np.float64
        """
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def get(self) -> np.float64:
        """Returns the IoU Value"""
        return self.get_miou()


class MultiLabelAcc:
    """
    Barebones Implementation of Multi-Label Accuracy
    """

    def __init__(self):
        self.count: int = 0
        self.correct: np.int64 = 0

    def reset(self):
        """Resets Values to Zero"""
        self.count = 0
        self.correct = 0

    def update(self, predict: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the values based on the provided input

        :param predict: Predicted Values
        :type predict: torch.Tensor
        :param target: True Values
        :type target: torch.Tensor
        """
        # Convert to flattened numpy arrays
        predict, target = converter(predict), converter(target)  # type: ignore
        # Update Values
        self.count += len(predict)
        self.correct += np.sum(predict == target)

    def get_acc(self) -> np.float64:
        """Returns the Accuracy

        :return: Accuracy Metric Value
        :rtype: np.float64
        """
        return self.correct * 1.0 / self.count

    def get(self) -> np.float64:
        """Returns the Accuracy"""
        return self.get_acc()


class AccTopk:
    """
    Barebones Implementation of Top-k Accuracy
    """

    def __init__(self, background_classes: int, k: int) -> None:
        self.background_classes = background_classes
        self.k: int = k
        self.count: int = 0
        self.top5_correct: np.int64 = 0  # type: ignore

    def reset(self) -> None:
        """Resets Values to Zero"""
        self.count = 0
        self.top5_correct = 0  # type: ignore

    def update(self, predict: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the values based on the provided input

        :param predict: Predicted Values
        :type predict: torch.Tensor
        :param target: True Values
        :type target: torch.Tensor
        """
        # Convert to flattened numpy arrays
        predict, target = converter(predict), converter(target)  # type: ignore
        self.count += len(predict)

        # Calculate Top-k Accuracy
        background_idx = (predict == self.background_classes) + (
            target == self.background_classes
        )
        not_background_idx = np.logical_not(background_idx)
        self.top5_correct += np.sum(predict[background_idx] == target[background_idx])
        self.top5_correct += np.sum(
            np.absolute(
                predict[not_background_idx] - target[not_background_idx]  # type: ignore # pylint: disable=C0321
            )
            < self.k
        )

    def get(self) -> np.float64:
        """Returns the Top-k Accuracy

        :return: Top-k Accuracy Metric Value
        :rtype: np.float64
        """
        return self.top5_correct * 1.0 / self.count


def update_metrics(metric_dict: dict, pair_data: dict) -> None:
    """
    Updates the Value of each Metric in the provided
    metric_dict using the provided pair_data.

    :param metric_dict: A dictionary containing the various metrics to be used
    :type metric_dict: Dict
    :param pair_data: A dictionary containing data in a pairwise manner
    :type pair_data: Dict
    """
    for i in range(len(metric_dict["name"])):
        metric_op: Any = metric_dict["op"][i]
        data_src: tuple = metric_dict["data_src"][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict: dict) -> None:
    """reset_metrics Resets the Values for each metric in the provided metric_dict

    :param metric_dict: A dictionary containing the various metrics to be used
    :type metric_dict: Dict
    """
    for metric in metric_dict["op"]:
        metric.reset()
