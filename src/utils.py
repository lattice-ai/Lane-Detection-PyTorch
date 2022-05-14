# -*- coding: utf-8 -*-
"""Utility Constants and Functions"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Union

import numpy as np
import torch

tusimple_row_anchor = [
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    124,
    128,
    132,
    136,
    140,
    144,
    148,
    152,
    156,
    160,
    164,
    168,
    172,
    176,
    180,
    184,
    188,
    192,
    196,
    200,
    204,
    208,
    212,
    216,
    220,
    224,
    228,
    232,
    236,
    240,
    244,
    248,
    252,
    256,
    260,
    264,
    268,
    272,
    276,
    280,
    284,
]

# pylint: disable=R1705
def str2bool(value: Any) -> Union[bool, Any]:
    """Utility Function to convert the given parameter to boolean

    :param value: Either a boolean or string
    :type value: Any
    :raises argparse.ArgumentTypeError: Raises the TypeError
    :return: A bool if the given input indicates a boolean intention
    :rtype: Union[bool, Any]
    """
    if isinstance(value, bool):
        return value
    elif value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def find_start_pos(row_sample: list, start_line: int) -> int:
    """
    Utility Function to find the starting position

    :param row_sample: A Row list
    :type row_sample: List
    :param start_line: Assumed Starting Position
    :type start_line: int
    :return: Starting Position
    :rtype: int
    """
    left, right = 0, len(row_sample) - 1
    while True:
        mid = int((left + right) / 2)
        if right - left == 1:
            return right
        if row_sample[mid] < start_line:
            left = mid
        if row_sample[mid] > start_line:
            right = mid
        if row_sample[mid] == start_line:
            return mid


def seed_everything(seed: int = 42) -> None:
    """
    Courtesy of https://twitter.com/kastnerkyle/status/1473361479143460872?
    """
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    random.seed(seed)
    torch_seed = random.randint(1, 1000000)
    torch_cuda_seed = random.randint(1, 1000000)
    numpy_seed = random.randint(1, 1000000)
    os_python_seed = random.randint(1, 1000000)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_cuda_seed)
    np.random.seed(numpy_seed)
    os.environ["PYTHONHASHSEED"] = str(os_python_seed)
