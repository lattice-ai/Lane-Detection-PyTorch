"""Test Ultra Fast Structure-aware Deep Lane Detection"""
from __future__ import annotations

import pytest
import torch

from src.models.structure_aware_model import ConvBlock


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "kernel_size"), ((128, 256, 1), (128, 64, 1))
)
def test_convblock_output_shape(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
):
    """Test to check output shape from convolutional blocks"""
    block = ConvBlock(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
    )

    demo_input = torch.rand(512, 128, 4, 4)

    output = block(demo_input)

    assert list(output.size()) == [512, out_channels, 4, 4]
