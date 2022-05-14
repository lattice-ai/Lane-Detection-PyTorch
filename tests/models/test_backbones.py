"""Test Custom Backbones"""
from __future__ import annotations

import pytest
import torch

from src.models.backbones import ResNet


@pytest.mark.parametrize(("variant"), (("18"), ("34")))
def test_backbone_output_shape(variant: str):
    """Test to check output shape from backbones"""
    model = ResNet(variant=variant, pretrained=False)

    demo_input = torch.rand(32, 3, 128, 128)

    output = model(demo_input)

    assert list(output[0].size()) == [32, 128, 16, 16]
    assert list(output[1].size()) == [32, 256, 8, 8]
    assert list(output[2].size()) == [32, 512, 4, 4]


@pytest.mark.parametrize(
    "variant",
    (
        "50",
        pytest.param([], id="empty_list"),
        pytest.param((), id="empty_tuple"),
    ),
)
def test_backbone_unknown_variant(variant: str):
    """
    Test that backbone fails to build if unknown variant is provided
    """

    with pytest.raises(NameError):
        ResNet(variant=variant, pretrained=False)
