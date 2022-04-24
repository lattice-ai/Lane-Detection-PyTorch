# -*- coding: utf-8 -*-
"""Utility Classes for various backbones"""
from __future__ import annotations

from typing import Tuple

import torch.nn.modules
import torchvision
from torch import nn


class ResNet(nn.Module):
    """
    Utility Class to be used to provide backbones
    for various architectures.
    """

    def __init__(self, variant: str, pretrained: bool = False) -> None:
        super().__init__()
        if variant == "18":
            self.model = torchvision.models.resnet18(pretrained=pretrained)
        elif variant == "34":
            self.model = torchvision.models.resnet34(pretrained=pretrained)
        else:
            raise NameError(
                f"""Unknown variant {variant}. Only resnet18 and resnet34
                are allowed"""
            )

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward Function for ResNet Backbones

        :param inputs: Input Tensor
        :type inputs: torch.Tensor
        :return: Output Representations from the last 3 blocks
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        feature = self.model.conv1(inputs)
        feature = self.model.bn1(feature)
        feature = self.model.relu(feature)
        feature = self.model.maxpool(feature)
        feature = self.model.layer1(feature)
        intermediate_feature_1 = self.model.layer2(feature)
        intermediate_feature_2 = self.model.layer3(intermediate_feature_1)
        intermediate_feature_3 = self.model.layer4(intermediate_feature_2)
        return intermediate_feature_1, intermediate_feature_2, intermediate_feature_3
