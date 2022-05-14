# -*- coding: utf-8 -*-
"""Basic Loss Functions and related utilities"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxFocalLoss(nn.Module):
    """
    A Basic Implementation of the Softmax Focal Loss
    """

    def __init__(self, gamma: int, ignore_lb: int = 255) -> None:
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the Focal Loss

        Focal Loss is an improved version of the Binary Cross Entropy Loss,
        which weighes the values differently based on a modulating factor

        :param logits: Predicted Values
        :type logits: torch.Tensor
        :param labels: True Values
        :type labels: torch.Tensor
        :return: Obtained Softmax Focal Loss
        :rtype: torch.Tensor
        """
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.0 - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss: torch.Tensor = self.nll(log_score, labels)
        return loss


class LaneSimilarityLoss(nn.Module):
    """
    This class implements the Lane Similarity Loss Function,
    based on the assumption that lanes are continuous, that is to say, the lane
    points in adjacent row anchors should be close to each other.
    """

    # pylint: disable=R0201
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates the Lane Similarity Loss

        Based on the formulation in "Ultra Fast Structure-aware Deep Lane Detection",
        the model outputs location of the lane as classification vectors. Thus,
        the continuous property of lanes is realized by calculating the L1 distance
        between adjacent row anchors.

        :param logits: Predicted Values
        :type logits: torch.Tensor
        :return: Similarity loss between adjacent row anchors
        :rtype: torch.Tensor
        """
        _, _, num_row_anchors, _ = logits.shape
        loss_all = []
        for i in range(0, num_row_anchors - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all, dim=0)
        smoothened_loss = torch.nn.functional.smooth_l1_loss(
            loss, torch.zeros_like(loss)
        )
        return smoothened_loss


class LaneShapeLoss(nn.Module):
    """Implementation of Shaped Based Loss for Lane Detection

    Generally, most of the lanes are straight. Even the curved lanes, because of
    the perspective effect seem straight. Based on this, the authors of
    "Ultra Fast Structure-aware Deep Lane Detection" propose a new loss function that
    uses Second-Order Difference Equations to constraint the shape of the lane.
    """

    def __init__(self) -> None:
        super().__init__()
        # We set reduction = "sum" to avoid division by tensor shapes later
        self.l1_loss = torch.nn.L1Loss(reduction="sum")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculates the Shape Based Lane Loss

        Based on the formulation in "Ultra Fast Structure-aware Deep Lane Detection",
        we use a second order difference equation to calculate the difference between
        pairwise predicted locations for the next two row anchors

        :param inputs: Predicted Values
        :type inputs: torch.Tensor
        :return: Shape Based Loss
        :rtype: torch.Tensor
        """
        loss = 0
        differences = []
        _, dim, num_row_anchors, _ = inputs.shape

        # Because argmax is not differentiable we take the Softmax
        # of the output probabilites as a differentiable approximation
        # of location. For more details see eq 5 of https://arxiv.org/pdf/2004.11757.pdf
        probs = torch.nn.functional.softmax(inputs[:, : dim - 1, :, :], dim=1)

        embedding = (
            torch.Tensor(np.arange(dim - 1)).float().to(probs.device).view(1, -1, 1, 1)
        )

        # Locations of lanes
        locations = torch.sum(probs * embedding, dim=1)

        # Store the Pairwise differenc in locations in a list
        # Since we're calculating pair-wise differences, we loop
        # over half the number of row anchors
        for i in range(0, num_row_anchors // 2):
            differences.append(locations[:, i, :] - locations[:, i + 1, :])

        # Calculate L1 loss between each pair and take
        for i in range(len(differences) - 1):
            loss += self.l1_loss(differences[i], differences[i + 1])
        return loss  # type: ignore
