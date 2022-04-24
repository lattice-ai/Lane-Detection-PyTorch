# -*- coding: utf-8 -*-
"""
Implementation of the Structure Aware Model from
'Ultra Fast Structure-aware Deep Lane Detection' by Zequn Qin, Huanyu Wang, and Xi Li.

This is a modified version of the Original Code from
https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/model/model.py

"""
from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import Tuple

import numpy as np
import torch
from rich.progress import track

from src.models.backbones import ResNet
from src.nn.loss import LaneShapeLoss
from src.nn.loss import LaneSimilarityLoss
from src.nn.loss import SoftmaxFocalLoss
from src.nn.metrics import AccTopk
from src.nn.metrics import IoU
from src.nn.metrics import MultiLabelAcc
from src.nn.metrics import reset_metrics
from src.nn.metrics import update_metrics


class ConvBlock(torch.nn.Module):
    """
    Utility Class for a convolution, followed by
    Batch Normalization and ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through the block

        :param x: input Tensor
        :type x: torch.Tensor
        :return: Output from the block
        :rtype: torch.Tensor
        """
        feature = self.conv(inputs)
        feature = self.batch_norm(feature)
        feature = self.relu(feature)
        return feature


# pylint: disable=R0902
class StructureAwareModel(torch.nn.Module):
    """
    Model from the paper "Ultra Fast Structure-aware Deep Lane Detection"
    by Zequn Qin, Huanyu Wang, and Xi Li.
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone: str = "18",
        cls_dim: Tuple[int, int, int] = (37, 10, 4),
        use_aux: bool = False,
    ) -> None:
        """
        Instantiate the Model

        :param pretrained: Whether to use a PreTrained Backbone, defaults to True
        :type pretrained: bool, optional
        :param backbone: Which ResNet variant to use, defaults to "18"
        :type backbone: str, optional
        :param cls_dim: (number of grid cells, number of row anchors, number of lanes),
                        defaults to (37, 10, 4)
        :type cls_dim: Tuple[int, int, int], optional
        :param use_aux: Whether to use the Auxiliary Branch, defaults to False
        :type use_aux: bool, optional
        """
        super().__init__()

        self.classifier_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # input : nchw
        # output: (w+1) * sample_rows * 4
        # Model Backbone
        self.model = ResNet(variant=backbone, pretrained=pretrained)

        self.pool = (
            torch.nn.Conv2d(512, 8, 1)
            if backbone in ["34", "18"]
            else torch.nn.Conv2d(2048, 8, 1)
        )

        # Auxiliary Segmentation Head
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBlock(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBlock(128, 128, 3, padding=1),
                ConvBlock(128, 128, 3, padding=1),
                ConvBlock(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                ConvBlock(256, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBlock(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvBlock(128, 128, 3, padding=1),
                ConvBlock(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                ConvBlock(512, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBlock(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvBlock(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                ConvBlock(384, 256, 3, padding=2, dilation=2),
                ConvBlock(256, 128, 3, padding=2, dilation=2),
                ConvBlock(128, 128, 3, padding=2, dilation=2),
                ConvBlock(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )
            self._init_weights(
                [self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine]
            )

        # Group Classification Head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initializes the weights and biases of the various layers in the network

        :param module: The Layer whose weights or bias is to be initialized
        :type module: torch.nn.Module
        :raises NameError: Specifies the Unknown Module
        """

        if isinstance(module, list):
            for mini_m in module:
                self._init_weights(mini_m)
        else:
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(0.0, std=0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Module):
                for mini_m in module.children():
                    self._init_weights(mini_m)
            else:
                raise NameError(f"Unknown Module: {module}")

    def forward(self, inputs: torch.Tensor):
        """Computes a forward pass through the model

        If the Auxiliary Head is being used the output is the
        group classification features and the segmentation features
        and only the group classification features otherwise

        :param inputs: Input Data
        :type inputs: torch.Tensor
        :return: Output Representations from the forward pass
        :rtype: Union[torch.Tensor, Union[torch.Tensor, torch.Tensor]]
        """
        # Get Features from the Backbone
        intermediate_feature_1, intermediate_feature_2, features = self.model(inputs)

        # Auxiliary Segmentation Head
        if self.use_aux:
            intermediate_feature_1 = self.aux_header2(intermediate_feature_1)
            intermediate_feature_2 = self.aux_header3(intermediate_feature_2)
            intermediate_feature_2 = torch.nn.functional.interpolate(
                intermediate_feature_2, scale_factor=2, mode="bilinear"
            )
            auxiliary_features = self.aux_header4(features)
            auxiliary_features = torch.nn.functional.interpolate(
                auxiliary_features, scale_factor=4, mode="bilinear"
            )
            aux_seg = torch.cat(
                [intermediate_feature_1, intermediate_feature_2, auxiliary_features],
                dim=1,
            )
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        # Reshape Tensor to feed to the Group Classification Head
        features = self.pool(features).view(-1, 1800)

        # Group Classification Head
        group_cls: torch.Tensor = self.classifier(features).view(
            -1, *self.classifier_dim
        )

        # Return the output from both the heads during training
        if self.use_aux:
            return group_cls, aux_seg  # type: ignore

        return group_cls


def structure_aware_model_inference(model, batch: tuple, use_aux: bool) -> dict:
    """
    Custom Inference Function for the Structure Aware Model

    This function returns a custom dictionary used for calculating metrics based on the
    condition if the auxiliary head will be used

    :param model: An Instance of the StructureAwareModel
    :type model: torch.nn.Module
    :param batch: batch of data from the TUSimple dataloader
    :type batch: Tuple
    :param use_aux: Whether to use the Auxiliary Head or not
    :type use_aux: bool
    :return: dictionary object based on the use_aux parameter
    :rtype: Dict
    """
    # If Segmentation Head is being used
    if use_aux:
        img, classification_label, segmentation_label = batch
        # Convert to torch Tensors
        img, classification_label, segmentation_label = (
            img.cuda(),
            classification_label.long().cuda(),
            segmentation_label.long().cuda(),
        )

        classification_output, segmentation_output = model(img)

        return {
            "cls_out": classification_output,
            "cls_label": classification_label,
            "seg_out": segmentation_output,
            "seg_label": segmentation_label,
        }
    # If only the Global Classification Head is being used
    else:
        img, classification_label = batch
        # Convert to torch Tensors
        img, classification_label = (img.cuda(), classification_label.long().cuda())

        classification_output = model(img)

        return {"cls_out": classification_output}


def structure_aware_loss_dict(args: Any) -> dict:
    """
    Creates a custom dictionary for the various loss functions used in
    the training phase

    :param args: NameSpace
    :type args: Any
    :return: custom dictionary containing various loss functions
    :rtype: Dict
    """

    if args.use_aux:
        loss_dict = {
            "name": ["cls_loss", "relation_loss", "aux_loss", "relation_dis"],
            "op": [
                SoftmaxFocalLoss(2),
                LaneSimilarityLoss(),
                torch.nn.CrossEntropyLoss(),
                LaneShapeLoss(),
            ],
            "weight": [1.0, args.sim_loss_w, 1.0, args.shp_loss_w],
            "data_src": [
                ("cls_out", "cls_label"),
                ("cls_out",),
                ("seg_out", "seg_label"),
                ("cls_out",),
            ],
        }
    else:
        loss_dict = {
            "name": ["cls_loss", "relation_loss", "relation_dis"],
            "op": [SoftmaxFocalLoss(2), LaneSimilarityLoss(), LaneShapeLoss()],
            "weight": [1.0, args.sim_loss_w, args.shp_loss_w],
            "data_src": [("cls_out", "cls_label"), ("cls_out",), ("cls_out",)],
        }

    return loss_dict


def structure_aware_metric_dict(args: Any) -> dict:
    """
    Creates a custom dictionary for the various metrics used in the training phase

    :param args: NameSpace
    :type args: Any
    :return: custom dictionary containing various loss metrics
    :rtype: Dict
    """

    if args.use_aux:
        metric_dict = {
            "name": ["top1", "top2", "top3", "iou"],
            "op": [
                MultiLabelAcc(),
                AccTopk(args.griding_num, 2),
                AccTopk(args.griding_num, 3),
                IoU(args.num_lanes + 1),
            ],
            "data_src": [
                ("cls_out", "cls_label"),
                ("cls_out", "cls_label"),
                ("cls_out", "cls_label"),
                ("seg_out", "seg_label"),
            ],
        }
    else:
        metric_dict = {
            "name": ["top1", "top2", "top3"],
            "op": [
                MultiLabelAcc(),
                AccTopk(args.griding_num, 2),
                AccTopk(args.griding_num, 3),
            ],
            "data_src": [
                ("cls_out", "cls_label"),
                ("cls_out", "cls_label"),
                ("cls_out", "cls_label"),
            ],
        }

    return metric_dict


def structure_aware_train_fn(
    model,
    data_loader: Iterable,
    loss_dict: dict,
    optimizer,
    scheduler,
    metric_dict: dict,
    use_aux: bool,
) -> dict:
    """
    Trains the model for one epoch

    :param model: An instance of the StructureAwareModel
    :type model: torch.nn.Module
    :param data_loader: A Pytorch Dataloader to be used for training
    :type data_loader: Iterable
    :param loss_dict: dictionary object for the various loss functions to be used
    :type loss_dict: Dict
    :param optimizer: The optimizer to be used for training
    :param scheduler: The scheduler to be used for training
    :param metric_dict: dictionary object for the various metrics to be used
    :type metric_dict: Dict
    :param use_aux: Whether to use the Auxiliary Head or not
    :type use_aux: bool
    :return: An updated metric dictionary
    :rtype: Dict
    """
    model.train()
    for _, batch in track(sequence=enumerate(data_loader), total=len(data_loader)):  # type: ignore # pylint: disable=C0321
        reset_metrics(metric_dict)
        results = structure_aware_model_inference(model, batch, use_aux)

        loss = calculate_loss(loss_dict, results)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        results = resolve_val_data(results, use_aux)

    update_metrics(metric_dict, results)

    return metric_dict


def calculate_loss(loss_dict: dict, results: dict) -> torch.Tensor:
    """
    Calculates the various losses

    :param loss_dict: custom dictionary for the various losses to be used
    :type loss_dict: Dict
    :param results: custom dictionary containing outputs from the model
    :type results: Dict
    :return: Cumulative Loss
    :rtype: torch.Tensor
    """
    loss = 0

    for i in range(len(loss_dict["name"])):

        data_src = loss_dict["data_src"][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict["op"][i](*datas)

        # weigh the loss by the provided weight
        loss += loss_cur * loss_dict["weight"][i]
    return loss  # type: ignore


def resolve_val_data(results: dict, use_aux: bool) -> dict:
    """
    Returns the Argmax of the model outputs(s)

    :param results: custom dictionary containing outputs from the model
    :type results: Dict
    :param use_aux: Whether to use the Auxiliary Head or not
    :type use_aux: bool
    :return: Validated Results
    :rtype: Dict
    """
    results["cls_out"] = torch.argmax(results["cls_out"], dim=1)
    if use_aux:
        results["seg_out"] = torch.argmax(results["seg_out"], dim=1)
    return results
