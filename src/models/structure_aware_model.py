# -*- coding: utf-8 -*-
"""
Implementation of the Structure Aware Model from
'Ultra Fast Structure-aware Deep Lane Detection' by Zequn Qin, Huanyu Wang, and Xi Li.

This is a modified version of the Original Code from
https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/model/model.py

"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
from rich.progress import track

from src.io.dataloader import tusimple_train_dataloader
from src.models.backbones import ResNet
from src.nn.loss import LaneShapeLoss, LaneSimilarityLoss, SoftmaxFocalLoss
from src.nn.metrics import AccTopk, IoU, MultiLabelAcc, reset_metrics, update_metrics

NUM_ROW_ANCHORS = 56


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


class StructureAwareTrainer:
    """Custom Trainer Class for the StructureAwareModel

    Utility Class to train a model as per:
    'Ultra Fast Structure-aware Deep Lane Detection'
    by Zequn Qin, Huanyu Wang, and Xi Li.
    """

    def __init__(
        self,
        epochs: int,
        data_root: str,
        use_pretrained: bool,
        backbone: str,
        griding_num: int,
        num_lanes: int,
        use_aux: bool,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        sim_loss_w: float = 1.0,
        shp_loss_w: float = 0.0,
    ) -> None:
        self.epochs = epochs
        self.data_root = data_root
        self.batch_size = batch_size
        self.griding_num = griding_num
        self.num_lanes = num_lanes
        self.use_aux = use_aux
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.sim_loss_w = sim_loss_w
        self.shp_loss_w = shp_loss_w

        # Get Metric Dict
        self.metric_dict: Dict = {}
        self.get_metric_dict()

        # Get Loss Dict
        self.loss_dict: Dict = {}
        self.get_loss_dict()

        # Get Dataloader
        self.train_loader = tusimple_train_dataloader(
            batch_size=self.batch_size,
            data_root=self.data_root,
            griding_num=self.griding_num,
            use_aux=self.use_aux,
            num_lanes=self.num_lanes,
        )

        # Instantiate Model
        self.model = StructureAwareModel(
            pretrained=use_pretrained,
            backbone=backbone,
            cls_dim=(self.griding_num + 1, NUM_ROW_ANCHORS, self.num_lanes),
            use_aux=self.use_aux,
        ).cuda()

        # Model Parameters
        self.model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )

        # Instantiate Optimizer
        self.optimizer = torch.optim.Adam(
            self.model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Instantiate Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.epochs * len(self.train_loader),  # type: ignore
        )

    def inference(self, batch) -> Dict:
        """
        inference Perform model inference on a single batch

        :param batch: A batch of data from the dataloader
        :return: Custom dictionary output
        :rtype: Dict
        """
        # If Segmentation Head is being used
        if self.use_aux:
            img, classification_label, segmentation_label = batch
            # Convert to torch Tensors
            img, classification_label, segmentation_label = (
                img.cuda(),
                classification_label.long().cuda(),
                segmentation_label.long().cuda(),
            )

            classification_output, segmentation_output = self.model(img)

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

            classification_output = self.model(img)

            return {"cls_out": classification_output}

    def train_fn(self) -> Dict:
        """
        Train the Model for one epoch

        :return: A dictionary with the updated metrics
        :rtype: Dict
        """
        # Iterate over batches of data
        for _, batch in track(sequence=enumerate(self.train_loader), total=len(self.train_loader)):  # type: ignore # pylint: disable=C0301
            # Reset Metric counters
            reset_metrics(self.metric_dict)
            # Get Output from the model
            results = self.inference(batch)
            # Calculate the loss
            loss = self.calculate_loss(self.loss_dict, results)
            # Perform backpropagation and update Optimizer and Scheduler
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Get Argmax of the outputs
            results = self.resolve_val_data(results)

        # Update and calculate metrics
        update_metrics(self.metric_dict, results)

        return self.metric_dict

    def train(self) -> None:
        """
        Trains the model based on the specified parameters
        """
        # Put the model into training mode
        # Let the work begin !!!
        self.model.train()

        for epoch in range(self.epochs):

            # Get updated metric values after an epoch
            updated_dict = self.train_fn()

            # Print metrics
            for me_name, me_op in zip(updated_dict["name"], updated_dict["op"]):
                print(f"Epoch: {epoch} | metric/{me_name}: {me_op.get()}")

            # Save Model Checkpoints
            if epoch % 10 == 0:
                model_state_dict = self.model.state_dict()
                state = {
                    "model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                }
                assert os.path.exists("checkpoints/"), os.makedirs("./checkpoints/")  # type: ignore # pylint: disable=C0321
                torch.save(state, f"checkpoints/ep{epoch}.pth")

    def get_loss_dict(self):
        """
        Creates a custom dictionary for the various loss functions used in
        the training phase

        :return: custom dictionary containing various loss functions
        :rtype: Dict
        """

        if self.use_aux:
            self.loss_dict = {
                "name": ["cls_loss", "relation_loss", "aux_loss", "relation_dis"],
                "op": [
                    SoftmaxFocalLoss(2),
                    LaneSimilarityLoss(),
                    torch.nn.CrossEntropyLoss(),
                    LaneShapeLoss(),
                ],
                "weight": [1.0, self.sim_loss_w, 1.0, self.shp_loss_w],
                "data_src": [
                    ("cls_out", "cls_label"),
                    ("cls_out",),
                    ("seg_out", "seg_label"),
                    ("cls_out",),
                ],
            }
        else:
            self.loss_dict = {
                "name": ["cls_loss", "relation_loss", "relation_dis"],
                "op": [SoftmaxFocalLoss(2), LaneSimilarityLoss(), LaneShapeLoss()],
                "weight": [1.0, self.sim_loss_w, self.shp_loss_w],
                "data_src": [("cls_out", "cls_label"), ("cls_out",), ("cls_out",)],
            }

    def get_metric_dict(self):
        """
        Creates a custom dictionary for the various metrics used in the training phase

        :return: custom dictionary containing various loss metrics
        :rtype: Dict
        """

        if self.use_aux:
            self.metric_dict = {
                "name": ["top1", "top2", "top3", "iou"],
                "op": [
                    MultiLabelAcc(),
                    AccTopk(self.griding_num, 2),
                    AccTopk(self.griding_num, 3),
                    IoU(self.num_lanes + 1),
                ],
                "data_src": [
                    ("cls_out", "cls_label"),
                    ("cls_out", "cls_label"),
                    ("cls_out", "cls_label"),
                    ("seg_out", "seg_label"),
                ],
            }
        else:
            self.metric_dict = {
                "name": ["top1", "top2", "top3"],
                "op": [
                    MultiLabelAcc(),
                    AccTopk(self.griding_num, 2),
                    AccTopk(self.griding_num, 3),
                ],
                "data_src": [
                    ("cls_out", "cls_label"),
                    ("cls_out", "cls_label"),
                    ("cls_out", "cls_label"),
                ],
            }

    def calculate_loss(self, loss_dict: Dict, results: Dict) -> torch.Tensor:
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

    def resolve_val_data(self, results: Dict) -> Dict:
        """
        Returns the Argmax of the model outputs(s)

        :param results: custom dictionary containing outputs from the model
        :type results: Dict
        :return: Validated Results
        :rtype: Dict
        """
        results["cls_out"] = torch.argmax(results["cls_out"], dim=1)
        if self.use_aux:
            results["seg_out"] = torch.argmax(results["seg_out"], dim=1)
        return results
