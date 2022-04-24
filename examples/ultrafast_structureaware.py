# -*- coding: utf-8 -*-
"""
Complete Training Script for 'Ultra Fast Structure-aware Deep Lane Detection'
by Zequn Qin, Huanyu Wang, and Xi Li.
"""
from __future__ import annotations

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from rich import print

from src.io.dataloader import tusimple_train_dataloader
from src.models.structure_aware_model import structure_aware_loss_dict
from src.models.structure_aware_model import structure_aware_metric_dict
from src.models.structure_aware_model import structure_aware_train_fn
from src.models.structure_aware_model import StructureAwareModel
from src.utils import str2bool


def seed_everything(seed: int = 42) -> None:
    """
    Courtesy of https://twitter.com/kastnerkyle/status/1473361479143460872?
    """
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    random.seed(seed)
    tseed = random.randint(1, 1000000)
    tcseed = random.randint(1, 1000000)
    npseed = random.randint(1, 1000000)
    ospyseed = random.randint(1, 1000000)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed(tcseed)
    np.random.seed(npseed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)


if __name__ == "__main__":
    # Constants and other utilities
    NUM_ROW_ANCHORS = 56
    seed_everything(seed=42)

    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data/tusimple/", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--griding_num", default=100, type=int)
    parser.add_argument("--backbone", default="18", type=str)
    parser.add_argument("--use_aux", default=True, type=str2bool)
    parser.add_argument("--use_pretrained", default=True, type=str2bool)
    parser.add_argument("--num_lanes", default=4, type=int)
    parser.add_argument("--learning_rate", default=4e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup", default="linear", type=str)
    parser.add_argument("--warmup_iters", default=100, type=int)
    parser.add_argument("--sim_loss_w", default=1.0, type=float)
    parser.add_argument("--shp_loss_w", default=0.0, type=float)
    args = parser.parse_args()

    train_loader = tusimple_train_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        griding_num=args.griding_num,
        use_aux=args.use_aux,
        num_lanes=args.num_lanes,
    )

    model = StructureAwareModel(
        pretrained=args.use_pretrained,
        backbone=args.backbone,
        cls_dim=(args.griding_num + 1, NUM_ROW_ANCHORS, args.num_lanes),
        use_aux=args.use_aux,
    ).cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(
        model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs * len(train_loader),  # type: ignore
    )

    metric_dict = structure_aware_metric_dict(args)
    loss_dict = structure_aware_loss_dict(args)

    for epoch in range(args.epochs):

        updated_dict = structure_aware_train_fn(
            model=model,
            data_loader=train_loader,
            loss_dict=loss_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_dict=metric_dict,
            use_aux=args.use_aux,
        )

        for me_name, me_op in zip(updated_dict["name"], updated_dict["op"]):
            print(f"Epoch: {epoch} | metric/{me_name}: {me_op.get()}")

        if epoch % 10 == 0:
            MODEL_STATE_DICT = model.state_dict()
            state = {"model": MODEL_STATE_DICT, "optimizer": optimizer.state_dict()}
            assert os.path.exists("checkpoints/"), os.makedirs("./checkpoints/")  # type: ignore # pylint: disable=C0321
            torch.save(state, f"checkpoints/ep{epoch}.pth")
