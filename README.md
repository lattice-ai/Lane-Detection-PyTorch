# Implementation of Lane Detection in PyTorch

This repository is aimed to contain implementations of various Lane Detection methods in PyTorch.

As of now, it only contains an implementation of '[**Ultra Fast Structure-aware Deep Lane Detection**](https://arxiv.org/pdf/2004.11757.pdf)' by Zequn Qin, Huanyu Wang, and Xi Li. To run the pipeline use this snippet

```python
from src.models.structure_aware_model import StructureAwareTrainer

trainer = StructureAwareTrainer(
    epochs=100,  # epochs to train for
    data_root="./data/tusimple/",  # Root Path to TUSimple
    use_pretrained=True,  # whether to use a pretrained backbone
    backbone="18",  # Which ResNet backbone to use only 18 and 34 are allowed
    griding_num=100,  # Number of Gridding Cells
    num_lanes=4,  # Maximum lanes per image
    use_aux=True,  # For training we use the auxiliary branch
    learning_rate=4e-4,  # Base Learning Rate
    weight_decay=1e-4,  # Weight Decay Value
    batch_size=32,  # Batch size for the dataloader
    sim_loss_w=1.0,  # Weight for the Similarity Loss Fn
    shp_loss_w=0.0,  # Weight for the Shape Loss Fn
)

trainer.train()
```
