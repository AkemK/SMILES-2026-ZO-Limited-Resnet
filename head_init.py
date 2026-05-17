"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.
    """
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)