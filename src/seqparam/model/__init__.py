"""Model layer: the sequence CNN and its param/checkpoint files.

`ModelParams` is torch-free (safe to import without torch); `SeqCNN` and the
build/checkpoint helpers import torch.
"""
from .params import ModelParams

__all__ = ["ModelParams"]
