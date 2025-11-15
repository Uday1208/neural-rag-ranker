# src/utils/device.py
"""
Device management and reproducibility utilities for PyTorch experiments.

This module abstracts over CPU vs CUDA device selection and centralizes
random seeding so that training and evaluation runs can be reproduced
more easily across machines.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Select the most appropriate torch.device for computation.

    Args:
        prefer_gpu:
            Whether to prefer a CUDA device when available.

    Returns:
        A torch.device instance pointing to "cuda" or "cpu".
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed:
            Integer seed value used for all involved random number generators.

    Notes:
        This function also sets environment variable PYTHONHASHSEED and
        configures cuDNN for deterministic behavior where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            # cuDNN may not be available in some environments (e.g., CPU-only)
            pass
