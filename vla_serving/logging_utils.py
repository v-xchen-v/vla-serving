# vla_serving/logging_utils.py
"""
Utility functions for logging requests and responses in the VLA server.
"""

import numpy as np
from PIL import Image


def save_depth_as_image(depth_array: np.ndarray, save_path: str, max_depth: float = 2000) -> None:
    ...
