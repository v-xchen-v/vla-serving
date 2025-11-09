"""VLA Serving: A framework for serving Vision-Language-Action models."""

__version__ = "0.1.0"

from .base_service import BaseModelService
from .loader import build_service_from_config

__all__ = [
    "BaseModelService",
    "build_service_from_config",
]