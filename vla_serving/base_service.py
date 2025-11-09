# vla_serving/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

ImageOrDepth = Any # e.g., PIL.Image.Image or np.Array

class BaseModelService(ABC):
    """Common interface for all models to be served."""
    
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Dict[str, any]) -> "BaseModelService":
        """
        Build the service from a config dict.
        cfg is the 'backend_config' section of your YAML. 
        """
        
    @abstractmethod
    def step(
        self, 
        image_list: List[ImageOrDepth],
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single inference step.
        Must return a JSON-serializable dict.
        """
        ...      

    