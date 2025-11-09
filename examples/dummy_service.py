# examples/dummy_service.py
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

try:
    from vla_serving import BaseModelService
    from vla_serving.base_service import ImageOrDepth
except ImportError:
    # Fallback for development/local imports
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vla_serving import BaseModelService
    from vla_serving.base_service import ImageOrDepth


class DummyService(BaseModelService):
    """
    A simple example adapter showing how to implement a model service.
    
    It pretends to process images + state + prompt and resturn a fake action.
    """
    def __init__(self, scale=1.0, offset=0.0):
        self.scale = scale
        self.offset = offset
        print(f"[DummyService] Initialized with scale={scale}, offset={offset}")
        
    # ------------------------------------------------------------------
    # Required factory method
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DummyService":
        """
        Example of building a service from YAML config dict.
        """
        return cls(
            scale=cfg.get("scale", 1.0),
            offset=cfg.get("offset", 0.0),
        )
        
    # ------------------------------------------------------------------
    # Required inference method
    # ------------------------------------------------------------------
    def step(
        self,
        image_list: List[ImageOrDepth],
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate an inference call.
        - image_list: list of PIL images or np.ndarray
        - task_description: instruction string
        - state: np.ndarray (optional)
        """
        num_images = len(image_list)
        prompt = task_description or "(none)"
        state_mean = float(np.mean(state)) if state is not None else 0.0
        
        # fake "action" computed from arbitrary formula
        fake_action = self.scale * (state_mean + num_images) + self.offset
        
        print(f"[DummyService] step called with {num_images} images, prompt='{prompt}', "
              f"state_mean={state_mean:.4f} => fake_action={fake_action:.4f}")
        
        return {
            "action": [fake_action], # return as a list for generality
            "meta": {
                "num_images": num_images,
                "prompt": prompt,
                "scale": self.scale,
                "offset": self.offset
            }
        }