import io
import json
from typing import Any, Dict, Tuple, List

import numpy as np
from PIL import Image
from .base_service import BaseModelService, ImageOrDepth
from .loader import build_service_from_config

try:
    import yaml
except ImportError:
    yaml = None
    

class ServingContext:
    """
    Holds the global model service and config for a running server.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.service: BaseModelService = build_service_from_config(cfg)

    def reset(self):
        self.service.reset()
    
def load_config(config_path: str) -> Dict[str, Any]:
    """Generic YAML config dict from <file>."""
    if yaml is None:
        raise ImportError("PyYAML is required to load configs.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def parse_image_from_requests(flask_request) -> List[ImageOrDepth]:
    """
    Expect files named image_0, image_1, ...
    We just load each as PIL.Image or np.ndarray (if you want special handling).
    """
    image_list = []
    for key in flask_request.files:
        if not key.startswith("image_"):
            continue
        idx = int(key[6:])
        image_file = flask_request.files[key]

        # Here: simple version, assume all are image-like
        img = Image.open(io.BytesIO(image_file.read()))
        img.load()
        image_list.append((idx, img))

    if not image_list:
        raise ValueError("No images received")

    image_list.sort(key=lambda x: x[0])
    return [x[1] for x in image_list]

def parse_query_json_from_request(flask_request) -> Dict[str, Any]:
    query_file = flask_request.files.get("json")
    if not query_file:
        raise ValueError("Missing json file")
    try:
        return json.load(query_file)
    except Exception as e:
        raise ValueError(f"Failed to parse json: {e}") from e