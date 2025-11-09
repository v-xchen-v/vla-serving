# vla_serving/sdk.py
"""
SDK-style client for the vla_serving HTTP API.

Provides a VLAClient class that hides:
- multi-part/form-data construction
- JSON serialization
- HTTP request/response handling

You just pass images, task_description, and state, and get back the parsed response.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from .server_core import convert_ndarray_to_list

import numpy as np
import requests
from PIL import Image

ImageType = Union[Image.Image, np.ndarray, None]

def _encode_image(img: ImageType, idx: int, image_format: str) -> Optional[tuple]:
    """
    Encode a single image into a (field_name, file_tuple) for requests.
    Returns None if img is None.
    """
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            arr = img
        pil_img = Image.fromarray(arr)
        
    if not isinstance(img, Image.Image):
        raise  TypeError(f"Unsupported image type at index {idx}: {type(img)}")
    
    img_bytes = io.BytesIO()
    fmt = image_format.upper()
    if fmt == "JPEG":
        img.save(img_bytes, format="JPEG")
        mime = 'image/jpeg'
        filename = f'image_{idx}.jpg'
    elif fmt == "PNG":
        img.save(img_bytes, format="PNG")
        mime = 'image/png'
        filename = f'image_{idx}.png'
    else:
        raise ValueError("Unsupported image format. Use 'JPEG' or 'PNG'.")
    img_bytes.seek(0)
    return (f'image_{idx}', (filename, img_bytes, mime))

def _build_files(
    image_list: Sequence[ImageType],
    payload_json: Dict[str, Any],
    image_format: str,
) -> List[tuple]:
    """
    Build the files list for requests from images and JSON payload.
    """
    files = []
    for i, img in enumerate(image_list):
        encoded = _encode_image(img, i, image_format)
        if encoded is not None:
            files.append(encoded)
    
    json_bytes = io.BytesIO(json.dumps(payload_json).encode('utf-8'))
    files.append(('json', ('data.json', json_bytes, 'application/json')))
    return files


class VLAClient:
    """
    Simple VLA client for a running vla_serving server.
    
    Example:
        client = VLAClient("http://localhost:5000", image_format="PNG")
        action = client.infer(
            images=[image0, image1],
            task_description="Pick up the red can.",
            state=robot_state_dict,
        )
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: Optional[float] = 30.0,
        default_image_format: str = "JPEG",
    ) -> None:
        """
        base_url: e.g., "http://localhost:5000"
        timeout: request timeout in seconds (None = no timeout)
        default_image_format: "JPEG" or "PNG"
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_image_format = default_image_format
        
        if self.default_image_format not in ["JPEG", "PNG"]:
            raise ValueError("default_image_format must be 'JPEG' or 'PNG'")
        
    @property
    def inference_url(self) -> str:
        return f"{self.base_url}/api/inference"
    
    def infer(
        self,
        images: Sequence[ImageType],
        task_description: str,
        state: Any = None,
        image_format: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Call /api/inferene on the server.
        
        images:
            Sequence of images (PIL.Image, np.ndarray, or None)
            Each non-None image is sent as image_<index>.
        task_description:
            Instruction string.
        state:
            Arbitrary state object (dict/list/np.ndarray) passed as 'state' in JSON.
        image_format:
            "JPEG" or "PNG". If None, uses default_image_format.
        extra_fields:
            Additional JSON fields to send along with 'task_description' and 'state'.
            e.g. {"use_state": False, "write_log": True}
        verbose:
            If True, prints timing info.
            
        Returns:
            Parsed JSON response from the server on success.
        Raises:
            RuntimeError on non-200 responses or request errors.
        """
        fmt = (image_format or self.default_image_format).upper()
        if fmt not in ["JPEG", "PNG"]:
            raise ValueError("image_format must be 'JPEG' or 'PNG'")
        
        if extra_fields is None:
            extra_fields = {}
            
        # convert state to list if it's a numpy array
        state = convert_ndarray_to_list(state)    
        
        payload_json: Dict[str, Any] = {
            "task_description": task_description,
            "state": state,
            "extra_fields": extra_fields,
        }
        
        t0 = time.time()
        files = _build_files(images, payload_json, fmt)
        t1 = time.time()
        
        if verbose:
            print(f"[VLAClient] Built request in {t1 - t0:.3f} seconds")
            
        try:
            response = requests.post(
                self.inference_url,
                files=files,
                timeout=self.timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Request failed: {str(e)}") from e
        
        t2 = time.time()
        if verbose:
            print(f"[VLAClient] Received response in {t2 - t1:.3f} seconds")

        if response.status_code != 200:
            msg = f"Server returned status {response.status_code}: {response.text}"
            if verbose:
                print(f"[VLAClient] {msg}")
            raise RuntimeError(msg)
        
        try:
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON response: {str(e)}") from e

        return result
