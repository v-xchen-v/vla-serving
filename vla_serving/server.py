# vla_serving/server.py
import argparse
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from .loader import build_service_from_config
# from .logging_utils import save_depth_as_image

from .server_core import load_config, parse_image_from_requests, parse_query_json_from_request
from .server_core import convert_ndarray_to_list

app = Flask(__name__)

SERVICE = None
SERVICE_CONFIG: Dict[str, Any] = {}
MODEL_NAME: str = ""
LOG_FOLDER: str = ""

def init_service(config_path: str, log_folder: Optional[str] = None):
    global SERVICE, SERVICE_CONFIG, MODEL_NAME, LOG_FOLDER
    SERVICE_CONFIG = load_config(config_path)
    SERVICE = build_service_from_config(SERVICE_CONFIG)
    MODEL_NAME = SERVICE_CONFIG.get("model_name", "default_model")
    
    # Set up log folder
    if log_folder is not None:
        LOG_FOLDER = log_folder
    else:
        LOG_FOLDER = "logging/api_logs"
    
    os.makedirs(LOG_FOLDER, exist_ok=True)
    print(f"[Server] Initialized service for model '{MODEL_NAME}'")
    print(f"[Server] Log folder: {LOG_FOLDER}")
    
@app.route("/api/inference", methods=["POST"])
def inference():
    global SERVICE, LOG_FOLDER
    if SERVICE is None:
        return jsonify({"error": "Service not initialized"}), 500
    
    try:
        image_list = parse_image_from_requests(request)
        query_json = parse_query_json_from_request(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    write_log = True
    name = None
    
    # Extract write_log option
    write_log = query_json.pop("write_log", True)
    
    # Set up logging
    if write_log:
        name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        task_desc = query_json.get("task_description", "")
        name += "_" + task_desc.replace(" ", "_")[:20]
        
        # Log images
        for i, image in enumerate(image_list):
            if isinstance(image, Image.Image):
                image.save(os.path.join(LOG_FOLDER, f"{name}_{i}.jpg"), format="JPEG")
            # else:
            #     save_depth_as_image(image, os.path.join(LOG_FOLDER, f"{name}_{i}.jpg"), 2000)
        
        # Log input query
        with open(os.path.join(LOG_FOLDER, f"{name}_query.json"), "w") as f:
            json.dump(query_json, f, indent=4)
        
    # Process state field
    if "state" in query_json:
        s = query_json["state"]
        if isinstance(s, list):
            query_json["state"] = np.array(s, dtype=np.float32)
        elif isinstance(s, np.ndarray):
            pass
        elif isinstance(s, dict):
            state_dict = s
            for key in state_dict:
                if isinstance(state_dict[key], list):
                    state_dict[key] = np.array(state_dict[key], dtype=np.float32)
        else:
            raise ValueError("state should be a list, dict, or numpy array")
        
    try:
        result = SERVICE.step(
            image_list=image_list,
            task_description=query_json.get("task_description", None),
            state=query_json.get("state", None),
            **{k: v for k, v in query_json.items() if k not in ["task_description", "state"]},
        )
    
        # Log the result if logging is enabled
        if write_log and name:
            with open(os.path.join(LOG_FOLDER, f"{name}_answer.json"), "w") as f:
                json.dump(convert_ndarray_to_list(result), f, indent=4)
        
        # recursively convert all numpy arrays in result to lists
        result = convert_ndarray_to_list(result) # to prevent jsonify issues (Object of type ndarray is not JSON serializable)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": "Inference failed", "details": str(e), "model_name": MODEL_NAME}), 500

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--log_folder", type=str, default=None, 
                       help="Path to folder for logging requests and responses")
    args = parser.parse_args()

    init_service(args.config, args.log_folder)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()