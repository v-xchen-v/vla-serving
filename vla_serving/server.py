# vla_serving/server.py
import argparse
import io
import json
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from .loader import build_service_from_config

from .server_core import load_config, parse_image_from_requests, parse_query_json_from_request

app = Flask(__name__)

SERVICE = None
SERVICE_CONFIG: Dict[str, Any] = {}
MODEL_NAME: str = ""

def init_service(config_path: str):
    global SERVICE, SERVICE_CONFIG, MODEL_NAME
    SERVICE_CONFIG = load_config(config_path)
    SERVICE = build_service_from_config(SERVICE_CONFIG)
    MODEL_NAME = SERVICE_CONFIG.get("model_name", "default_model")
    print(f"[Server] Initialized service for model '{MODEL_NAME}'")
    
@app.route("/api/inference", methods=["POST"])
def inference():
    global SERVICE
    if SERVICE is None:
        return jsonify({"error": "Service not initialized"}), 500
    
    try:
        image_list = parse_image_from_requests(request)
        query_json = parse_query_json_from_request(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
        
    if "state" in query_json:
        s = query_json["state"]
        if isinstance(s, list):
            query_json["state"] = np.array(s, dtype=np.float32)
        
    try:
        result = SERVICE.step(
            image_list=image_list,
            task_description=query_json.get("task_description", None),
            state=query_json.get("state", None),
            **{k: v for k, v in query_json.items() if k not in ["task_description", "state"]},
        )
        
        # recursively convert all numpy arrays in result to lists

        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_ndarray(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarray(i) for i in obj]
            else:
                return obj

        result = convert_ndarray(result) # to prevent jsonify issues (Object of type ndarray is not JSON serializable)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": "Inference failed", "details": str(e), "model_name": MODEL_NAME}), 500

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    init_service(args.config)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()