#!/usr/bin/env python3
"""
Example script to launch a VLA Model Server using vla_serving framework.
"""

import os
import sys
import os.path as osp
import argparse

# Add project paths
PROJECT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
VLA_SERVING_PATH = osp.join(PROJECT_DIR, 'vla-serving')

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
if VLA_SERVING_PATH not in sys.path:
    sys.path.append(VLA_SERVING_PATH)
    
from vla_serving.server import app, init_service

def main():
    parser = argparse.ArgumentParser(description="VLA Model Dummy Server")
    parser.add_argument(
        "--config", 
        type=str, 
        default=osp.join(osp.dirname(__file__), "dummy.yaml"),
        help="Path to the VLA service configuration file"
    )
    parser.add_argument("--port", type=int, default=5555, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    parser.add_argument("--log_folder", type=str, default=None, 
                       help="Path to folder for logging requests and responses")
    
    args = parser.parse_args()
    
    print(f"[VLA Server] Starting VLA model server...")
    print(f"[VLA Server] Config: {args.config}")
    print(f"[VLA Server] Host: {args.host}")
    print(f"[VLA Server] Port: {args.port}")
    print(f"[VLA Server] Log folder: {args.log_folder}")
    
    # Initialize the VLA service
    init_service(args.config, args.log_folder)
    
    # Run the Flask server
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
