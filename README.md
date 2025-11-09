# VLA Serving

A lightweight serving framework for Vision-Language-Action (VLA) models.

## Installation
```bash
git clone https://github.com/v-xchen-v/vla-serving.git
cd vla-serving
pip install -e .
```

## Usage
### As a server
```bash
python -m vla_serving.server --config path/to/config.yaml --port 5000
```
### As a library
```python
from vla_serving import BaseModelService, build_service_from_config

# Implement your model service
class MyVLAService(BaseModelService):
    @classmethod
    def from_config(cls, cfg):
        return cls()
    
    def step(self, image_list, task_description=None, state=None, **kwargs):
        # Your inference logic here
        return {"action": [1.0, 0.5, 0.0]}

# Use the service
config = {"backend_config": {...}}
service = build_service_from_config(config)
```