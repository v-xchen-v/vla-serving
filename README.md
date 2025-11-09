# VLA Serving

A lightweight serving framework for Vision-Language-Action (VLA) models.

## Installation
```bash
git clone https://github.com/v-xchen-v/vla-serving.git
cd vla-serving
pip install -e .
```

## Examples

### Quick Start with Dummy Service

The easiest way to get started is with the included dummy service that demonstrates the basic API:

#### 1. Start the server
```bash
cd path-to-vla_serving
# Start the dummy service server
python -m vla_serving.server --config examples.yaml --port 555
```

#### 2. Test with curl
```bash
# Create test data
echo '{"task_description": "pick up the red cube", "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}' > query.json

# Create a dummy test image (requires ImageMagick)
convert -size 640x480 xc:blue test_image.jpg

# Make inference request
curl -X POST http://localhost:5555/inference \
  -F "image_0=@test_image.jpg" \
  -F "json=@query.json"
```


#### 3. Expected Response
```json
{
  "action": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
  "metadata": {
    "model_name": "DummyService",
    "num_images": 1,
    "task_description": "pick up the red cube"
  }
}
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

## Workflow when you "meet a new model"

When you want to integrate a new VLA model to use the serving framework.

### Step 1: Understand the Model Interface

First, analyze your new model to understand:
- **Input format**: What types of images does it expect? (RGB, depth, segmentation masks?)
- **State representation**: What robot state information does it need? (joint positions, gripper state, etc.)
- **Task specification**: How does it receive task instructions? (text prompts, goal images, etc.)
- **Output format**: What does it return? (action vectors, waypoints, discrete actions?)

### Step 2: Create a Model Service Adapter
Create a new service class that inherits from `BaseModelService`:
```python
# my_model_service.py
from typing import Any, Dict, List, Optional
import numpy as np
from vla_serving import BaseModelService
from vla_serving.base_service import ImageOrDepth

class MyModelService(BaseModelService):
    def __init__(self, model_path: str, device: str = "cuda"):
        # Initialize your model here
        self.model = load_my_model(model_path, device)
        self.device = device
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MyModelService":
        """Factory method to create service from config."""
        return cls(
            model_path=cfg["model_path"],
            device=cfg.get("device", "cuda")
        )
    
    def step(
        self, 
        image_list: List[ImageOrDepth],
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference and return action."""
        # Preprocess inputs
        images = self._preprocess_images(image_list)
        processed_state = self._preprocess_state(state)
        
        # Run model inference
        with torch.no_grad():
            action = self.model.predict(
                images=images,
                state=processed_state,
                instruction=task_description
            )
        
        # Return JSON-serializable result
        return {
            "action": action.tolist(),
            "confidence": float(self.model.last_confidence),
            "metadata": {
                "model_name": "MyModel",
                "num_images": len(image_list)
            }
        }
    
    def _preprocess_images(self, image_list: List[ImageOrDepth]) -> torch.Tensor:
        # Convert PIL images to model format
        pass
    
    def _preprocess_state(self, state: Optional[np.ndarray]) -> torch.Tensor:
        # Convert state to model format
        pass
```

### Step 3: Create Configuration FileCreate a YAML config file for your model:

```yaml
# my_model_config.yaml
model_name: "MyAwesomeVLAModel"

backend_class: "path.to.my_model_service.MyModelService"

backend_config:
  model_path: "/path/to/model/weights.pth"
  device: "cuda"
  batch_size: 1
  # Add any model-specific parameters
```

### Step 4: Test Your Integration

#### 4.1 Unit Test Your Service
```python
# test_my_service.py
from PIL import Image
import numpy as np

# Test your service directly
config = {"model_path": "/path/to/model.pth", "device": "cpu"}
service = MyModelService.from_config(config)

# Create dummy inputs
dummy_image = Image.new('RGB', (640, 480), color='red')
dummy_state = np.array([0.1, 0.2, 0.3, 0.4])

# Test inference
result = service.step(
    image_list=[dummy_image],
    task_description="pick up the red cube",
    state=dummy_state
)

print(f"Action: {result['action']}")
```

#### 4.2 Test with Server
```bash
# Start the server
python -m vla_serving.server --config my_model_config.yaml --port 5555

# Test with curl (in another terminal)
echo '{"task_description": "pick up object", "state": [0,0,0,0,0,0,0]}' > query.json

curl -X POST http://localhost:5555/inference \
  -F "image_0=@test_image.jpg" \
  -F "json=@query.json"
```

### Step 5: Handle Common Integration Challenge


#### Image Format Mismatches
```python
def _preprocess_images(self, image_list: List[ImageOrDepth]) -> torch.Tensor:
    processed = []
    for img in image_list:
        if isinstance(img, PIL.Image.Image):
            # Convert PIL to numpy
            img_array = np.array(img)
        elif isinstance(img, np.ndarray):
            img_array = img
        
        # Resize if needed
        if img_array.shape[:2] != (224, 224):
            img_array = cv2.resize(img_array, (224, 224))
        
        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        processed.append(img_array)
    
    return torch.tensor(np.stack(processed))
```

#### State Dimension Mismatches
```python
def _preprocess_state(self, state: Optional[np.ndarray]) -> torch.Tensor:
    if state is None:
        # Return zero state with expected dimensions
        state = np.zeros(7)  # 7-DOF robot arm
    
    # Ensure correct shape
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    
    return torch.tensor(state, dtype=torch.float32)
```

#### Error Handling
```python
def step(self, image_list, task_description=None, state=None, **kwargs):
    try:
        # Your inference code
        action = self.model.predict(...)
        return {"action": action.tolist()}
    except Exception as e:
        # Log error and return safe action
        print(f"Model inference failed: {e}")
        return {
            "action": [0.0] * 7,  # Safe zero action
            "error": "inference_failed",
            "details": str(e)
        }
```