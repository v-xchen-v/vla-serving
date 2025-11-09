# DummyService example

This example shows how to implement a custom model service using `vla_serving`.

## Files

- `dummy_service.py`  
  A minimal adapter implementing `BaseModelService`. It pretends to be a model:
  it looks at the number of images and the mean of the state vector and returns
  a fake `action`.

- `configs/dummy.yaml`  
  Example YAML config that points the serving core to this `DummyService`.

## How to run

From the **repo root** (the folder that contains `vla_serving/` and `examples/`):

```bash
cd vla_serving
python -m vla_serving.server --config examples/dummy.yaml --port 555
```
You'll see:
```
[DummyService] Initialized with scale=2.5, offset=1.0
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5555
```

## Test the API (example curl)
You can seed a fake request with any image and simple JSON
```
curl -X POST http://localhost:5555/api/inference \
  -F "image_0=@/path/to/test.jpg" \
  -F json=@<(printf '{"task_description": "test dummy model", "state": [1,2,3,4]}')
```
Respones:
```
{
    "action":[9.75],
    "meta": {
        "num_images":1,
        "offset":1.0,
        "prompt":"test dummy model",
        "scale":2.5
    }
}
```