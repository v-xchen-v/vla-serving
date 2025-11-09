"""
Example script to demonstrate how to use sdk/VLAClient to infer from a VLA server.
"""

from PIL import Image
import numpy as np
import io
from vla_serving.sdk import VLAClient

def create_dummy_images():
    """Create dummy images for testing."""
    images = []
    for i in range(4):  # head_rgb, head_depth, arm_rgb, arm_depth
        # Create a simple gradient image
        array = np.linspace(0, 255, 520*520*3, dtype=np.uint8).reshape((520, 520, 3))
        img = Image.fromarray(array)
        images.append(img)
    return images

def main():
    # Initialize the VLA client
    client = VLAClient(base_url="http://localhost:5555")

    # Create dummy images
    image_list = create_dummy_images()

    # Dummy task description and state
    task_description = "Pick up the red block and place it on the blue block."
    state = np.random.rand(6).astype(np.float32)  # Example robot state

    # Send inference request
    response = client.infer(
        images=image_list,
        task_description=task_description,
        state=state,
        image_format="JPEG",
        write_log=True  # Enable server-side logging
    )

    # Print the response
    print("Inference Response:")
    print(response)
    
if __name__ == "__main__":
    main()