import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Load trained generator
generator = tf.keras.models.load_model("generator_cifar10_1000epochs.keras")

app = FastAPI()

class Prompt(BaseModel):
    seed: int = None

@app.post("/generate")
def generate_image(data: Prompt):
    # If seed not provided, choose random
    seed = data.seed if data.seed is not None else np.random.randint(0, 999999)
    tf.random.set_seed(seed)

    noise = tf.random.normal([1, 100])
    generated = generator(noise, training=False)[0]

    # Convert [-1,1] → [0,255]
    img = ((generated + 1) * 127.5).numpy().astype("uint8")

    # Convert to base64 image
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode()

    return {
        "seed": seed,
        "image_base64": encoded_img
    }
