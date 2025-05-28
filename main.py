from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO

app = FastAPI()

# Load DreamShaper XL model
model_id = "leveragedpixel/DreamShaper-XL1.0"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

# Define request schema
class PromptRequest(BaseModel):
    prompt: str

# Image generation route
@app.post("/generate")
def generate_image(req: PromptRequest):
    # Generate image from prompt
    image = pipe(req.prompt, num_inference_steps=30).images[0]

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return result
    return {
        "status": "success",
        "image": img_str
    }
