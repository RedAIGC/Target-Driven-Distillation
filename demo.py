# !pip install opencv-python transformers accelerate 
import torch
import diffusers
from diffusers import StableDiffusionXLPipeline
from tdd_scheduler import TDDScheduler

device = "cuda"
tdd_lora_path = "tdd_lora/sdxl_tdd_lora_weights.safetensors"

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to(device)

pipe.scheduler = TDDScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(tdd_lora_path, adapter_name="accelerate")
pipe.fuse_lora()

prompt="A photo of a cat made of water."

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1.7,
    eta=0.2, 
    generator=torch.Generator(device=device).manual_seed(546237),
).images[0]
