import spaces
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel 
from tdd_scheduler import TDDScheduler
from safetensors.torch import load_file
from PIL import Image
import transformers
transformers.utils.move_cache()

SAFETY_CHECKER = False

loaded_acc = None
device = "cuda"

ACC_lora={
    "TDD":"sdxl_tdd_wo_adv_lora.safetensors",
    "TDD_adv":"sdxl_tdd_lora_weights.safetensors",
}

if torch.cuda.is_available():
    base1 = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    base2 = UNet2DConditionModel.from_pretrained(
        "frankjoshua/realvisxlV40_v40Bakedvae", subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=base1,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    pipe_sdxl.load_lora_weights("RED-AIGC/TDD", weight_name=ACC_lora["TDD"], adapter_name="TDD")
    pipe_sdxl.load_lora_weights("RED-AIGC/TDD", weight_name=ACC_lora["TDD_adv"], adapter_name="TDD_adv")
    pipe_sdxl.scheduler = TDDScheduler.from_config(pipe_sdxl.scheduler.config)

    pipe_sdxl_real = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=base2,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    pipe_sdxl_real.load_lora_weights("RED-AIGC/TDD", weight_name=ACC_lora["TDD"], adapter_name="TDD")
    pipe_sdxl_real.load_lora_weights("RED-AIGC/TDD", weight_name=ACC_lora["TDD_adv"], adapter_name="TDD_adv")
    pipe_sdxl_real.scheduler = TDDScheduler.from_config(pipe_sdxl.scheduler.config)

def update_base_model(ckpt):
    if torch.cuda.is_available():
        pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
    return pipe_sdxl


if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to(device)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
        has_nsfw_concepts = safety_checker(
            images=[images], clip_input=safety_checker_input.pixel_values.to(device)
        )

        return images, has_nsfw_concepts


@spaces.GPU(enable_queue=True, duration=3)
def generate_image(
    prompt,
    negative_prompt,
    ckpt,
    acc,
    num_inference_steps,
    guidance_scale,
    eta,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    global loaded_acc
    #pipe = pipe_sdxl #if mode == "sdxl" else pipe_sd15

    if ckpt == "Real":
        pipe = pipe_sdxl_real
    else:
        pipe = pipe_sdxl

    if loaded_acc != acc:
        #pipe.load_lora_weights(ACC_lora[acc], adapter_name=acc)
        pipe.set_adapters([acc], adapter_weights=[1.0])
        print(pipe.get_active_adapters())
        loaded_acc = acc

    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
    )

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(results.images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512))
        return images[0]
    return results.images[0]

css = """
h1 {
    text-align: center;
    display:block;
}
.gradio-container {
  max-width: 70.5rem !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
# ✨Target-Driven Distillation✨
Target-Driven Distillation (TDD) is a state-of-the-art consistency distillation model that largely accelerates the inference processes of diffusion models. Using its delicate strategies of *target timestep selection* and *decoupled guidance*, models distilled by TDD can generated highly detailed images with only a few steps.
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?logo=arxiv)](https://arxiv.org) [![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/RedAIGC/TDD)
"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt")
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt")
                with gr.Row():
                    steps = gr.Slider(
                        label="Sampling Steps",
                        minimum=4,
                        maximum=8,
                        step=1,
                        value=4,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1,
                        maximum=4,
                        step=0.1,
                        value=2.0,
                    )
                with gr.Row():
                    eta = gr.Slider(
                        label="eta",
                        minimum=0,
                        maximum=0.3,
                        step=0.1,
                        value=0.2,
                    )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1)

                with gr.Row():

                    ckpt = gr.Dropdown(
                        label="Base Model",
                        choices=["SDXL-1.0", "Real"],
                        value="SDXL-1.0",
                    )

                    acc = gr.Dropdown(
                        label="Accelerate Lora",
                        choices=["TDD", "TDD_adv"],
                        value="TDD_adv",
                    )

        with gr.Column(scale=1):
            with gr.Group():
                img = gr.Image(label="TDD Image", value="cat.png")
                submit_sdxl = gr.Button("Run on SDXL")
    gr.Examples(
        examples=[
            ["A photo of a cat made of water.", "", "SDXL-1.0", "TDD_adv", 4, 1.7, 0.2, 546237],
            ["A photo of a dog made of water.", "", "SDXL-1.0", "TDD_adv", 4, 1.7, 0.2, 546237],

        ],
        inputs=[prompt, negative_prompt, ckpt, acc, steps, guidance_scale, eta, seed],
        outputs=[img],
        fn=generate_image,
        cache_examples="lazy",
    )

    gr.on(
        fn=generate_image,
        triggers=[ckpt.change, prompt.submit, submit_sdxl.click],
        inputs=[prompt, negative_prompt, ckpt, acc, steps, guidance_scale, eta, seed],
        outputs=[img],
    )

demo.queue(api_open=False).launch(show_api=False)
