<div align="center">

## ⚡️Target-Driven-Distillation⚡️

[[Paper]](https://arxiv.org/pdf) [[Project Page ✨]](https://g-u-n.github.io/projects/tdd/) [[Pre-trained Models in 🤗Hugging Face]](https://huggingface.co/redaigc/tdd_Weights) [[Demo]](https://huggingface.co/spaces/radames/Target-Driven-Distillation-tdd) [[Civitai]](https://civitai.com/models/487106/TDD-loras-of-stable-diffusion-xl-for-fast-image-generation)  ![visitors](https://visitor-badge.laobi.icu/badge?page_id=G-U-N.Target-Driven-Distillation)


</div>

<div align="center">
  <img src="_assets_/teaser/teaser.png" alt="teaser" style="zoom:80%;" />
</div>



### News
- [2024.08]: Release the training script of TDD-LoRA with Stable Diffusion XL. See [text_to_image_SDXL](https://github.com/G-U-N/Target-Driven-Distillation/blob/master/code/text_to_image_sdxl/run.sh). Release the weights of TDD-LORA with Stable Diffusion XL.  See [tdd_Weights](https://huggingface.co/redaigc/tdd_Weights).

| TDD-SDXL-2step-Deterministic | TDD-SDXL-4step-Deterministic    | TDD-SDXL-Stochastic (treat it as a clearer LCM) |
|:--------------------------------------:|:--------------------------------------:|:--------------------------------------:|
| ![Image 1](assets/imgs/SDXL_2step.png) | ![Image 2](assets/imgs/SDXL_4step_deterministic.png) | ![Image 3](assets/imgs/SDXL_4step_stochastic.png) |


- [2024.08.23]: [Hugging Face Demo](https://huggingface.co/spaces/radames/Target-Driven-Distillation-tdd) is available. Thanks [@radames](https://github.com/radames) for the commit!
- [2024.08.22]: Release TDD-LoRA weights of [Stable Diffusion v1.5](https://huggingface.co/redaigc/tdd_SD15_LoRAs/tree/main) and [Stable Diffusion XL](https://huggingface.co/redaigc/tdd_SDXL_LoRAs/tree/main) on huggingface.
- [2024.08.21]: Release Training Script of TDD-LoRA with Stable Diffusion v1.5. See [tran_tdd_lora_sd15.sh](code/text_to_image_sd15/train_tdd_lora_sd15.sh).
  >  We train the weights with 8 A 800. But my tentative experimental results suggest that using just one GPU can still achieve good results.

- [2024.08.20]: [Technical report](https://arxiv.org/pdf/2405.18407) is available on arXiv.

| One-Step Generation Comparison by HyperSD | One-Step Generation Comparison by tdd|
|:--------------:|:-----------:|
| ![hypersd](assets/imgs/hypersd.png) | ![ours](assets/imgs/ours.png) |
| hypersd | ours |

Our model has **clearly better generation diversity** than the cocurrent work HyperSD.

## Introduction

Target-Driven-Distillation (tdd) is (probably) current one of the most powerful sampling acceleration strategy for fast text-conditioned image generation in large diffusion models. 

Consistency Model (CM), proposed by Yang Song et al, is a promising new famility of generative models that can generate high-fidelity images with very few steps (generally 2 steps) under the unconditional and class-conditional settings.  Previous work, latent-consistency model (LCM), tried to replicate the power of consistency models for text-conditioned generation, but generally failed to achieve pleasant results, especially in low-step regime (1~4 steps). Instead,  we believe tdd is a much more successful extension to the original consistency models for high-resolution, text-conditioned image generation, better replicating the power of original consistency models for more advanced generation settings.

Generally, we show there are mainly three limitations of tdd:

- LCM lacks flexibility for CFG choosing and is insensitive to negative prompts.
- LCM fails to produce consistent results under different inference steps. Its results are blurry when step is too large (Stochastic sampling error) or small (inability).
- LCM produces bad and blurry results at low-step regime.

These limitaions can be explicitly viewed from the following figure.

***We generalize the design space of consistency models for high-resolution text-conditioned image generation, analyzing and tackling the limitations in the previous work LCM.***

<div align="center">
  <img src="assets/imgs/flaws.png" alt="teaser" style="zoom:15%;" />
</div>


**The core idea of our method is phasing the whole ODE trajectory into multiple sub-trajectories.**  The following figure illustrates the learning paradigm difference among diffusion models (DMs), consistency models (CMs), consistency trajectory models (CTMs), and our proposed Target-Driven-Distillations (tdds).


<div align="center">
  <img src="assets/imgs/diff.png" alt="teaser" style="zoom:80%;" />
</div>


For a better comparison, we also implement a baseline, which we termed as simpleCTM. We adapt the high-level idea of CTM from the k-diffusion framework into the DDPM framework with stable diffusion, and compare its performance. When trained with the same resource, our method achieves significant superior performance. 


## Samples of tdd

tdd can achieve text-conditioned image synthesis with good quality in 1, 2, 4, 8, 16 steps. 

<div align="center">
  <img src="assets/imgs/teaser.png" alt="teaser" style="zoom:80%;" />
</div>

## Comparison

tdd achieves advanced generation results compared with current open-available powerful fast generation models, including the GAN-based methods: SDXL-Turbo, SD-Turbo, SDXL-Lightning; rectified-flow-based method: InstaFlow;  CM-based methods: LCM, SimpleCTM. 

<div align="center">
  <img src="assets/imgs/comparison.png" alt="comparison" style="zoom:80%;" />
</div>



## Concact & Collaboration

If you have any questions about the code, please do not hesitate to contact me!

Email: polu@xiaohongshu.com


