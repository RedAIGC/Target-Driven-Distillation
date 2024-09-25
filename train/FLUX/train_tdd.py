# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available
from dataset import SDXLText2ImageDataset


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.2")

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


@torch.no_grad()
def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    step,
    is_final_validation=False,
):
    logger.info("Running validation... ")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    print("Loading LORA weights... ")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()
    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, image in enumerate(images)
                    ]
                }
            )
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class EulerSolver:
    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50,
                 num_inference_steps_min=4, num_inference_steps_max=8):
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (
            np.arange(1, euler_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray([0] + self.euler_timesteps[:-1].tolist())
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()
        )
        # tdd parameters(DDPM)
        self.tdd_sigmas = sigmas
        self.merged_timesteps = self.set_timesteps_s(1000, num_inference_steps_min, num_inference_steps_max)
        
        # convert to torch tensors
        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.euler_timesteps_prev = torch.from_numpy(self.euler_timesteps_prev).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)
        # tdd parameters(DDPM)
        self.tdd_sigmas = torch.from_numpy(self.tdd_sigmas.copy())
        self.merged_timesteps = torch.from_numpy(self.merged_timesteps)

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.euler_timesteps_prev = self.euler_timesteps_prev.to(device)

        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        self.tdd_sigmas = self.tdd_sigmas.to(device)
        self.merged_timesteps = self.merged_timesteps.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_prev = extract_into_tensor(
            self.sigmas_prev, timestep_index, model_pred.shape
        )
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev
        
    def set_timesteps_s(
        self,
        num_train_timesteps=1000,
        num_inference_steps_min=4,
        num_inference_steps_max=8,
    ):
        merged_timesteps = []
        origin_timesteps = self.euler_timesteps[::-1].copy()
        for i in range(num_inference_steps_min, num_inference_steps_max + 1):
            inference_indices = np.linspace(0, len(origin_timesteps), num=i, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            timesteps = origin_timesteps[inference_indices]
            merged_timesteps.append(timesteps)
        merged_timesteps = np.concatenate(merged_timesteps)
        merged_timesteps = np.unique(merged_timesteps)
        merged_timesteps = np.sort(merged_timesteps)
        merged_timesteps = np.insert(merged_timesteps, 0, 0)[:-1]
        return merged_timesteps

    def select_s(self, timestep_index, s_ratio):
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, self.merged_timesteps.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= self.merged_timesteps
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = self.merged_timesteps.size(0) - 1 - last_valid_index
        lower_bounds = torch.clamp(self.merged_timesteps[last_valid_index] - 260, min=0) #260
        upper_bounds = self.merged_timesteps[last_valid_index]

        valid_indices = [torch.where((self.merged_timesteps >= lb) & (self.merged_timesteps <= ub))[0] for lb, ub in zip(lower_bounds, upper_bounds)]
        random_indices = [torch.randint(0, len(idx), (1,)).item() for idx in valid_indices]
        timestep_index = torch.tensor([self.merged_timesteps[valid_indices[i][ri]] for i, ri in enumerate(random_indices)], device=expanded_timestep_index.device)

        timestep_index = torch.floor(
            (1 - torch.rand(timestep_index.size(), device=timestep_index.device) * s_ratio) * timestep_index
        ).to(dtype=torch.long)
        return timestep_index

    def tdd_step(self, sample, model_pred, timestep_index, timestep_index_s, is_target=False):
        if is_target:
            sigma = extract_into_tensor(self.sigmas_prev, timestep_index, model_pred.shape)
        else:
            sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_prev_s = extract_into_tensor(
            self.tdd_sigmas, timestep_index_s, model_pred.shape
        )
        x_prev_s = sample + (sigma_prev_s - sigma) * model_pred
        return x_prev_s

def import_model_class_from_model_name_or_path(
    pretrained_teacher_model: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_teacher_model,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tdd-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--use_fix_crop_and_size",
        action="store_true",
        help="Whether or not to use the fixed crop and size for the teacher model.",
        default=False,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="sigma_sqrt",
        choices=["sigma_sqrt", "logit_normal", "mode"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--set_grad_to_none",
        action="store_true",
        help="Whether or not to delete the gradients when calling optimizer.zero_grad().",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.5,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_euler_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Using xformers----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--val_infer_step", default=4, type=int)
    parser.add_argument("--num_inference_steps_min", default=4, type=int)
    parser.add_argument("--num_inference_steps_max", default=8, type=int)
    
    parser.add_argument("--s_ratio", default=0.3, type=float)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")
    return args


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load tokenizers from FLUX checkpoint.
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer_2", revision=args.teacher_revision
    )

    # 2. Load text encoders from FLUX checkpoint.
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.teacher_revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.teacher_revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_teacher_model, subfolder="text_encoder", revision=args.teacher_revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_teacher_model, subfolder="text_encoder_2", revision=args.teacher_revision
    )

    # 3. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler"
    )
    # noise_scheduler_solver = FlowMatchEulerDiscreteScheduler.from_pretrained(
    #     args.pretrained_teacher_model, subfolder="scheduler"
    # )
    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two
    )
    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae_path = (
        args.pretrained_teacher_model
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.teacher_revision,
    )

    # 5. Create online (`transformer`) student transformer.
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="transformer", revision=args.teacher_revision
    )

    # 6. Load teacher transformer from FLUX checkpoint
    # teacher_transformer = FluxTransformer2DModel.from_pretrained(
    #     args.pretrained_teacher_model, subfolder="transformer", revision=args.teacher_revision
    # )

    # 7. Freeze teacher vae, text_encoders, transformer and teacher_transformer
    transformer.requires_grad_(False)
    # teacher_transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # 8. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    # teacher_transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    #print(noise_scheduler.config.num_train_timesteps)
    solver = EulerSolver(
        noise_scheduler.sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
        num_inference_steps_min=args.num_inference_steps_min, num_inference_steps_max=args.num_inference_steps_max
    )
    print(solver.merged_timesteps)
    solver.to(accelerator.device)

    # 9. Add LoRA to the student Transformer, only the LoRA projection matrix will be updated by the optimizer.
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2"],
        # target_modules=[
        #     "to_k", "to_q", "to_v", "to_out.0",
        #     "proj_in",
        #     "proj_out",
        #     "ff.net.0.proj",
        #     "ff.net.2",
        #     # new
        #     "context_embedder", "x_embedder",
        #     "linear", "linear_1", "linear_2",
        #     "proj_mlp",
        #     "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
        #     "ff_context.net.0.proj", "ff_context.net.2"],
    )
    transformer.add_adapter(transformer_lora_config)

    # may not be available yet
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            if xformers_version < version.parse("0.0.27"):
                logger.warn("the code may get stuck with the currently installed version of xformers, it is recommended to install a newer version (>= 0.0.27)")

            transformer.enable_xformers_memory_efficient_attention()
            # teacher_transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")
        
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 10. Handle saving and loading of checkpoints
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # 13. Dataset creation and data processing
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    dataset = SDXLText2ImageDataset(
        train_shards_path_or_url=args.train_shards_path_or_url,
        num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
        resolution=args.resolution,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        use_fix_crop_and_size=args.use_fix_crop_and_size,
        random_flip=args.random_flip,
    )
    train_dataloader = dataset.train_dataloader

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # 16. Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                pixel_values, prompts, _, _ = batch
                pixel_values = pixel_values.to(dtype=vae.dtype, device=vae.device)

                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    weight_dtype,
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                index = torch.randint(0, args.num_euler_timesteps, (bsz,), device=model_input.device).long()

                sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
                sigmas_prev = extract_into_tensor(solver.sigmas_prev, index, model_input.shape)
                timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).squeeze()
                timesteps_prev = (sigmas_prev * noise_scheduler.config.num_train_timesteps).squeeze()
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input


                guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                guidance = guidance.expand(model_input.shape[0])

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # 20.4.9. Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                # model_pred = FluxPipeline._unpack_latents(
                #     model_pred,
                #     height=int(model_input.shape[2] * vae_scale_factor / 2),
                #     width=int(model_input.shape[3] * vae_scale_factor / 2),
                #     vae_scale_factor=vae_scale_factor,
                # )
                timestep_index_s = solver.select_s(timesteps_prev, args.s_ratio)
                model_pred = solver.tdd_step(packed_noisy_model_input, model_pred, index, timestep_index_s)

                #weighting = sd3()

                with torch.no_grad():
                    with torch.autocast("cuda"):
                        # teacher_model_pred = teacher_transformer(
                        #     hidden_states=packed_noisy_model_input.float(),
                        #     # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                        #     timestep=timesteps / 1000,
                        #     guidance=guidance,
                        #     pooled_projections=pooled_prompt_embeds.float(),
                        #     encoder_hidden_states=prompt_embeds.float(),
                        #     txt_ids=text_ids.float(),
                        #     img_ids=latent_image_ids.float(),
                        #     return_dict=False,
                        # )[0]
                        
                        # use a single model to save VRAM
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
                        unwrapped_transformer.disable_adapters()
                        teacher_model_pred = unwrapped_transformer(
                            hidden_states=packed_noisy_model_input.float(),
                            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds.float(),
                            encoder_hidden_states=prompt_embeds.float(),
                            txt_ids=text_ids.float(),
                            img_ids=latent_image_ids.float(),
                            return_dict=False,
                        )[0]
                        unwrapped_transformer.enable_adapters()

                        x_prev = solver.euler_step(packed_noisy_model_input, teacher_model_pred, index)
                        # teacher_model_pred = FluxPipeline._unpack_latents(
                        #     teacher_model_pred,
                        #     height=int(model_input.shape[2] * vae_scale_factor / 2),
                        #     width=int(model_input.shape[3] * vae_scale_factor / 2),
                        #     vae_scale_factor=vae_scale_factor,
                        # )

                # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_pred = transformer(
                            hidden_states=x_prev.float(),
                            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                            timestep=timesteps_prev / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds.float(),
                            encoder_hidden_states=prompt_embeds.float(),
                            txt_ids=text_ids.float(),
                            img_ids=latent_image_ids.float(),
                            return_dict=False,
                        )[0]
                    target = solver.tdd_step(x_prev, target_pred, index, timestep_index_s, True)

                # 20.4.13. Calculate loss
                if args.loss_type == "l2":
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                elif args.loss_type == "huber":
                    loss = torch.mean(
                        torch.sqrt(
                            (model_pred.float() - target.float()) ** 2 + args.huber_c**2
                        )
                        - args.huber_c
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer_lora_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none = args.set_grad_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    # create pipeline
                    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                    pipeline = FluxPipeline.from_pretrained(
                        args.pretrained_teacher_model,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        transformer=accelerator.unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )

                    validation_prompts = [
                        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
                        # "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                        # "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                        # "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
                    ]
                    for prompt in validation_prompts:
                        pipeline_args = {
                            "prompt": prompt,
                            "num_inference_steps": args.val_infer_step,
                            "guidance_scale": 2,
                        }
                        with torch.no_grad():
                            images = log_validation(
                                pipeline=pipeline,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args=pipeline_args,
                                step=global_step,
                            )
                    del text_encoder_one, text_encoder_two
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_teacher_model,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=global_step,
                is_final_validation=True,
            )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
