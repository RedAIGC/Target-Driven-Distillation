import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from typing import Union, Optional, Dict, Any, Tuple
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers


from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import CLIPTextModel,T5EncoderModel
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modified_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    image_rotary_emb = self.pos_embed(ids)
    output_features = []

    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        output_features.append(hidden_states)
        # return output_features
    # print(len(output_features))
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        output_features.append(hidden_states)
    # print(len(output_features))

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    return output_features

class TransformerBasedDiscriminatorHead(nn.Module):
    def __init__(self, input_dim=64, output_channel=1):
        super().__init__()
        self.proj_out = nn.Linear(3072, input_dim, bias=True)  # 将输入投影到 input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),  # 可选
            nn.GELU(),
            nn.Linear(input_dim, output_channel)  # 输出通道
        )
        self.layer_norm = nn.LayerNorm(input_dim)  # 残差连接后的 LayerNorm

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim] -> [1, 4096, 3072]
        
        # Step 1: 投影输入维度
        x_proj = self.proj_out(x)  # [batch_size, seq_len, input_dim]
        
        # Step 2: 平均池化（reduce over seq_len）
        x_reduced = x_proj.mean(dim=-2)  # [batch_size, input_dim]

        # Step 3: 残差连接
        residual = x_reduced  # 保留 Residual 路径
        x_out = self.mlp(x_reduced)  # 经过 MLP
        
        # Step 4: 添加残差，并归一化
        x_out = self.layer_norm(x_out + residual)  # 残差连接 + LayerNorm

        return x_out
        
class TransformerFluxDiscriminator(nn.Module):
    def __init__(self, transformer, head_num=57,num_heads_per_block=1):
        super().__init__()
        self.transformer=transformer
        self.num_heads_per_block = num_heads_per_block
        self.head_num=head_num
        # transformer_output_dims=[transformer_output_dim]*head_num
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBasedDiscriminatorHead()
                        for _ in range(num_heads_per_block)
                    ]
                )
                for _ in range(self.head_num)
            ]
        )

    def _forward(self, packed_noisy_model_input, timesteps,guidance,pooled_prompt_embeds,prompt_embeds,text_ids,latent_image_ids):
        """
        Args:
            features (list of tensors): 
                Features from `transformer_blocks` and `single_transformer_blocks`.
                Example: [19 * torch.Size([1, 4096, 3072]), 38 * torch.Size([1, 4608, 3072])]
        """
        features = modified_forward(
            self.transformer,
            hidden_states=packed_noisy_model_input,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )
        # features=[features[18],features[37],features[56]] 
        assert self.head_num == len(features)
        outputs = []
        for feature_set, head_set in zip(features, self.heads):
            for feature, head in zip(feature_set, head_set):
                outputs.append(head(feature))
        return outputs
    
    def forward(self, flag, *args):
        if flag == "d_loss":
            return self.d_loss(*args)
        elif flag == "g_loss":
            return self.g_loss(*args)
        else:
            assert 0, "not supported"

    def d_loss(
        self,
        sample_fake,
        sample_real,
        timesteps,
        guidance,
        pooled_prompt_embeds,
        prompt_embeds,
        text_ids,
        latent_image_ids,
        weight=1,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            # sample_fake.detach(), timestep, encoder_hidden_states, added_cond_kwargs
            sample_fake.detach(), timesteps,guidance,pooled_prompt_embeds,prompt_embeds,text_ids,latent_image_ids
        )
        real_outputs = self._forward(
            # sample_real.detach(), timestep, encoder_hidden_states, added_cond_kwargs
            sample_real.detach(), timesteps,guidance,pooled_prompt_embeds,prompt_embeds,text_ids,latent_image_ids
        )
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (
                torch.mean(weight * torch.relu(fake_output.float() + 1))
                + torch.mean(weight * torch.relu(1 - real_output.float()))
            ) / (self.head_num * self.num_heads_per_block)
        return loss

    def g_loss(
        self, 
        sample_fake,
        timesteps,
        guidance,
        pooled_prompt_embeds,
        prompt_embeds,
        text_ids,
        latent_image_ids,
        weight=1,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake, timesteps,guidance,pooled_prompt_embeds,prompt_embeds,text_ids,latent_image_ids
        )
        for fake_output in fake_outputs:
            loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
                self.head_num * self.num_heads_per_block
            )
        return loss
    
