# coding=utf-8

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright 2023-present the HuggingFace Inc. team.
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
# limitations under the License.

from .config import PeftType


def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()
    if model.peft_config.peft_type == PeftType.LORA:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = model.peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

    elif model.peft_config.peft_type == PeftType.BOTTLENECK:
        # return the state dict of the model with Bottleneck adapters
        bias = model.peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "adapter_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "adapter_" in k or "bias" in k}
        elif bias == "adapter_only":
            to_return = {}
            for k in state_dict:
                if "adapter_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("adapter_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    elif model.peft_config.peft_type == PeftType.MAMBA_PEFT:
        to_return = {}
        keys = [k for k, _ in state_dict.items()]
        for key in keys:
            if 'head' in key:
                to_return[key] = state_dict[key]
            elif 'adaptf' in key:
                to_return[key] = state_dict[key]
            elif 'lora_out_proj' in key:
                to_return[key] = state_dict[key]
            elif 'lora_in_proj' in key:
                to_return[key] = state_dict[key]
            elif 'lora_X' in key:
                to_return[key] = state_dict[key]
            elif 'lora_Z' in key:
                to_return[key] = state_dict[key]
            elif 'lora_x_proj' in key:
                to_return[key] = state_dict[key]
            elif 'lora_d' in key:
                to_return[key] = state_dict[key]
            elif 'lora_B' in key:
                to_return[key] = state_dict[key]
            elif 'lora_C' in key:
                to_return[key] = state_dict[key]
            elif 'lora_dt' in key:
                to_return[key] = state_dict[key]
            elif 'lora_conv1d' in key:
                to_return[key] = state_dict[key]
            elif 'lora_patch_embed' in key:
                to_return[key] = state_dict[key]
            elif 'fulra_' in key:
                to_return[key] = state_dict[key]
            elif 'prefix_encoder' in key:
                to_return[key] = state_dict[key]
            elif 'prompt_encoder' in key:
                to_return[key] = state_dict[key]
            elif ("A_log" in key.split(".") or "A_b_log" in key.split(".")) and model.peft_config.learnable_A and (not model.peft_config.learnable_A_v2):
                to_return[key] = state_dict[key]
            elif "learnable_A" in key and model.peft_config.learnable_A and model.peft_config.learnable_A_v2:
                to_return[key] = state_dict[key]
            elif ("D" in key.split(".") or "D_b" in key.split(".")) and model.peft_config.learnable_D and (not model.peft_config.learnable_D_v2):
                to_return[key] = state_dict[key]
            elif "learnable_D" in key and model.peft_config.learnable_D and model.peft_config.learnable_D_v2:
                to_return[key] = state_dict[key]
            elif ("conv1d" in key.split(".") or "conv1d_b" in key.split(".")) and model.peft_config.learnable_conv1d and (not model.peft_config.learnable_conv1d_v2):
                to_return[key] = state_dict[key]
            elif "learnable_conv1d" in key and model.peft_config.learnable_conv1d and model.peft_config.learnable_conv1d_v2:
                to_return[key] = state_dict[key]
            elif "cls_token" in key and model.peft_config.learnable_cls_token and (not model.peft_config.learnable_cls_token_v2):
                to_return[key] = state_dict[key]
            elif "learnable_cls_token" in key and model.peft_config.learnable_cls_token and model.peft_config.learnable_cls_token_v2:
                to_return[key] = state_dict[key]
            elif "pos_embed" in key and model.peft_config.learnable_pos_embed and (not model.peft_config.learnable_pos_embed_v2):
                to_return[key] = state_dict[key]
            elif "learnable_pos_embed" in key and model.peft_config.learnable_pos_embed and model.peft_config.learnable_pos_embed_v2:
                to_return[key] = state_dict[key]
            elif "bias" in key and "dt_proj" in key and model.peft_config.learnable_bias and (not model.peft_config.learnable_bias_v2):
                to_return[key] = state_dict[key]
            elif "learnable_bias" in key and model.peft_config.learnable_bias and model.peft_config.learnable_bias_v2:
                to_return[key] = state_dict[key]
            elif '_scan_addi' in key:
                to_return[key] = state_dict[key]

    else:
        to_return = {}
        if model.peft_config.inference_mode:
            prompt_embeddings = model.prompt_encoder.embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save()
        to_return["prompt_embeddings"] = prompt_embeddings
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """

    model.load_state_dict(peft_model_state_dict, strict=False)
    if model.peft_config.peft_type != PeftType.LORA and model.peft_config.peft_type != PeftType.BOTTLENECK and model.peft_config.peft_type != PeftType.MAMBA_PEFT:
        model.prompt_encoder.embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return model
