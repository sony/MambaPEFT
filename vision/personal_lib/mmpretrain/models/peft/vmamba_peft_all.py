# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List

import torch
from mmengine.logging import print_log
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS

import torch
from torch import nn
import math
from typing import Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys, os
from einops import rearrange, repeat
import math

from personal_lib.mmpretrain.models.backbones.vmamba.vmamba import VSSBlock, Backbone_VSSM, SS2D, Linear2d, selective_scan_fn, cross_merge_fn, cross_scan_fn, checkpoint


def forward_corev2_peft(
    self,
    x: torch.Tensor=None, 
    # ==============================
    force_fp32=False, # True: input fp32
    # ==============================
    ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
    no_einsum=False, # replace einsum with linear or conv1d to raise throughput
    # ==============================
    selective_scan_backend = None,
    # ==============================
    scan_mode = "cross2d",
    scan_force_torch = False,
    # ==============================
    **peft_kwargs
):
    assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
    _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
    assert isinstance(_scan_mode, int)
    delta_softplus = True
    out_norm = self.out_norm
    channel_first = self.channel_first
    to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

    B, D, H, W = x.shape
    N = self.d_state
    K, D, R = self.k_group, self.d_inner, self.dt_rank
    L = H * W


    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
    

    x_proj_weight = self.x_proj_weight
    if self.lora_x_proj is not None:
        x_proj_weight = x_proj_weight + self.s_x_proj * self.lora_x_proj.adapter_up_weight @ self.lora_x_proj.adapter_down_weight
    if self.lora_d is not None:
        x_proj_weight = x_proj_weight + torch.cat([self.s_d * self.lora_d.adapter_up_weight @ self.lora_d.adapter_down_weight, torch.zeros((K, self.d_state*2, x_proj_weight.shape[-1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=1)
    if self.lora_B is not None:
        x_proj_weight = x_proj_weight + torch.cat([torch.zeros((K, self.dt_rank, x_proj_weight.shape[-1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_B * self.lora_B.adapter_up_weight @ self.lora_B.adapter_down_weight, torch.zeros((K, self.d_state, x_proj_weight.shape[-1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=1)
    if self.lora_C is not None:
        x_proj_weight = x_proj_weight + torch.cat([torch.zeros((K, self.dt_rank+self.d_state, x_proj_weight.shape[-1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_C * self.lora_C.adapter_up_weight @ self.lora_C.adapter_down_weight], dim=1)

    if self.scan_addition_num > 0:
        if self.scan_addition_pos=="suffix":
            x_proj_weight = torch.cat([x_proj_weight[:, :self.dt_rank + self.d_state], self.x_proj_scan_addi[:, :self.scan_addition_num], x_proj_weight[:, -self.d_state:], self.x_proj_scan_addi[:, self.scan_addition_num:]], dim=1).contiguous()
        else:
            x_proj_weight = torch.cat([x_proj_weight[:, :self.dt_rank], self.x_proj_scan_addi[:, :self.scan_addition_num], x_proj_weight[:, self.dt_rank: self.dt_rank + self.d_state], self.x_proj_scan_addi[:, self.scan_addition_num:], x_proj_weight[:, -self.d_state:]], dim=1).contiguous()
    else:
        x_proj_weight = x_proj_weight
    
    dt_proj_weight = self.dt_projs_weight
    dt_proj_bias = self.dt_projs_bias.float()
    if self.lora_dt is not None:
        dt_proj_weight = dt_proj_weight + self.s_dt * self.lora_dt.adapter_up_weight @ self.lora_dt.adapter_down_weight
    
    if self.learnable_bias is not None:
        dt_proj_bias = dt_proj_bias + self.learnable_bias

    if _scan_mode == -1:
        raise NotImplementedError
    else:
        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        if peft_kwargs.get("inner_single_prefix") is not None: # 先頭にprefix (f,b共に同じprefix)
            single_prefix = peft_kwargs.get("inner_single_prefix").permute(0,2,1)
            xs = torch.cat([single_prefix[:,None,:,:], xs], dim=3)
            vertual_token_num = single_prefix.shape[2]
        elif peft_kwargs.get("inner_dual_prefix") is not None: # 先頭にprefix (f,b別々のprefix)
            dual_prefix = peft_kwargs.get("inner_dual_prefix").permute(0,2,1)
            vertual_token_num = dual_prefix.shape[2]
            xs = torch.cat([dual_prefix.view(B,K,-1,vertual_token_num), xs], dim=3)
        elif peft_kwargs.get("inner_single_suffix") is not None: # 先頭にsuffix (f,b共に同じprefix)
            single_prefix = peft_kwargs.get("inner_single_suffix").permute(0,2,1)
            xs = torch.cat([xs, single_prefix[:,None,:,:]], dim=3)
            vertual_token_num = single_prefix.shape[2]
        else:
            vertual_token_num = 0
            xs = xs
    
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N+self.scan_addition_num, N+self.scan_addition_num], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_proj_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N+self.scan_addition_num, N+self.scan_addition_num], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_proj_weight)

        xs = xs.view(B, -1, L+vertual_token_num)
        dts = dts.contiguous().view(B, -1, L+vertual_token_num)

        if self.learnable_A_logs is not None:
            As = -torch.exp((self.A_logs+self.learnable_A_logs).float())  # (k * c, d_state)
        else:
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)

        if self.scan_addition_num > 0:
            A_addi = -torch.exp(self.A_log_scan_addi.float())
            
            if self.scan_addition_pos=="suffix":
                As = torch.cat([As, A_addi], dim=1)
            else:
                As = torch.cat([A_addi, As], dim=1)

        Ds = self.Ds.to(torch.float) if self.learnable_D is None else (self.Ds+self.learnable_D).float() # (K * c)
        Bs = Bs.contiguous().view(B, K, N+self.scan_addition_num, L+vertual_token_num)
        Cs = Cs.contiguous().view(B, K, N+self.scan_addition_num, L+vertual_token_num)
        delta_bias = dt_proj_bias.view(-1)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        out: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        )
        
        if "inner_single_prefix" in peft_kwargs.keys() or "inner_dual_prefix" in peft_kwargs.keys():
            out = out [..., vertual_token_num:]
        elif "inner_single_suffix" in peft_kwargs.keys():
            out = out [..., :-vertual_token_num]
        
        ys = out.view(B, K, -1, H, W)

        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y, H=H, W=W,
            ))

    y = y.view(B, -1, H, W)
    if not channel_first:
        y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
    y = out_norm(y)

    return y.to(x.dtype)


def forwardv2_peft(self, x: torch.Tensor, **peft_kwargs):
    x_ = x
    x = self.in_proj(x_)
    if self.lora_in_proj is not None:
        x = x + self.s_in_proj*self.lora_in_proj(x_)
    
    if not self.disable_z:
        if self.lora_X is not None:
            x[..., :x.shape[-1]//2] = x[..., :x.shape[-1]//2] + self.s_X*self.lora_X(x_)
        if self.lora_Z is not None:
            x[..., x.shape[-1]//2:] = x[..., x.shape[-1]//2:] + self.s_Z*self.lora_Z(x_)

    if not self.disable_z:
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
        if not self.disable_z_act:
            z = self.act(z)
    if not self.channel_first:
        x = x.permute(0, 3, 1, 2).contiguous()
    if self.with_dconv:
        if self.learnable_conv1d_weight is not None:
            conv1d_weight = self.conv2d.weight + self.learnable_conv1d_weight
        else:
            conv1d_weight = self.conv2d.weight
        x = F.conv2d(x, conv1d_weight, bias=self.conv2d.bias, padding=self.conv2d.padding, groups=self.conv2d.groups) # (b, d, h, w)

    x = self.act(x)
    y = self.forward_core(x, **peft_kwargs)
    y = self.out_act(y)
    if not self.disable_z:
        y = y * z
    
    if self.lora_out_proj is None:
        out = self.out_proj(y)
    else:
        out = self.out_proj(y) + self.lora_out_proj(y)*self.s
    
    if self.adaptf is not None:
        out = out + self.s_adaptf*self.adaptf(y)

    return self.dropout(out)

def vssblock_forward(self, input: torch.Tensor, **peft_kwargs):
    x = input
    if self.ssm_branch:
        if self.post_norm:
            x = x + self.drop_path(self.norm(self.op(x), **peft_kwargs))
        else:
            x = x + self.drop_path(self.op(self.norm(x), **peft_kwargs))
    if self.mlp_branch:
        if self.post_norm:
            x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
    return x

def vssblock__forward(self, input: torch.Tensor, **peft_kwargs):
    if self.use_checkpoint:
        return checkpoint.checkpoint(self._forward, input, **peft_kwargs)
    else:
        return self._forward(input, **peft_kwargs)


def backbone_vssm_forward_peft(self, x):
    def layer_forward(l, x):
        x = l.blocks(x)
        y = l.downsample(x)
        return x, y

    x = self.patch_embed(x)

    if self.prefix_encoder is not None:
        prefixes = self.prefix_encoder(x.shape[0])

    p_ind = 0
    outs = []
    for i, layer in enumerate(self.layers):
        if self.prefix_encoder is not None:
            for block in layer.blocks:
                peft_kwargs={}
                peft_kwargs[self.prefix_type] = prefixes[...,p_ind:p_ind+self.prefix_ch_per_layer[i]]
                p_ind += self.prefix_ch_per_layer[i]
                x = block(x, **peft_kwargs)
            o = x
            x = layer.downsample(x)
        else:
            o, x = layer_forward(layer, x) # (B, H, W, C)
        
        if not self.use_pretrained_last_norm:
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

    if self.use_pretrained_last_norm:
        x = self.classifier(x)
        if not self.channel_first:
            out = out.permute(0, 3, 1, 2)
        outs.append(x)

    if len(self.out_indices) == 0:
        return x
    
    return outs



class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32, use_act=False, group=1, channel_first=True):
        super().__init__()

        if group==1:
            self.adapter_down = Linear2d(in_channels, dim, bias=False)
            self.adapter_up = Linear2d(dim, out_channels, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            adapter_down = [
                nn.Linear(in_channels, dim, bias=False)
                for _ in range(group)
            ]
            self.adapter_down_weight = nn.Parameter(torch.stack([t.weight for t in adapter_down], dim=0))
            adapter_up = [
                nn.Linear(dim, out_channels, bias=False)
                for _ in range(group)
            ]
            self.adapter_up_weight = nn.Parameter(torch.stack([t.weight for t in adapter_up], dim=0))
            nn.init.zeros_(self.adapter_up_weight.data)

        if use_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.group = group

    def forward(self, x):
        assert self.group==1
       
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up

class Conv2dAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32, kernel_size=(3, 3), stride=(1,1), padding=(1,1)):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Conv2d(in_channels, dim, kernel_size, stride, padding, bias=False)
            self.adapter_up = nn.Conv2d(dim, out_channels, (1, 1), (1, 1), bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            raise NotImplementedError
            # self.adapter_down = QLinear(768, dim, bit)
            # self.adapter_up = QLinear(dim, 768, bit)
            # nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        x_down = self.adapter_down(x)  
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        return x_up

# Based on https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning/model.py
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, num_per_layer*layers*hidden)
    '''
    def __init__(self, prefix_projection=True, token_dims=[768], num_virtual_tokens=20, encoder_hidden_size=768, num_layers=[12], num_per_layer=2):
        super().__init__()
        self.prefix_projection = prefix_projection
        self.prompt_tokens = torch.arange(
            num_virtual_tokens
        ).long()

        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, int(sum(token_dims)/len(token_dims)))
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(int(sum(token_dims)/len(token_dims)), encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, sum([num_layers[i] * num_per_layer * token_dims[i] for i in range(len(token_dims))])),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, sum([num_layers[i] * num_per_layer * token_dims[i] for i in range(len(token_dims))]))

    def forward(self, batch_size):
        prefix = (
            self.prompt_tokens
            .unsqueeze(0)
            .to(self.embedding.weight.device)
        )

        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        past_key_values = past_key_values.expand(batch_size, -1, -1)
        return past_key_values
    

def set_peft(model, 
             # Adaptformer
             adaptformer=False, dim_adaptf=32, s_adaptf=1, bit_adaptf=32, 
             # LoRA config out_proj (AdaptFormer)
             lora_out_proj=False, dim=32, s=1, bit=32, 
             # LoRA config in_proj (X and Z),
             lora_in_proj=False, dim_in_proj=32, s_in_proj=1, bit_in_proj=32,
             # LoRA config X,
             lora_X=False, dim_X=32, s_X=1, bit_X=32,
             # LoRA confg Z
             lora_Z=False, dim_Z=32, s_Z=1, bit_Z=32, 
             # LoRA config x_proj (d, B, and C),
             lora_x_proj=False, dim_x_proj=4, s_x_proj=1, bit_x_proj=32,
             # LoRA config d,
             lora_d=False, dim_d=4, s_d=1, bit_d=32,
             # LoRA config B,
             lora_B=False, dim_B=4, s_B=1, bit_B=32,
             # LoRA config C,
             lora_C=False, dim_C=4, s_C=1, bit_C=32,
             # LoRA config dt,
             lora_dt=False, dim_dt=4, s_dt=1, bit_dt=32,
             # LoRA config patch_embed Conv2d,
             lora_patch_embed=False, dim_patch_embed=32, s_patch_embed=1, bit_patch_embed=32,
             # prefix tuning config
             prefix_tuning=False, prefix_type="inner_single_prefix", prefix_projection=True, num_virtual_tokens=1, encoder_hidden_size=None,
             # prompt tuning config
             prompt_tuning=False, prompt_type="prefix", prompt_projection=True, prompt_num_tokens=2,
             # MambaならではのPEFTを考えてみた？Scanする数（state_d）を増やすという方向性
             additional_scan=False, scan_addition_num=1, scan_addition_pos="suffix", scan_A_constant=None, scan_A_copy_from_last=False, zero_init_x_proj=False,
             # Aのfinetuning
             learnable_A=False, learnable_A_v2=False,
             # Dのfinetuning
             learnable_D=False, learnable_D_v2=False,
             # conv1dのfinetuning
             learnable_conv1d=False, learnable_conv1d_v2=False,
             # cls_tokenのfinetuning
             learnable_cls_token=False, learnable_cls_token_v2=False,
             # pos_embedのfinetuning
             learnable_pos_embed=False, learnable_pos_embed_v2=False,
             # biasのfinetuning
             learnable_bias=False, learnable_bias_v2=False,
             ):
    
    if type(model) == Backbone_VSSM:
        if prefix_tuning:
            num_layers = model.depths
            token_dims = model.dims
            if prefix_type in ["inner_single_prefix", "inner_single_suffix"]:
                num_per_layer = 1
            elif prefix_type in ["inner_dual_prefix"]:
                num_per_layer = model.layers[0][0][0].op.k_group
            else:
                raise NotImplementedError
            
            if prefix_type in ["inner_single_prefix", "inner_dual_prefix", "inner_single_suffix"]:
                token_dims = [int(token_dims[i] * model.ssm_ratio) for i in range(len(token_dims))]
            encoder_hidden_size_ = encoder_hidden_size if encoder_hidden_size is not None else max(token_dims)
            model.prefix_ch_per_layer = [int(token_dims[i] * num_per_layer) for i in range(len(token_dims))]
            model.prefix_encoder = PrefixEncoder(prefix_projection, token_dims, num_virtual_tokens, encoder_hidden_size_, num_layers, num_per_layer)
            model.prefix_type = prefix_type        
        else:
            model.prefix_encoder = None

        # if prompt_tuning:
        #     token_dim = model.dims[0]
        #     model.prompt_encoder = PrefixEncoder(prompt_projection, [token_dim], prompt_num_tokens, token_dim, [1], 1)
        #     model.prompt_type = prompt_type
        # else:
        #     model.prompt_encoder = None
        
        if learnable_cls_token and learnable_cls_token_v2:
            model.learnable_cls_token= nn.Parameter(torch.zeros_like(model.cls_token.data))
        else:
            model.learnable_cls_token = None
        
        if learnable_pos_embed and learnable_pos_embed_v2:
            model.learnable_pos_embed= nn.Parameter(torch.zeros_like(model.pos_embed.data))
        else:
            model.learnable_pos_embed = None

        bound_method = backbone_vssm_forward_peft.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method)

    for layer in model.children():
        if type(layer) == SS2D:
            # print(f"add lora to {layer}")

            if adaptformer:
                in_ch, out_ch = layer.d_inner, layer.d_model
                layer.adaptf = Adapter(in_ch, out_ch, dim_adaptf, bit_adaptf, use_act=True, channel_first=layer.channel_first)
                layer.s_adaptf = s_adaptf
            else:
                layer.adaptf = None

            if lora_out_proj:
                in_ch, out_ch = layer.d_inner, layer.d_model
                layer.lora_out_proj = Adapter(in_ch, out_ch, dim, bit, channel_first=layer.channel_first)
                layer.s = s
            else:
                layer.lora_out_proj = None
            
            if lora_in_proj:
                in_ch, out_ch = layer.d_model, layer.d_inner
                layer.lora_in_proj = Adapter(in_ch, out_ch, dim_in_proj, bit_in_proj, channel_first=layer.channel_first)
                layer.s_in_proj = s_in_proj
            else:
                layer.lora_in_proj = None
            
            if lora_X:
                in_ch, out_ch = layer.d_model, layer.d_inner
                layer.lora_X = Adapter(in_ch, out_ch, dim_X, bit_X, channel_first=layer.channel_first)
                layer.s_X = s_X
            else:
                layer.lora_X = None
            
            if lora_Z:
                in_ch, out_ch = layer.d_model, layer.d_inner
                layer.lora_Z = Adapter(in_ch, out_ch, dim_Z, bit_Z, channel_first=layer.channel_first)
                layer.s_Z = s_Z
            else:
                layer.lora_Z = None
            
            if lora_x_proj:
                in_ch, out_ch = layer.d_inner, (layer.dt_rank+layer.d_state*2)
                layer.lora_x_proj = Adapter(in_ch, out_ch, dim_x_proj, bit_x_proj, group=layer.k_group, channel_first=layer.channel_first)
                layer.s_x_proj = s_x_proj
            else:
                layer.lora_x_proj = None
            
            if lora_d:
                in_ch, out_ch = layer.d_inner, layer.dt_rank
                layer.lora_d = Adapter(in_ch, out_ch, dim_d, bit_d, group=layer.k_group, channel_first=layer.channel_first)
                layer.s_d = s_d
            else:
                layer.lora_d = None
            
            if lora_B:
                in_ch, out_ch = layer.d_inner, layer.d_state
                layer.lora_B = Adapter(in_ch, out_ch, dim_B, bit_B, group=layer.k_group, channel_first=layer.channel_first)
                layer.s_B = s_B
            else:
                layer.lora_B = None
            
            if lora_C:
                in_ch, out_ch = layer.d_inner, layer.d_state
                layer.lora_C = Adapter(in_ch, out_ch, dim_C, bit_C, group=layer.k_group, channel_first=layer.channel_first)
                
                layer.s_C = s_C
            else:
                layer.lora_C = None
            
            if lora_dt:
                in_ch, out_ch = layer.dt_rank, layer.d_inner
                layer.lora_dt = Adapter(in_ch, out_ch, dim_dt, bit_dt, group=layer.k_group, channel_first=layer.channel_first)
                layer.s_dt = s_dt
            else:
                layer.lora_dt = None
            
            if learnable_conv1d and learnable_conv1d_v2:
                device = layer.conv2d.weight.device
                layer.learnable_conv1d_weight = nn.Parameter(torch.zeros_like(layer.conv2d.weight, device=device))
            else:
                layer.learnable_conv1d_weight = None
                
            if additional_scan:
                d_state = layer.d_state
                d_inner = layer.d_inner
                device = layer.A_logs.device
                dtype= layer.A_logs.dtype
                
                if scan_A_copy_from_last:
                    if scan_addition_pos == "suffix":
                        A_scan_addi_log= repeat(layer.A_logs.data.view(layer.k_group, layer.d_inner, layer.d_state)[:, :, -1], "k d -> k d n",
                            n=scan_addition_num,
                        ).reshape(layer.k_group*layer.d_inner,scan_addition_num).contiguous()
                    else:
                        A_scan_addi_log= repeat(layer.A_logs.data.view(layer.k_group, layer.d_inner, layer.d_state)[:, :, 0], "k d -> k d n",
                            n=scan_addition_num,
                        ).reshape(layer.k_group*layer.d_inner,scan_addition_num).contiguous()
                elif scan_A_constant is None:
                    A_scan_addi = repeat(
                        torch.arange(d_state+1, d_state+1+scan_addition_num, dtype=dtype, device=device),
                        "n -> k d n",
                        k=layer.k_groups,
                        d=d_inner,
                    ).reshape(layer.k_group*layer.d_inner,scan_addition_num).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_logs in fp32   
                else:
                    A_scan_addi = repeat(
                        scan_A_constant*torch.ones((scan_addition_num,), dtype=dtype, device=device),
                        "n -> k d n",
                        k=layer.k_groups,
                        d=d_inner,
                    ).reshape(layer.k_group*layer.d_inner,scan_addition_num).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_logs in fp32      
                layer.A_log_scan_addi = nn.Parameter(A_scan_addi_log)
                layer.A_log_scan_addi._no_weight_decay = True

                x_proj_scan_addi = [
                    nn.Linear(d_inner, scan_addition_num * 2, bias=False, dtype=dtype, device=device)
                    for _ in range(layer.k_group)
                ]
                layer.x_proj_scan_addi = nn.Parameter(torch.stack([t.weight for t in x_proj_scan_addi], dim=0))

                layer.scan_addition_num=scan_addition_num
                layer.scan_addition_pos=scan_addition_pos

                if zero_init_x_proj:
                    nn.init.zeros_(layer.x_proj_scan_addi.data)
            else:
                layer.scan_addition_num=0
            
            if learnable_A and learnable_A_v2:
                device = layer.A_logs.data.device
                layer.learnable_A_logs = nn.Parameter(torch.zeros_like(layer.A_logs.data, device=device))
            else:
                layer.learnable_A_logs = None

            if learnable_D and learnable_D_v2:
                device = layer.Ds.data.device
                layer.learnable_D = nn.Parameter(torch.zeros_like(layer.Ds.data, device=device))
            else:
                layer.learnable_D = None

            if learnable_bias and learnable_bias_v2:
                device = layer.dt_projs_bias.device
                layer.learnable_bias = nn.Parameter(torch.zeros_like(layer.dt_projs_bias, device=device))
            else:
                layer.learnable_bias = None


            bound_method = forward_corev2_peft.__get__(layer, layer.__class__)
            setattr(layer, 'forward_core', bound_method)
            bound_method = forwardv2_peft.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == VSSBlock:
            bound_method = vssblock_forward.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            bound_method = vssblock__forward.__get__(layer, layer.__class__)
            setattr(layer, '_forward', bound_method)
        # elif type(layer) == PatchEmbed:
        #     if lora_patch_embed:
        #         layer.s_patch_embed=s_patch_embed
        #         layer.lora_patch_embed = Conv2dAdapter(layer.proj.in_channels, layer.proj.out_channels, dim_patch_embed, bit_patch_embed, layer.proj.kernel_size, layer.proj.stride, layer.proj.padding)
        #         bound_method = forward_patch_embed.__get__(layer, layer.__class__)
        #         setattr(layer, 'forward', bound_method)

        
        if len(list(layer.children())) != 0:
            set_peft(layer,
                     # Adaptformer
                     adaptformer, dim_adaptf, s_adaptf, bit_adaptf,
                     lora_out_proj, dim, s, bit,
                     lora_in_proj, dim_in_proj, s_in_proj, bit_in_proj,
                     # LoRA config X,
                     lora_X, dim_X, s_X, bit_X,
                     # LoRA confg Z
                     lora_Z, dim_Z, s_Z, bit_Z, 
                     # LoRA config x_proj (d, B, and C),
                     lora_x_proj, dim_x_proj, s_x_proj, bit_x_proj,
                     # LoRA config d,
                     lora_d, dim_d, s_d, bit_d,
                     # LoRA config B,
                     lora_B, dim_B, s_B, bit_B,
                     # LoRA config C,
                     lora_C, dim_C, s_C, bit_C,
                     # LoRA config dt,
                     lora_dt, dim_dt, s_dt, bit_dt,
                     # LoRA config patch_embed Conv2d,
                     lora_patch_embed, dim_patch_embed, s_patch_embed, bit_patch_embed,
                     # Prefix tuning
                     prefix_tuning, prefix_type, prefix_projection, num_virtual_tokens, encoder_hidden_size,
                     # prompt tuning config
                     prompt_tuning, prompt_type, prompt_projection, prompt_num_tokens,
                     # Additional Scan
                     additional_scan, scan_addition_num, scan_addition_pos, scan_A_constant, scan_A_copy_from_last, zero_init_x_proj,
                     # Aのfinetuning
                     learnable_A, learnable_A_v2,
                     # Dのfinetuning
                     learnable_D, learnable_D_v2,
                     # conv1dのfinetuning
                     learnable_conv1d, learnable_conv1d_v2,
                     # cls_tokenのfinetuning
                     learnable_cls_token, learnable_cls_token_v2,
                     # pos_embedのfinetuning
                     learnable_pos_embed, learnable_pos_embed_v2,
                     # biasのfinetuning
                     learnable_bias, learnable_bias_v2,)



@MODELS.register_module()
class VSSMPEFTModel(BaseModule):
    """

    Examples:
        >>> model = LoRAModel(
        ...     module=dict(type='VisionTransformer', arch='b'),
        ...     alpha=4,
        ...     rank=4,
        ...     drop_rate=0.1,
        ...     targets=[
        ...         dict(type='.*qkv'), # regular expression
        ...         dict(type='proj', alpha=8, rank=8), # suffix
        ...     ])
    """

    def __init__(self,
                 module: dict,
                 # Adaptformer
                 adaptformer=False, dim_adaptf=32, s_adaptf=1, bit_adaptf=32, train_adaptformer=True, 
                 # LoRA config out_proj (AdaptFormer)
                lora_out_proj=False, dim=32, s=1, bit=32, train_lora_out_proj=True,
                # LoRA config in_proj (X and Z),
                lora_in_proj=False, dim_in_proj=32, s_in_proj=1, bit_in_proj=32, train_lora_in_proj=True,
                # LoRA config X,
                lora_X=False, dim_X=32, s_X=1, bit_X=32, train_lora_X=True,
                # LoRA confg Z
                lora_Z=False, dim_Z=32, s_Z=1, bit_Z=32, train_lora_Z=True,
                # LoRA config x_proj (d, B, and C),
                lora_x_proj=False, dim_x_proj=4, s_x_proj=1, bit_x_proj=32, train_lora_x_proj=True,
                # LoRA config d,
                lora_d=False, dim_d=4, s_d=1, bit_d=32, train_lora_d=True,
                # LoRA config B,
                lora_B=False, dim_B=4, s_B=1, bit_B=32, train_lora_B=True,
                # LoRA config C,
                lora_C=False, dim_C=4, s_C=1, bit_C=32, train_lora_C=True,
                # LoRA config dt,
                lora_dt=False, dim_dt=4, s_dt=1, bit_dt=32, train_lora_dt=True,
                # LoRA config patch_embed Conv2d,
                lora_patch_embed=False, dim_patch_embed=32, s_patch_embed=1, bit_patch_embed=32, train_lora_patch_embed=True,
                # prefix tuning config
                prefix_tuning=False, prefix_type="inner_single_prefix", prefix_projection=True, num_virtual_tokens=1, encoder_hidden_size=None, train_prefix=True,
                # prompt tuning config
                prompt_tuning=False, prompt_type="prefix", prompt_projection=True, prompt_num_tokens=2, train_prompt=True,
                # MambaならではのPEFTを考えてみた？Scanする数（state_d）を増やすという方向性
                additional_scan=False, scan_addition_num=1, scan_addition_pos="suffix", scan_A_constant=None, scan_A_copy_from_last=False, zero_init_x_proj=False, train_additional_scan=True,
                # Bias tuning like
                learnable_A=False, learnable_A_v2=False, train_A=True,
                learnable_D=False, learnable_D_v2=False, train_D=True,
                learnable_conv1d=False, learnable_conv1d_v2=False, train_conv1d=True,
                learnable_cls_token=False, learnable_cls_token_v2=False, train_cls_token=True,
                learnable_pos_embed=False, learnable_pos_embed_v2=False, train_pos_embed=True,
                learnable_bias=False, learnable_bias_v2=False, train_bias=True,
                 ):

        super().__init__()

        assert not prompt_tuning
        assert not learnable_cls_token
        assert not learnable_pos_embed

        module = MODELS.build(module)
        # module.init_weights()

        self.module = module
        self.adaptformer=adaptformer
        self.train_adaptformer=train_adaptformer
        self.lora_out_proj=lora_out_proj
        self.train_lora_out_proj = train_lora_out_proj
        self.lora_in_proj=lora_in_proj
        self.train_lora_in_proj = train_lora_in_proj
        self.lora_X=lora_X
        self.train_lora_X = train_lora_X
        self.lora_Z=lora_Z
        self.train_lora_Z = train_lora_Z
        self.lora_x_proj=lora_x_proj
        self.train_lora_x_proj = train_lora_x_proj
        self.lora_d=lora_d
        self.train_lora_d = train_lora_d
        self.lora_B=lora_B
        self.train_lora_B = train_lora_B
        self.lora_C=lora_C
        self.train_lora_C = train_lora_C
        self.lora_dt=lora_dt
        self.train_lora_dt = train_lora_dt
        self.lora_patch_embed=lora_patch_embed
        self.train_lora_patch_embed = train_lora_patch_embed
        self.prefix_tuning=prefix_tuning
        self.train_prefix = train_prefix
        self.prompt_tuning=prompt_tuning
        self.train_prompt = train_prompt
        self.additional_scan=additional_scan
        self.train_additional_scan = train_additional_scan
        self.learnable_A=learnable_A
        self.learnable_A_v2=learnable_A_v2
        self.train_A = train_A
        self.learnable_D=learnable_D
        self.learnable_D_v2=learnable_D_v2
        self.train_D = train_D
        self.learnable_conv1d=learnable_conv1d
        self.learnable_conv1d_v2=learnable_conv1d_v2
        self.train_conv1d = train_conv1d
        self.learnable_cls_token=learnable_cls_token
        self.learnable_cls_token_v2=learnable_cls_token_v2
        self.train_cls_token=train_cls_token
        self.learnable_pos_embed=learnable_pos_embed
        self.learnable_pos_embed_v2=learnable_pos_embed_v2
        self.train_pos_embed=train_pos_embed
        self.learnable_bias=learnable_bias
        self.learnable_bias_v2=learnable_bias_v2
        self.train_bias=train_bias
        set_peft(self,
                    # Adaptformer
                    adaptformer, dim_adaptf, s_adaptf, bit_adaptf, 
                    # LoRA config out_proj (AdaptFormer),
                    lora_out_proj, dim, s, bit,
                    # LoRA config in_proj
                    lora_in_proj, dim_in_proj, s_in_proj, bit_in_proj,
                    # LoRA config X,
                    lora_X, dim_X, s_X, bit_X,
                    # LoRA confg Z
                    lora_Z, dim_Z, s_Z, bit_Z, 
                    # LoRA config x_proj (d, B, and C),
                    lora_x_proj, dim_x_proj, s_x_proj, bit_x_proj,
                    # LoRA config d,
                    lora_d, dim_d, s_d, bit_d,
                    # LoRA config B,
                    lora_B, dim_B, s_B, bit_B,
                    # LoRA config C,
                    lora_C, dim_C, s_C, bit_C,
                    # LoRA config dt,
                    lora_dt, dim_dt, s_dt, bit_dt,
                    # LoRA config patch_embed Conv2d,
                    lora_patch_embed, dim_patch_embed, s_patch_embed, bit_patch_embed,
                    # prefix tuning,
                    prefix_tuning, prefix_type, prefix_projection, num_virtual_tokens, encoder_hidden_size,
                    # prompt tuning config
                    prompt_tuning, prompt_type, prompt_projection, prompt_num_tokens,
                    # additional scan config
                    additional_scan, scan_addition_num, scan_addition_pos, scan_A_constant, scan_A_copy_from_last, zero_init_x_proj,
                    # Aのfinetuning
                    learnable_A, learnable_A_v2,
                    # Dのfinetuning
                    learnable_D, learnable_D_v2,
                    # conv1dのfinetuning
                    learnable_conv1d, learnable_conv1d_v2,
                    # cls_tokenのfinetuning
                    learnable_cls_token, learnable_cls_token_v2,
                    # pos_embedのfinetuning
                    learnable_pos_embed, learnable_pos_embed_v2,
                    # biasのfinetuning
                    learnable_bias, learnable_bias_v2)

        self._set_lora_trainable()
        self._register_state_dict_hooks()

    def _set_lora_trainable(self):
        """Set only the lora parameters trainable."""
        for n, p in self.named_parameters():
            if 'head' in n and p.requires_grad:
                p.requires_grad = True
            elif 'adaptf' in n and self.train_adaptformer:
                p.requires_grad = True
            elif 'lora_out_proj' in n and self.train_lora_out_proj:
                p.requires_grad = True
            elif 'lora_in_proj' in n and self.train_lora_in_proj:
                p.requires_grad = True
            elif 'lora_X' in n and self.train_lora_X:
                p.requires_grad = True
            elif 'lora_Z' in n and self.train_lora_Z:
                p.requires_grad = True
            elif 'lora_x_proj' in n and self.train_lora_x_proj:
                p.requires_grad = True
            elif 'lora_d' in n and self.train_lora_d:
                p.requires_grad = True
            elif 'lora_B' in n and self.train_lora_B:
                p.requires_grad = True
            elif 'lora_C' in n and self.train_lora_C:
                p.requires_grad = True
            elif 'lora_dt' in n and self.train_lora_dt:
                p.requires_grad = True
            elif 'lora_patch_embed' in n and self.train_lora_patch_embed:
                p.requires_grad = True
            elif 'prefix_encoder' in n and self.train_prefix:
                p.requires_grad = True
            elif 'prompt_encoder' in n and self.train_prompt:
                p.requires_grad = True
            elif "A_logs" in n.split(".") and self.learnable_A and self.train_A and (not self.learnable_A_v2):
                p.requires_grad = True
            elif "learnable_A" in n and self.learnable_A and self.learnable_A_v2 and self.train_A:
                p.requires_grad = True
            elif "Ds" in n.split(".") and self.learnable_D and self.train_D and (not self.learnable_D_v2):
                p.requires_grad = True
            elif "learnable_D" in n and self.learnable_D and self.learnable_D_v2 and self.train_D:
                p.requires_grad = True
            elif "conv2d" in n.split(".") and self.learnable_conv1d and self.train_conv1d and (not self.learnable_conv1d_v2):
                p.requires_grad = True
            elif "learnable_conv1d" in n and self.learnable_conv1d and self.learnable_conv1d_v2 and self.train_conv1d:
                p.requires_grad = True
            elif "cls_token" in n and self.learnable_cls_token and self.train_cls_token and (not self.learnable_cls_token_v2):
                p.requires_grad = True
            elif "learnable_cls_token" in n and self.learnable_cls_token and self.learnable_cls_token_v2 and self.train_cls_token:
                p.requires_grad = True
            elif "pos_embed" in n and self.learnable_pos_embed and self.train_pos_embed and (not self.learnable_pos_embed_v2):
                p.requires_grad = True
            elif "learnable_pos_embed" in n and self.learnable_pos_embed and self.learnable_pos_embed_v2 and self.train_pos_embed:
                p.requires_grad = True
            elif "bias" in n and "dt_proj" in n and self.learnable_bias and self.train_bias and (not self.learnable_bias_v2):
                p.requires_grad = True
            elif "learnable_bias" in n and self.learnable_bias and self.learnable_bias_v2 and self.train_bias:
                p.requires_grad = True
            elif '_scan_addi' in n and self.train_additional_scan:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_log(f'number of params: {n_parameters}', logger='current')

    def _register_state_dict_hooks(self):
        """Register state dict hooks.

        Register state dict saving hooks to save only the lora parameters to
        the state dict. And register state dict loading hooks to handle the
        incompatible keys while loading the state dict.
        """

        def _state_dict_hook(module, state_dict, prefix, local_metadata):
            """Save only the lora parameters to the state dict."""
            keys = [k for k, _ in state_dict.items()]
            for key in keys:
                if 'head' in key:
                    continue
                elif 'adaptf' in key:
                    continue
                elif 'lora_out_proj' in key:
                    continue
                elif 'lora_in_proj' in key:
                    continue
                elif 'lora_X' in key:
                    continue
                elif 'lora_Z' in key:
                    continue
                elif 'lora_x_proj' in key:
                    continue
                elif 'lora_d' in key:
                    continue
                elif 'lora_B' in key:
                    continue
                elif 'lora_C' in key:
                    continue
                elif 'lora_dt' in key:
                    continue
                elif 'lora_patch_embed' in key:
                    continue
                elif 'prefix_encoder' in key:
                    continue
                elif 'prompt_encoder' in key:
                    continue
                elif "A_logs" in key.split(".") and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif "Ds" in key.split(".") and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif "conv2d" in key.split(".") and self.learnable_conv1d and (not self.learnable_conv1d_v2):
                    continue
                elif "learnable_conv1d" in key and self.learnable_conv1d and self.learnable_conv1d_v2:
                    continue
                elif "cls_token" in key and self.learnable_cls_token and (not self.learnable_cls_token_v2):
                    continue
                elif "learnable_cls_token" in key and self.learnable_cls_token and self.learnable_cls_token_v2:
                    continue
                elif "pos_embed" in key and self.learnable_pos_embed and (not self.learnable_pos_embed_v2):
                    continue
                elif "learnable_pos_embed" in key and self.learnable_pos_embed and self.learnable_pos_embed_v2:
                    continue
                elif "bias" in key and "dt_proj" in key and self.learnable_bias and (not self.learnable_bias_v2):
                    continue
                elif "learnable_bias" in key and self.learnable_bias and self.learnable_bias_v2:
                    continue
                elif '_scan_addi' in key:
                    continue
                else:
                    state_dict.pop(key)

        self._register_state_dict_hook(_state_dict_hook)

        def _load_state_dict_post_hook(module, incompatible_keys):
            """Handle the incompatible keys while loading the state dict."""
            missing_keys = incompatible_keys.missing_keys.copy()
            for key in missing_keys:
                if 'head' in key:
                    continue
                elif 'adaptf' in key:
                    continue
                elif 'lora_out_proj' in key:
                    continue
                elif 'lora_in_proj' in key:
                    continue
                elif 'lora_X' in key:
                    continue
                elif 'lora_Z' in key:
                    continue
                elif 'lora_x_proj' in key:
                    continue
                elif 'lora_d' in key:
                    continue
                elif 'lora_B' in key:
                    continue
                elif 'lora_C' in key:
                    continue
                elif 'lora_dt' in key:
                    continue
                elif 'lora_patch_embed' in key:
                    continue
                elif 'prefix_encoder' in key:
                    continue
                elif 'prompt_encoder' in key:
                    continue
                elif "A_logs" in key.split(".") and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif "Ds" in key.split(".") and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif "conv2d" in key.split(".") and self.learnable_conv1d and (not self.learnable_conv1d_v2):
                    continue
                elif "learnable_conv1d" in key and self.learnable_conv1d and self.learnable_conv1d_v2:
                    continue
                elif "cls_token" in key and self.learnable_cls_token and (not self.learnable_cls_token_v2):
                    continue
                elif "learnable_cls_token" in key and self.learnable_cls_token and self.learnable_cls_token_v2:
                    continue
                elif "pos_embed" in key and self.learnable_pos_embed and (not self.learnable_pos_embed_v2):
                    continue
                elif "learnable_pos_embed" in key and self.learnable_pos_embed and self.learnable_pos_embed_v2:
                    continue
                elif "bias" in key and "dt_proj" in key and self.learnable_bias and (not self.learnable_bias_v2):
                    continue
                elif "learnable_bias" in key and self.learnable_bias and self.learnable_bias_v2:
                    continue
                elif '_scan_addi' in key:
                    continue
                else:
                    incompatible_keys.missing_keys.remove(key)

            unexpected_keys = incompatible_keys.unexpected_keys.copy()
            for key in unexpected_keys:
                if 'head' in key:
                    continue
                elif 'adaptf' in key:
                    continue
                elif 'lora_out_proj' in key:
                    continue
                elif 'lora_in_proj' in key:
                    continue
                elif 'lora_X' in key:
                    continue
                elif 'lora_Z' in key:
                    continue
                elif 'lora_x_proj' in key:
                    continue
                elif 'lora_d' in key:
                    continue
                elif 'lora_B' in key:
                    continue
                elif 'lora_C' in key:
                    continue
                elif 'lora_dt' in key:
                    continue
                elif 'lora_patch_embed' in key:
                    continue
                elif 'prefix_encoder' in key:
                    continue
                elif 'prompt_encoder' in key:
                    continue
                elif "A_logs" in key.split(".") and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif "Ds" in key.split(".") and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif "conv2d" in key.split(".") and self.learnable_conv1d and (not self.train_conv1d_v2):
                    continue
                elif "learnable_conv1d" in key and self.learnable_conv1d and self.learnable_conv1d_v2:
                    continue
                elif "cls_token" in key and self.learnable_cls_token and (not self.learnable_cls_token_v2):
                    continue
                elif "learnable_cls_token" in key and self.learnable_cls_token and self.learnable_cls_token_v2:
                    continue
                elif "pos_embed" in key and self.learnable_pos_embed and (not self.learnable_pos_embed_v2):
                    continue
                elif "learnable_pos_embed" in key and self.learnable_pos_embed and self.learnable_pos_embed_v2:
                    continue
                elif "bias" in key and "dt_proj" in key and self.learnable_bias and (not self.learnable_bias_v2):
                    continue
                elif "learnable_bias" in key and self.learnable_bias and self.learnable_bias_v2:
                    continue
                elif '_scan_addi' in key:
                    continue
                else:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.module.__getattribute__(name)
