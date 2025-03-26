# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
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
from mmpretrain.models.utils import resize_pos_embed
import math

from personal_lib.mmpretrain.models.backbones.vim import VisionMamba, Block, BidirectionalMamba, PatchEmbed

try:
    from mamba_ssm_1p1p1.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn, mamba_inner_fn_no_out_proj = None, None

try:
    try:
        from mamba_ssm_1p1p1.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    except:
        from bi_mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def forward_mamba_peft(self, hidden_states, inference_params=None, **peft_kwargs):
    """
    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """

    if peft_kwargs.get("outer_single_prefix") is not None: # biMamba考慮せずに先頭にprefix
        hidden_states = torch.cat([peft_kwargs.get("outer_single_prefix"), hidden_states], dim=1)
        vertual_token_num = peft_kwargs.get("outer_single_prefix").shape[1]
    elif peft_kwargs.get("outer_dual_prefix") is not None: # biMambaを考慮して先頭と最後の両方にprefix
        outer_dual_prefix = peft_kwargs.get("outer_dual_prefix")
        hidden_states = torch.cat([outer_dual_prefix[..., :outer_dual_prefix.shape[2]//2], hidden_states, outer_dual_prefix[..., outer_dual_prefix.shape[2]//2:]], dim=1)
        vertual_token_num = outer_dual_prefix.shape[1]

    batch, seqlen, dim = hidden_states.shape

    conv_state, ssm_state = None, None
    if inference_params is not None:
        raise NotImplementedError

    # We do matmul and transpose BLH -> HBL at the same time
    if self.lora_in_proj is None and self.lora_X is None and self.lora_Z is None:
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    else:
        xz = self.in_proj(hidden_states)
        if self.lora_in_proj is not None:
            if self.use_dora:
                xz = xz + F.linear(hidden_states, self.lora_in_proj(self.in_proj.weight, self.s_in_proj))
            else:
                xz = xz + self.s_in_proj*self.lora_in_proj(hidden_states)
        if self.lora_X is not None:
            if self.use_dora:
                xz[..., :xz.shape[-1]//2] = xz[..., :xz.shape[-1]//2] + F.linear(hidden_states, self.lora_X(self.in_proj.weight[:xz.shape[-1]//2], self.s_X))
            else:
                xz[..., :xz.shape[-1]//2] = xz[..., :xz.shape[-1]//2] + self.s_X*self.lora_X(hidden_states)
        if self.lora_Z is not None:
            if self.use_dora:
                xz[..., xz.shape[-1]//2:] = xz[..., xz.shape[-1]//2:] + F.linear(hidden_states, self.lora_Z(self.in_proj.weight[xz.shape[-1]//2:], self.s_Z))
            else:
                xz[..., xz.shape[-1]//2:] = xz[..., xz.shape[-1]//2:] + self.s_Z*self.lora_Z(hidden_states)
        xz = rearrange(xz, 'b l d -> b d l')

    if self.learnable_A_log is not None:
        A = -torch.exp((self.A_log+self.learnable_A_log).float())  # (d_inner, d_state)
    else:
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

    if self.scan_addition_num > 0:
        A_addi = -torch.exp(self.A_log_scan_addi.float())
        
        if self.scan_addition_pos=="suffix":
            A = torch.cat([A, A_addi], dim=1)
        else:
            A = torch.cat([A_addi, A], dim=1)

    # In the backward pass we write dx and dz next to each other to avoid torch.cat
    if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
        if self.bimamba_type == "v1":
            raise NotImplementedError
        elif self.bimamba_type == "v2":
            if self.learnable_A_b_log is not None:
                A_b = -torch.exp((self.A_b_log+self.learnable_A_b_log).float())  # (d_inner, d_state)
            else:
                A_b = -torch.exp(self.A_b_log.float())
            
            x_proj_weight = self.x_proj.weight
            x_proj_b_weight = self.x_proj_b.weight
            if self.lora_x_proj is not None:
                if self.use_dora:
                    x_proj_weight = x_proj_weight +  self.lora_x_proj(self.x_proj.weight, self.s_x_proj)
                    x_proj_b_weight = x_proj_b_weight + self.lora_x_proj_b(self.x_proj_b.weight, self.s_x_proj)
                else:
                    x_proj_weight = x_proj_weight + self.s_x_proj * self.lora_x_proj.adapter_up.weight @ self.lora_x_proj.adapter_down.weight
                    x_proj_b_weight = x_proj_b_weight + self.s_x_proj * self.lora_x_proj_b.adapter_up.weight @ self.lora_x_proj_b.adapter_down.weight
            if self.lora_d is not None:
                if self.use_dora:
                    x_proj_weight = x_proj_weight + torch.cat([self.lora_d(self.x_proj.weight[:self.dt_rank],self.s_d), torch.zeros((self.d_state*2, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([self.lora_d_b(self.x_proj_b.weight[:self.dt_rank],self.s_d), torch.zeros((self.d_state*2, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                else:
                    x_proj_weight = x_proj_weight + torch.cat([self.s_d * self.lora_d.adapter_up.weight @ self.lora_d.adapter_down.weight, torch.zeros((self.d_state*2, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([self.s_d * self.lora_d_b.adapter_up.weight @ self.lora_d_b.adapter_down.weight, torch.zeros((self.d_state*2, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
            if self.lora_B is not None:
                if self.use_dora:
                    x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.dt_rank, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.lora_B(self.x_proj.weight[self.dt_rank:self.dt_rank+self.d_state], self.s_B), torch.zeros((self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([torch.zeros((self.dt_rank, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.lora_B_b(self.x_proj_b.weight[self.dt_rank:self.dt_rank+self.d_state], self.s_B), torch.zeros((self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                else:
                    x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.dt_rank, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_B * self.lora_B.adapter_up.weight @ self.lora_B.adapter_down.weight, torch.zeros((self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([torch.zeros((self.dt_rank, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_B * self.lora_B_b.adapter_up.weight @ self.lora_B_b.adapter_down.weight, torch.zeros((self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
            if self.lora_C is not None:
                if self.use_dora:
                    x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.dt_rank+self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.lora_C(self.x_proj.weight[self.dt_rank+self.d_state:], self.s_C)], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([torch.zeros((self.dt_rank+self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.lora_C_b(self.x_proj_b.weight[self.dt_rank+self.d_state:], self.s_C)], dim=0)
                else:
                    x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.dt_rank+self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_C * self.lora_C.adapter_up.weight @ self.lora_C.adapter_down.weight], dim=0)
                    x_proj_b_weight = x_proj_b_weight + torch.cat([torch.zeros((self.dt_rank+self.d_state, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_C * self.lora_C_b.adapter_up.weight @ self.lora_C_b.adapter_down.weight], dim=0)

            if self.scan_addition_num > 0:
                A_b_addi = -torch.exp(self.A_b_log_scan_addi.float())

                if self.scan_addition_pos=="suffix":
                    A_b = torch.cat([A_b, A_b_addi], dim=1)
                    x_proj_weight = torch.cat([x_proj_weight[:self.dt_rank + self.d_state], self.x_proj_scan_addi.weight[:self.scan_addition_num], x_proj_weight[-self.d_state:], self.x_proj_scan_addi.weight[self.scan_addition_num:]], dim=0).contiguous()
                    x_proj_b_weight = torch.cat([x_proj_b_weight[:self.dt_rank + self.d_state], self.x_proj_b_scan_addi.weight[:self.scan_addition_num], x_proj_b_weight[-self.d_state:], self.x_proj_b_scan_addi.weight[self.scan_addition_num:]], dim=0).contiguous()
                else:
                    A_b = torch.cat([A_b_addi, A_b], dim=1)
                    x_proj_weight = torch.cat([x_proj_weight[:self.dt_rank], self.x_proj_scan_addi.weight[:self.scan_addition_num], x_proj_weight[self.dt_rank: self.dt_rank + self.d_state], self.x_proj_scan_addi.weight[self.scan_addition_num:], x_proj_weight[-self.d_state:]], dim=0).contiguous()
                    x_proj_b_weight = torch.cat([x_proj_b_weight[:self.dt_rank], self.x_proj_b_scan_addi.weight[:self.scan_addition_num], x_proj_b_weight[self.dt_rank: self.dt_rank + self.d_state], self.x_proj_b_scan_addi.weight[self.scan_addition_num:], x_proj_b_weight[-self.d_state:]], dim=0).contiguous()
            else:
                x_proj_weight = x_proj_weight
                x_proj_b_weight = x_proj_b_weight

            dt_proj_weight = self.dt_proj.weight
            dt_proj_b_weight = self.dt_proj_b.weight
            dt_proj_bias = self.dt_proj.bias.float()
            dt_proj_b_bias = self.dt_proj_b.bias.float()
            if self.lora_dt is not None:
                if self.use_dora:
                    dt_proj_weight = dt_proj_weight + self.lora_dt(dt_proj_weight, self.s_dt)
                    dt_proj_b_weight = dt_proj_b_weight + self.lora_dt_b(dt_proj_b_weight, self.s_dt)
                else:
                    dt_proj_weight = dt_proj_weight + self.s_dt * self.lora_dt.adapter_up.weight @ self.lora_dt.adapter_down.weight
                    dt_proj_b_weight = dt_proj_b_weight + self.s_dt * self.lora_dt_b.adapter_up.weight @ self.lora_dt_b.adapter_down.weight
            
            if self.learnable_bias is not None:
                dt_proj_bias = dt_proj_bias + self.learnable_bias
                dt_proj_b_bias = dt_proj_b_bias + self.learnable_bias_b

            if self.learnable_conv1d_weight is not None:
                conv1d_weight = self.conv1d.weight + self.learnable_conv1d_weight
            else:
                conv1d_weight = self.conv1d.weight
            
            if self.learnable_conv1d_weight_b is not None:
                conv1d_weight_b = self.conv1d_b.weight + self.learnable_conv1d_weight_b
            else:
                conv1d_weight_b = self.conv1d_b.weight

            if peft_kwargs.get("inner_single_prefix") is not None: # 先頭にprefix (f,b共に同じprefix)
                single_prefix = peft_kwargs.get("inner_single_prefix").permute(0,2,1)
                xz_f = torch.cat([single_prefix, xz], dim=2)
                xz_b = torch.cat([single_prefix, xz.flip([-1])], dim=2)
                vertual_token_num = single_prefix.shape[2]
            elif peft_kwargs.get("inner_dual_prefix") is not None: # 先頭にprefix (f,b別々のprefix)
                dual_prefix = peft_kwargs.get("inner_dual_prefix").permute(0,2,1)
                xz_f = torch.cat([dual_prefix[:, :dual_prefix.shape[1]//2], xz], dim=2)
                xz_b = torch.cat([dual_prefix[:, dual_prefix.shape[1]//2:], xz.flip([-1])], dim=2)
                vertual_token_num = dual_prefix.shape[2]
            elif peft_kwargs.get("inner_single_suffix") is not None: # 先頭にsuffix (f,b共に同じprefix)
                single_prefix = peft_kwargs.get("inner_single_suffix").permute(0,2,1)
                xz_f = torch.cat([xz, single_prefix], dim=2)
                xz_b = torch.cat([xz.flip([-1]), single_prefix], dim=2)
                vertual_token_num = single_prefix.shape[2]
            elif peft_kwargs.get("inner_single_infix") is not None: # 中間にinfix (f,b共に同じinfix)
                single_prefix = peft_kwargs.get("inner_single_infix").permute(0,2,1)
                vertual_token_num = single_prefix.shape[2]
                
                xz_f_ = torch.tensor_split(xz, vertual_token_num+1, dim=2)
                xz_f_len = [xz_f_[i].shape[2]+1 for i in range(len(xz_f_))]
                cum_xz_f_len = np.cumsum(xz_f_len)
                xz_f = [xz_f_[(i+1)//2] if i%2==0 else single_prefix[:,:,[i//2]] for i in range(vertual_token_num*2+1)]
                xz_f = torch.cat(xz_f, dim=2)

                xz_b = xz_f.flip([-1])
            elif peft_kwargs.get("inner_dual_infix") is not None: # 中間にinfix (f,b異なるinfix)
                single_prefix = peft_kwargs.get("inner_dual_infix").permute(0,2,1)
                single_prefix, single_prefix_b = torch.tensor_split(single_prefix, 2, dim=1)
                vertual_token_num = single_prefix.shape[2]
                
                xz_f_ = torch.tensor_split(xz, vertual_token_num+1, dim=2)
                xz_f_len = [xz_f_[i].shape[2]+1 for i in range(len(xz_f_))]
                cum_xz_f_len = np.cumsum(xz_f_len)
                xz_f = [xz_f_[(i+1)//2] if i%2==0 else single_prefix[:,:,[i//2]] for i in range(vertual_token_num*2+1)]
                xz_f = torch.cat(xz_f, dim=2)

                xz_b_ = torch.tensor_split(xz.flip([-1]), vertual_token_num+1, dim=2)
                xz_b = [xz_b_[(i+1)//2] if i%2==0 else single_prefix_b[:,:,[i//2]] for i in range(vertual_token_num*2+1)]
                xz_b = torch.cat(xz_b, dim=2)
            elif peft_kwargs.get("inner_dual_centfix") is not None: # 真ん中にaffix (f,b異なるprefix)
                dual_prefix = peft_kwargs.get("inner_dual_centfix").permute(0,2,1)
                cls_token_pos = (xz.shape[2]+1)//2
                xz_f = torch.cat([xz[:, :, :cls_token_pos], dual_prefix[:, :dual_prefix.shape[1]//2], xz[:, :, cls_token_pos:]], dim=2)
                xz_b = torch.cat([xz[:, :, :cls_token_pos+1], dual_prefix[:, dual_prefix.shape[1]//2:], xz[:, :, cls_token_pos+1:]], dim=2)
                xz_b =  xz_b.flip([-1])
                vertual_token_num = dual_prefix.shape[2]
            elif peft_kwargs.get("inner_dual_centfix_v2") is not None: # 真ん中にaffix (f,b異なるprefix)
                dual_prefix = peft_kwargs.get("inner_dual_centfix_v2").permute(0,2,1)
                cls_token_pos = (xz.shape[2]+1)//2
                xz_f = torch.cat([xz[:, :, :cls_token_pos], dual_prefix[:, :dual_prefix.shape[1]//2, :dual_prefix.shape[2]//2], xz[:, :, [cls_token_pos]], dual_prefix[:,  :dual_prefix.shape[1]//2, dual_prefix.shape[2]//2:], xz[:, :, cls_token_pos+1:]], dim=2)
                xz_b = torch.cat([xz[:, :, :cls_token_pos], dual_prefix[:, dual_prefix.shape[1]//2:, :dual_prefix.shape[2]//2], xz[:, :, [cls_token_pos]], dual_prefix[:,  dual_prefix.shape[1]//2:, dual_prefix.shape[2]//2:], xz[:, :, cls_token_pos+1:]], dim=2)
                xz_b =  xz_b.flip([-1])
                vertual_token_num = dual_prefix.shape[2]
            elif peft_kwargs.get("inner_single_centfix_v2") is not None: # 真ん中にaffix (f,b共に同じprefix)
                single_prefix = peft_kwargs.get("inner_single_centfix_v2").permute(0,2,1)
                cls_token_pos = (xz.shape[2]+1)//2
                xz_f = torch.cat([xz[:, :, :cls_token_pos], single_prefix[:, :, :single_prefix.shape[2]//2], xz[:, :, [cls_token_pos]], single_prefix[:, :, single_prefix.shape[2]//2:], xz[:, :, cls_token_pos+1:]], dim=2)
                xz_b =  xz_f.flip([-1])
                vertual_token_num = single_prefix.shape[2]
            else:
                xz_f = xz
                xz_b = xz.flip([-1])

            out = mamba_inner_fn_no_out_proj(
                xz_f,
                conv1d_weight,
                self.conv1d.bias,
                x_proj_weight,
                dt_proj_weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float() if self.learnable_D is None else (self.D+self.learnable_D).float(),
                delta_bias=dt_proj_bias,
                delta_softplus=True,
            )
            out_b = mamba_inner_fn_no_out_proj(
                xz_b,
                conv1d_weight_b,
                self.conv1d_b.bias,
                x_proj_b_weight,
                dt_proj_b_weight,
                A_b,
                None,
                None,
                self.D_b.float() if self.learnable_D_b is None else (self.D_b+self.learnable_D_b).float(),
                delta_bias=dt_proj_b_bias,
                delta_softplus=True,
            )

            if "inner_single_prefix" in peft_kwargs.keys() or "inner_dual_prefix" in peft_kwargs.keys():
                out = out [..., vertual_token_num:]
                out_b = out_b[..., vertual_token_num:]
            elif "outer_single_prefix" in peft_kwargs.keys():
                out = out [..., vertual_token_num:]
                out_b = out_b[..., :-vertual_token_num]
            elif "outer_dual_prefix" in peft_kwargs.keys():
                out = out [..., vertual_token_num:-vertual_token_num]
                out_b = out_b[..., vertual_token_num:-vertual_token_num]
            elif "inner_single_suffix" in peft_kwargs.keys():
                out = out [..., :-vertual_token_num]
                out_b = out_b[..., :-vertual_token_num]
            elif "inner_single_infix" in peft_kwargs.keys():
                cum_xz_f_len = np.append(0, cum_xz_f_len)
                out = torch.cat([out[...,cum_xz_f_len[i]:cum_xz_f_len[i+1]-1] for i in range(len(cum_xz_f_len)-1)], dim=-1)
                out_b_f = out_b.flip([-1]) # TODO: More efficient impl
                out_b_f = torch.cat([out_b_f[...,cum_xz_f_len[i]:cum_xz_f_len[i+1]-1] for i in range(len(cum_xz_f_len)-1)], dim=-1)
                out_b = out_b_f.flip([-1])
            elif "inner_dual_infix" in peft_kwargs.keys():
                cum_xz_f_len = np.append(0, cum_xz_f_len)
                out = torch.cat([out[...,cum_xz_f_len[i]:cum_xz_f_len[i+1]-1] for i in range(len(cum_xz_f_len)-1)], dim=-1)
                out_b = torch.cat([out_b[...,cum_xz_f_len[i]:cum_xz_f_len[i+1]-1] for i in range(len(cum_xz_f_len)-1)], dim=-1)
            elif "inner_dual_centfix" in peft_kwargs.keys():
                out = torch.cat([out[..., :cls_token_pos], out[..., cls_token_pos+vertual_token_num:]], dim=-1)
                out_b = torch.cat([out_b[..., :cls_token_pos+1], out_b[..., cls_token_pos+1+vertual_token_num:]], dim=-1)
            elif "inner_dual_centfix_v2" in peft_kwargs.keys():
                right_len = xz.shape[2]-cls_token_pos-1
                out = torch.cat([out[..., :cls_token_pos], out[..., [cls_token_pos+dual_prefix.shape[2]//2]], out[..., -right_len:]], dim=-1)
                out_b = torch.cat([out_b[..., :right_len], out[..., [- (cls_token_pos+(dual_prefix.shape[2] - (dual_prefix.shape[2]//2)))]], out[..., -cls_token_pos:]], dim=-1)
            elif "inner_single_centfix_v2" in peft_kwargs.keys():
                right_len = xz.shape[2]-cls_token_pos-1
                out = torch.cat([out[..., :cls_token_pos], out[..., [cls_token_pos+vertual_token_num]], out[..., -right_len:]], dim=-1)
                out_b = torch.cat([out_b[..., :right_len], out[..., [- (cls_token_pos+vertual_token_num)]], out[..., -cls_token_pos:]], dim=-1)
                

            # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            if not self.if_devide_out:
                out_ = rearrange(out + out_b.flip([-1]), "b d l -> b l d")
                if self.lora_out_proj is None:
                    out = F.linear(out_, self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_dora:
                        out = F.linear(out_, self.out_proj.weight+self.lora_out_proj(self.out_proj.weight, self.s), self.out_proj.bias)
                    else:
                        out = F.linear(out_, self.out_proj.weight, self.out_proj.bias) + self.lora_out_proj(out_)*self.s

            else:
                out_ = rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2
                if self.lora_out_proj is None:
                    out = F.linear(out_, self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_dora:
                        out = F.linear(out_, self.out_proj.weight+self.lora_out_proj(self.out_proj.weight, self.s), self.out_proj.bias)
                    else:
                        out = F.linear(out_, self.out_proj.weight, self.out_proj.bias) + self.lora_out_proj(out_)*self.s        
            if self.adaptf is not None:
                out = out + self.s_adaptf*self.adaptf(out_)
        else:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                x_proj_weight,
                dt_proj_weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
    else:
        raise NotImplementedError
    return out

def forward_vimblock_peft(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **peft_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **peft_kwargs)

        return hidden_states, residual


def forward_features_visionmamba_peft(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add the dist_token
    height, width = x.shape[2:4]

    x = self.patch_embed(x)
    B, M, _ = x.shape

    if self.if_cls_token:
        if self.use_double_cls_token:
            raise NotImplementedError
        else:
            if self.use_middle_cls_token:
                
                cls_token = self.cls_token.expand(B, -1, -1)
                if self.learnable_cls_token is not None:
                    cls_token = cls_token + self.learnable_cls_token.expand(B, -1, -1)
                token_position = M // 2
                # add cls token in the middle
                x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            elif if_random_cls_token_position:
                raise NotImplementedError
            else:
                raise NotImplementedError
            M = x.shape[1]

    if self.if_abs_pos_embed:
        # x = x + resize_pos_embed(
        #         self.pos_embed,
        #         self.patch_embed.grid_size,
        #         ((height - self.patch_size) // self.patch_stride + 1, (width - self.patch_size) // self.patch_stride + 1),
        #         mode='bicubic',
        #         num_extra_tokens=self.num_tokens)
        x = x + self.pos_embed
        if self.learnable_pos_embed is not None:
            x = x + self.learnable_pos_embed
        x = self.pos_drop(x)

    if if_random_token_rank:

        # 生成随机 shuffle 索引
        shuffle_indices = torch.randperm(M)

        if isinstance(token_position, list):
            print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
        else:
            print("original value: ", x[0, token_position, 0])
        print("original token_position: ", token_position)

        # 执行 shuffle
        x = x[:, shuffle_indices, :]

        if isinstance(token_position, list):
            # 找到 cls token 在 shuffle 之后的新位置
            new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
            token_position = new_token_position
        else:
            # 找到 cls token 在 shuffle 之后的新位置
            token_position = torch.where(shuffle_indices == token_position)[0].item()

        if isinstance(token_position, list):
            print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
        else:
            print("new value: ", x[0, token_position, 0])
        print("new token_position: ", token_position)


    if self.prompt_encoder is not None: # 先頭にprefix (f,b共に同じprefix)
        prompts = self.prompt_encoder(x.shape[0])
        if self.prompt_type == "prefix":
            x = torch.cat([prompts, x], dim=1)
            if isinstance(token_position, list):
                raise NotImplementedError
            else:
                token_position = token_position + prompts.shape[1]
        elif self.prompt_type == "infix":
            vertual_token_num = prompts.shape[1]
            x_ = torch.tensor_split(x, vertual_token_num+1, dim=1)
            if isinstance(token_position, list):
                raise NotImplementedError
            else:
                token_position = token_position + prompts.shape[1]
                token_position = token_position + token_position//math.ceil(x.shape[1]/(vertual_token_num+1))
            x = [x_[(i+1)//2] if i%2==0 else prompts[:,[i//2]] for i in range(vertual_token_num*2+1)]
            x = torch.cat(x, dim=1)
        elif self.prompt_type == "suffix":
            x = torch.cat([x, prompts], dim=1)
        elif self.prompt_type == "prefix_suffix":
            prompts = torch.tensor_split(prompts, 2, dim=1)
            x = torch.cat([prompts[0],x, prompts[1]], dim=1)
            if isinstance(token_position, list):
                raise NotImplementedError
            else:
                token_position = token_position + prompts[0].shape[1]
        else:
            raise NotImplementedError

    if_flip_img_sequences = False
    if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
        x = x.flip([1])
        if_flip_img_sequences = True

    # mamba impl
    residual = None
    hidden_states = x
    if not self.if_bidirectional:
        if self.prefix_encoder is not None:
            prefixes = self.prefix_encoder(x.shape[0])


        for i_l, layer in enumerate(self.layers):
            
            if if_flip_img_sequences and self.if_rope:
                hidden_states = hidden_states.flip([1])
                if residual is not None:
                    residual = residual.flip([1])

            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            if if_flip_img_sequences and self.if_rope:
                hidden_states = hidden_states.flip([1])
                if residual is not None:
                    residual = residual.flip([1])

            peft_kwargs={}
            if self.prefix_encoder is not None:
                peft_kwargs[self.prefix_type] = prefixes[...,i_l*self.prefix_ch_per_layer:(i_l+1)*self.prefix_ch_per_layer]

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **peft_kwargs
            )
    else:
        raise NotImplementedError

    if not self.fused_add_norm:
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
    else:
        # Set prenorm=False here since we don't need the residual
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )

    # return only cls token if it exists
    if self.if_cls_token:
        if self.use_double_cls_token:
            return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
        else:
            if self.use_middle_cls_token:
                return hidden_states[:, token_position, :]
            elif if_random_cls_token_position:
                return hidden_states[:, token_position, :]
            else:
                return hidden_states[:, token_position, :]

    if self.final_pool_type == 'none':
        return hidden_states[:, -1, :]
    elif self.final_pool_type == 'mean':
        return hidden_states.mean(dim=1)
    elif self.final_pool_type == 'max':
        return hidden_states
    elif self.final_pool_type == 'all':
        return hidden_states
    else:
        raise NotImplementedError

def forward_patch_embed(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x) + self.lora_patch_embed(x)*self.s_patch_embed
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    x = self.norm(x)
    return x


class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32, use_act=False):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Linear(in_channels, dim, bias=False)
            self.adapter_up = nn.Linear(dim, out_channels, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            raise NotImplementedError
            # self.adapter_down = QLinear(768, dim, bit)
            # self.adapter_up = QLinear(dim, 768, bit)
            # nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        if use_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up

class DoRAAdapter(nn.Module):

    def __init__(self, in_channels, out_channels, dim, bit=32, use_act=False, ori_weight=None):
        super().__init__()
        if bit == 32:
            self.adapter_down = nn.Linear(in_channels, dim, bias=False)
            self.adapter_up = nn.Linear(dim, out_channels, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
            self.m = nn.Parameter(ori_weight.norm(p=2, dim=0, keepdim=True))
        else:
            raise NotImplementedError
            # self.adapter_down = QLinear(768, dim, bit)
            # self.adapter_up = QLinear(dim, 768, bit)
            # nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        if use_act:
            raise NotImplementedError
        else:
            self.act = nn.Identity()
        self.dim = dim

    def forward(self, weight, scale):
        lora = self.adapter_up.weight @ self.adapter_down.weight
        numerator = weight + scale*lora
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator.detach()
        new_weight = self.m * directional_component
        return new_weight-weight

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
    def __init__(self, prefix_projection=True, token_dim=768, num_virtual_tokens=20, encoder_hidden_size=768, num_layers=12, num_per_layer=2):
        super().__init__()
        self.prefix_projection = prefix_projection
        self.prompt_tokens = torch.arange(
            num_virtual_tokens
        ).long()
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * num_per_layer * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * num_per_layer * token_dim)

    def forward(self, batch_size):
        prefix = (
            self.prompt_tokens
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.embedding.weight.device)
        )
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
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
             use_dora=False,
             ):
    
    if type(model) == VisionMamba:
        if prefix_tuning:
            num_layers = len(model.layers)
            token_dim = model.embed_dim
            if prefix_type in ["inner_single_prefix", "outer_single_prefix", "inner_single_suffix", "inner_single_infix", "inner_single_centfix_v2"]:
                num_per_layer = 1
            elif prefix_type in ["inner_dual_prefix", "outer_dual_prefix", "inner_dual_infix", "inner_dual_centfix", "inner_dual_centfix_v2"]:
                num_per_layer = 2
            else:
                raise NotImplementedError
            
            if prefix_type in ["inner_single_prefix", "inner_dual_prefix", "inner_single_suffix", "inner_single_infix", "inner_dual_infix", "inner_dual_centfix", "inner_dual_centfix_v2", "inner_single_centfix_v2"]:
                token_dim = token_dim * model.layers[0].mixer.expand * 2
            encoder_hidden_size_ = encoder_hidden_size if encoder_hidden_size is not None else token_dim
            model.prefix_ch_per_layer = num_per_layer*token_dim
            model.prefix_encoder = PrefixEncoder(prefix_projection, token_dim, num_virtual_tokens, encoder_hidden_size_, num_layers, num_per_layer)
            model.prefix_type = prefix_type        
        else:
            model.prefix_encoder = None

        if prompt_tuning:
            token_dim = model.embed_dim
            model.prompt_encoder = PrefixEncoder(prompt_projection, token_dim, prompt_num_tokens, token_dim, 1, 1)
            model.prompt_type = prompt_type
        else:
            model.prompt_encoder = None
        
        if learnable_cls_token and learnable_cls_token_v2:
            model.learnable_cls_token= nn.Parameter(torch.zeros_like(model.cls_token.data))
        else:
            model.learnable_cls_token = None
        
        if learnable_pos_embed and learnable_pos_embed_v2:
            model.learnable_pos_embed= nn.Parameter(torch.zeros_like(model.pos_embed.data))
        else:
            model.learnable_pos_embed = None

        bound_method = forward_features_visionmamba_peft.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)

    for layer in model.children():
        if type(layer) == BidirectionalMamba:
            # print(f"add lora to {layer}")

            if adaptformer:
                in_ch, out_ch = layer.d_inner, layer.d_model
                layer.adaptf = Adapter(in_ch, out_ch, dim_adaptf, bit_adaptf, use_act=True)
                layer.s_adaptf = s_adaptf
            else:
                layer.adaptf = None

            layer.use_dora = use_dora

            if lora_out_proj:
                in_ch, out_ch = layer.d_inner, layer.d_model
                if use_dora:
                    layer.lora_out_proj = DoRAAdapter(in_ch, out_ch, dim, bit, ori_weight=layer.out_proj.weight)
                else:
                    layer.lora_out_proj = Adapter(in_ch, out_ch, dim, bit)
                layer.s = s
            else:
                layer.lora_out_proj = None
            
            if lora_in_proj:
                in_ch, out_ch = layer.d_model, layer.d_inner*2
                if use_dora:
                    layer.lora_in_proj = DoRAAdapter(in_ch, out_ch, dim_in_proj, bit_in_proj, ori_weight=layer.in_proj.weight)
                else:
                    layer.lora_in_proj = Adapter(in_ch, out_ch, dim_in_proj, bit_in_proj)
                layer.s_in_proj = s_in_proj
            else:
                layer.lora_in_proj = None
            
            if lora_X:
                in_ch, out_ch = layer.d_model, layer.d_inner
                if use_dora:
                    layer.lora_X = DoRAAdapter(in_ch, out_ch, dim_X, bit_X, ori_weight=layer.in_proj.weight[:out_ch])
                else:
                    layer.lora_X = Adapter(in_ch, out_ch, dim_X, bit_X)
                layer.s_X = s_X
            else:
                layer.lora_X = None
            
            if lora_Z:
                in_ch, out_ch = layer.d_model, layer.d_inner
                if use_dora:
                    layer.lora_Z = DoRAAdapter(in_ch, out_ch, dim_Z, bit_Z, ori_weight=layer.in_proj.weight[out_ch:])
                else:
                    layer.lora_Z = Adapter(in_ch, out_ch, dim_Z, bit_Z)
                layer.s_Z = s_Z
            else:
                layer.lora_Z = None
            
            if lora_x_proj:
                in_ch, out_ch = layer.d_inner, layer.dt_rank+layer.d_state*2
                if use_dora:
                    layer.lora_x_proj = DoRAAdapter(in_ch, out_ch, dim_x_proj, bit_x_proj, ori_weight=layer.x_proj.weight)
                    layer.lora_x_proj_b = DoRAAdapter(in_ch, out_ch, dim_x_proj, bit_x_proj, ori_weight=layer.x_proj_b.weight)
                else:
                    layer.lora_x_proj = Adapter(in_ch, out_ch, dim_x_proj, bit_x_proj)
                    layer.lora_x_proj_b = Adapter(in_ch, out_ch, dim_x_proj, bit_x_proj)
                layer.s_x_proj = s_x_proj
            else:
                layer.lora_x_proj = None
            
            if lora_d:
                in_ch, out_ch = layer.d_inner, layer.dt_rank
                if use_dora:
                    layer.lora_d = DoRAAdapter(in_ch, out_ch, dim_d, bit_d, ori_weight=layer.x_proj.weight[:layer.dt_rank])
                    layer.lora_d_b = DoRAAdapter(in_ch, out_ch, dim_d, bit_d, ori_weight=layer.x_proj.weight[:layer.dt_rank])
                else:
                    layer.lora_d = Adapter(in_ch, out_ch, dim_d, bit_d)
                    layer.lora_d_b = Adapter(in_ch, out_ch, dim_d, bit_d)
                layer.s_d = s_d
            else:
                layer.lora_d = None
            
            if lora_B:
                in_ch, out_ch = layer.d_inner, layer.d_state
                if use_dora:
                    layer.lora_B = DoRAAdapter(in_ch, out_ch, dim_B, bit_B, ori_weight=layer.x_proj.weight[layer.dt_rank:layer.dt_rank+layer.d_state])
                    layer.lora_B_b = DoRAAdapter(in_ch, out_ch, dim_B, bit_B, ori_weight=layer.x_proj.weight[layer.dt_rank:layer.dt_rank+layer.d_state])
                else:
                    layer.lora_B = Adapter(in_ch, out_ch, dim_B, bit_B)
                    layer.lora_B_b = Adapter(in_ch, out_ch, dim_B, bit_B)
                layer.s_B = s_B
            else:
                layer.lora_B = None
            
            if lora_C:
                in_ch, out_ch = layer.d_inner, layer.d_state
                if use_dora:
                    layer.lora_C = DoRAAdapter(in_ch, out_ch, dim_C, bit_C, ori_weight=layer.x_proj.weight[layer.dt_rank+layer.d_state:])
                    layer.lora_C_b = DoRAAdapter(in_ch, out_ch, dim_C, bit_C, ori_weight=layer.x_proj.weight[layer.dt_rank+layer.d_state:])
                else:
                    layer.lora_C = Adapter(in_ch, out_ch, dim_C, bit_C)
                    layer.lora_C_b = Adapter(in_ch, out_ch, dim_C, bit_C)
                layer.s_C = s_C
            else:
                layer.lora_C = None
            
            if lora_dt:
                in_ch, out_ch = layer.dt_rank, layer.d_inner
                if use_dora:
                    layer.lora_dt = DoRAAdapter(in_ch, out_ch, dim_dt, bit_dt, ori_weight=layer.dt_proj.weight)
                    layer.lora_dt_b = DoRAAdapter(in_ch, out_ch, dim_dt, bit_dt, ori_weight=layer.dt_proj_b.weight)
                else:
                    layer.lora_dt = Adapter(in_ch, out_ch, dim_dt, bit_dt)
                    layer.lora_dt_b = Adapter(in_ch, out_ch, dim_dt, bit_dt)
                layer.s_dt = s_dt
            else:
                layer.lora_dt = None
            
            if learnable_conv1d and learnable_conv1d_v2:
                device = layer.conv1d.weight.device
                layer.learnable_conv1d_weight = nn.Parameter(torch.zeros_like(layer.conv1d.weight, device=device))
                layer.learnable_conv1d_weight_b = nn.Parameter(torch.zeros_like(layer.conv1d_b.weight, device=device))
            else:
                layer.learnable_conv1d_weight = None
                layer.learnable_conv1d_weight_b = None
                

            if additional_scan:
                d_state = layer.d_state
                d_inner = layer.d_inner
                device = layer.A_log.device
                dtype= layer.A_log.dtype
                
                if scan_A_copy_from_last:
                    if scan_addition_pos == "suffix":
                        A_scan_addi_log= repeat(layer.A_log.data[:, -1], "d -> d n",
                            n=scan_addition_num,
                        ).contiguous()
                    else:
                        A_scan_addi_log= repeat(layer.A_log.data[:, 0], "d -> d n",
                            n=scan_addition_num,
                        ).contiguous()
                elif scan_A_constant is None:
                    A_scan_addi = repeat(
                        torch.arange(d_state+1, d_state+1+scan_addition_num, dtype=dtype, device=device),
                        "n -> d n",
                        d=d_inner,
                    ).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_log in fp32   
                else:
                    A_scan_addi = repeat(
                        scan_A_constant*torch.ones((scan_addition_num,), dtype=dtype, device=device),
                        "n -> d n",
                        d=d_inner,
                    ).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_log in fp32      
                layer.A_log_scan_addi = nn.Parameter(A_scan_addi_log)
                layer.A_log_scan_addi._no_weight_decay = True
                layer.x_proj_scan_addi = nn.Linear(
                    d_inner, scan_addition_num * 2, bias=False, dtype=dtype, device=device
                )

                if layer.bimamba_type == "v2":
                    if scan_A_copy_from_last:
                        if scan_addition_pos == "suffix":
                            A_b_scan_addi_log= repeat(layer.A_b_log.data[:, -1], "d -> d n",
                                n=scan_addition_num,
                            ).contiguous()
                        else:
                            A_b_scan_addi_log= repeat(layer.A_b_log.data[:, 0], "d -> d n",
                                n=scan_addition_num,
                            ).contiguous()
                        
                    elif scan_A_constant is None:
                        A_b_scan_addi = repeat(
                            torch.arange(d_state+1, d_state+1+scan_addition_num, dtype=dtype, device=device),
                            "n -> d n",
                            d=d_inner,
                        ).contiguous()
                        A_b_scan_addi_log = torch.log(A_scan_addi)  # Keep A_log in fp32
                    else:
                        A_b_scan_addi = repeat(
                            scan_A_constant*torch.ones((scan_addition_num,), dtype=dtype, device=device),
                            "n -> d n",
                            d=d_inner,
                        ).contiguous()
                        A_b_scan_addi_log = torch.log(A_b_scan_addi)  # Keep A_log in fp32      
                    layer.A_b_log_scan_addi = nn.Parameter(A_b_scan_addi_log)
                    layer.A_b_log_scan_addi._no_weight_decay = True,
                    layer.x_proj_b_scan_addi = nn.Linear(
                        d_inner, scan_addition_num * 2, bias=False, dtype=dtype, device=device
                    )
                else:
                    raise NotImplementedError
                layer.scan_addition_num=scan_addition_num
                layer.scan_addition_pos=scan_addition_pos

                if zero_init_x_proj:
                    nn.init.zeros_(layer.x_proj_scan_addi.weight)
                    if layer.bimamba_type == "v2":
                        nn.init.zeros_(layer.x_proj_b_scan_addi.weight)
            else:
                layer.scan_addition_num=-1
            
            if learnable_A and learnable_A_v2:
                device = layer.A_log.data.device
                layer.learnable_A_log = nn.Parameter(torch.zeros_like(layer.A_log.data, device=device))
                layer.learnable_A_b_log = nn.Parameter(torch.zeros_like(layer.A_b_log.data, device=device))
            else:
                layer.learnable_A_log = None
                layer.learnable_A_b_log = None

            if learnable_D and learnable_D_v2:
                device = layer.D.data.device
                layer.learnable_D = nn.Parameter(torch.zeros_like(layer.D.data, device=device))
                layer.learnable_D_b = nn.Parameter(torch.zeros_like(layer.D_b.data, device=device))
            else:
                layer.learnable_D = None
                layer.learnable_D_b = None

            if learnable_bias and learnable_bias_v2:
                device = layer.dt_proj.bias.device
                layer.learnable_bias = nn.Parameter(torch.zeros_like(layer.dt_proj.bias, device=device))
                layer.learnable_bias_b = nn.Parameter(torch.zeros_like(layer.dt_proj_b.bias, device=device))
            else:
                layer.learnable_bias = None
                layer.learnable_bias_b = None


                

                

            bound_method = forward_mamba_peft.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == Block:
            bound_method = forward_vimblock_peft.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == PatchEmbed:
            if lora_patch_embed:
                layer.s_patch_embed=s_patch_embed
                layer.lora_patch_embed = Conv2dAdapter(layer.proj.in_channels, layer.proj.out_channels, dim_patch_embed, bit_patch_embed, layer.proj.kernel_size, layer.proj.stride, layer.proj.padding)
                bound_method = forward_patch_embed.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)

        
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
                     learnable_bias, learnable_bias_v2,
                     use_dora,)



@MODELS.register_module()
class VimPEFTModel(BaseModule):
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
                use_dora=False,
                 ):

        super().__init__()

        module = MODELS.build(module)
        module.init_weights()

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
                    learnable_bias, learnable_bias_v2,
                    use_dora,)

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
            elif ("A_log" in n.split(".") or "A_b_log" in n.split(".")) and self.learnable_A and self.train_A and (not self.learnable_A_v2):
                p.requires_grad = True
            elif "learnable_A" in n and self.learnable_A and self.learnable_A_v2 and self.train_A:
                p.requires_grad = True
            elif ("D" in n.split(".") or "D_b" in n.split(".")) and self.learnable_D and self.train_D and (not self.learnable_D_v2):
                p.requires_grad = True
            elif "learnable_D" in n and self.learnable_D and self.learnable_D_v2 and self.train_D:
                p.requires_grad = True
            elif ("conv1d" in n.split(".") or "conv1d_b" in n.split(".")) and self.learnable_conv1d and self.train_conv1d and (not self.learnable_conv1d_v2):
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
                elif ("A_log" in key.split(".") or "A_b_log" in key.split(".")) and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif ("D" in key.split(".") or "D_b" in key.split(".")) and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif ("conv1d" in key.split(".") or "conv1d_b" in key.split(".")) and self.learnable_conv1d and (not self.learnable_conv1d_v2):
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
                elif ("A_log" in key.split(".") or "A_b_log" in key.split(".")) and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif ("D" in key.split(".") or "D_b" in key.split(".")) and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif ("conv1d" in key.split(".") or "conv1d_b" in key.split(".")) and self.learnable_conv1d and (not self.learnable_conv1d_v2):
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
                elif ("A_log" in key.split(".") or "A_b_log" in key.split(".")) and self.learnable_A and (not self.learnable_A_v2):
                    continue
                elif "learnable_A" in key and self.learnable_A and self.learnable_A_v2:
                    continue
                elif ("D" in key.split(".") or "D_b" in key.split(".")) and self.learnable_D and (not self.learnable_D_v2):
                    continue
                elif "learnable_D" in key and self.learnable_D and self.learnable_D_v2:
                    continue
                elif ("conv1d" in key.split(".") or "conv1d_b" in key.split(".")) and self.learnable_conv1d and (not self.train_conv1d_v2):
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
