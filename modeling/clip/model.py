from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from modeling.backbones.vit_pytorch import trunc_normal_
from tutel import moe as tutel_moe
import os
import cv2

global_var: int = 0


def change_function(global_var):
    global_var = global_var+1
    return global_var


class Expert(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_dim=512,
            groups=1,
            scale=1.0
    ):
        super().__init__()
        r = hidden_dim
        self.conv_A = nn.Conv1d(in_features, r, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(r, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

        self.groups = groups
        self.r = r
        self.scale = scale

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x
        result = self.conv_A(x)
        result = self.dropout(result)
        result = self.conv_B(result)
        result = result * self.scale + residual
        result = result.transpose(1, 2).contiguous()
        return result


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0,
                                                                               1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1)
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        xproj = self.attnpool(x4)

        return x3, x4, xproj


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class RepAdapter(nn.Module):
    def __init__(
            self,
            model_dim=768,
            local_experts=1,
            sharded_count=1,
            my_config=None,
            hidden_dim=512,
            groups=1,
            scale=1.0
    ):
        super().__init__()
        r = hidden_dim
        self.conv_A = nn.Conv1d(model_dim, r, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(r, model_dim, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

        self.groups = groups
        self.r = r
        self.scale = scale

    def forward(self, x, ctx):
        if ctx.sharded_count > 1:
          raise Exception("`sharded_count > 1` is not implemented within this expert, Model parallel is disabled.")
        x = x.transpose(1, 2)
        residual = x
        result = self.conv_A(x)
        result = self.dropout(result)
        result = self.conv_B(result)
        result = result * self.scale + residual
        result = result.transpose(1, 2).contiguous()
        return result

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, pattern=None, Adapter=True):
        super().__init__()
        self.Adapter = Adapter
        self.aux_loss = None
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if Adapter is True:
            num_local_experts = 6
            top_value = 1
            model_dim = d_model
            a2a_ffn_overlap_degree = 1
            dist_rank = 1
            'cosine_top'
            self.smoe = tutel_moe.moe_layer(
                gate_type={'type': 'cosine_top', 'k': top_value, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
                experts={'type': 'custom', 'module': RepAdapter, 'count_per_node': num_local_experts, 'my_config': None},
                model_dim=model_dim,
                scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
                seeds=(1, dist_rank + 1, 1),
                a2a_ffn_overlap_degree=a2a_ffn_overlap_degree,
                use_2dh=False,
                batch_prioritized_routing=True,
                is_gshard_loss=False,
            )
            self.aux_loss_weights = 0.01

        self.begin = -1
        if pattern is not None:
            if "prompt" in pattern:
                self.k = 4
                dropout = 0.0
                self.adapter_prompt_rgb = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_prompt_nir = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_prompt_tir = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_transfer = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                                      QuickGELU(),
                                                      nn.Dropout(dropout),
                                                      nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_r = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_n = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_t = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
            if "adapter" in pattern:
                self.adapter_ffn = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                                 QuickGELU(),
                                                 nn.Linear(int(d_model // 2), int(d_model)))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        out = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return out

    def forward_ori(self, x: torch.Tensor, index=None):
        x = x + self.attention(self.ln_1(x))
        if self.Adapter is True and x.shape[1] > 3:
            if index == 11:
                norm_x = self.ln_2(x)
                x = x + self.smoe(norm_x.float().transpose(0, 1)).transpose(0, 1)
                self.aux_loss = self.smoe.l_aux * self.aux_loss_weights
            else:
                x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_adapter(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.mlp(self.ln_2(x)) + adapter_ffn
        return x

    def forward_with_prompt_only_first_layer(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            if index == 0:
                n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                r = last_prompt + self.adapter_transfer(last_prompt)
                n2r = last_prompt
                t2r = last_prompt
            else:
                r = last_prompt
                n2r = last_prompt
                t2r = last_prompt
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            if index == 0:
                r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                n = last_prompt + self.adapter_transfer(last_prompt)
                r2n = last_prompt
                t2n = last_prompt
            else:
                n = last_prompt
                r2n = last_prompt
                t2n = last_prompt
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            if index == 0:
                r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                t = last_prompt + self.adapter_transfer(last_prompt)
                r2t = last_prompt
                n2t = last_prompt
            else:
                t = last_prompt
                r2t = last_prompt
                n2t = last_prompt
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt_adapter(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)

        x = x + self.attention(self.ln_1(x))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.mlp(self.ln_2(x)) + adapter_ffn
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward(self, x: torch.Tensor, modality=None, index=None, last_prompt=None, prompt_sign=True,
                adapter_sign=True):
        if prompt_sign and adapter_sign:
            return self.forward_with_prompt_adapter(x, modality, index, last_prompt)
        elif prompt_sign and not adapter_sign:
            if index > self.begin:
                return self.forward_with_prompt(x, modality, index, last_prompt)
            else:
                return self.forward_ori(x), None
        elif not prompt_sign and adapter_sign:
            if index > self.begin:
                return self.forward_with_adapter(x)
            else:
                return self.forward_ori(x)
        else:
            # MFRNet only use this branch
            return self.forward_ori(x, index)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, pattern=None, Adapter=True):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, pattern, Adapter=Adapter) for _ in range(layers)])

    def forward(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        return self.resblocks(x, modality, index, last_prompt)


def conv3x3_block(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

  block = nn.Sequential(
    conv_layer,
    nn.BatchNorm2d(out_planes),
    nn.ReLU(inplace=True),
  )
  return block


class Gen_lack_feature(nn.Module):
    def __init__(self, model_dim=768, local_experts=1,
            sharded_count=1,
            my_config=None, kernel_size=1, stride=1):
        super().__init__()
        self.model_dim = model_dim
        self.conv_1 = nn.Conv1d(model_dim, 1024, 1, bias=True)
        self.conv_2 = nn.Conv1d(1024, model_dim, 1, bias=True)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, ctx):
        if ctx.sharded_count > 1:
          raise Exception("`sharded_count > 1` is not implemented within this expert, Model parallel is disabled.")
        out = self.conv_2(self.dropout(self.conv_1(x.transpose(1, 2)))).transpose(1, 2).contiguous()
        return out

class Gen_lack_weight(nn.Module):
    def __init__(self, model_dim=768):
        super().__init__()
        self.aapw = torch.nn.AdaptiveAvgPool2d((1, model_dim))
        self.lw = torch.nn.Linear(model_dim, 1)

    def forward(self, x):
        out = self.lw(self.aapw(x))
        return out

class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int,
                 heads: int, output_dim: int, cfg: dict):
        super().__init__()
        self.miss_type = cfg.TEST.MISS
        self.Adapter = cfg.MODEL.ADA
        self.gen_lack = cfg.MODEL.GEN
        self.prompt_sign = cfg.MODEL.PROMPT
        self.adapter_sign = cfg.MODEL.ADAPTER
        self.pattern = ['nothing']
        if self.prompt_sign:
            self.pattern.append('prompt')
        if self.adapter_sign:
            self.pattern.append('adapter')
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size,
                               bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, pattern=self.pattern, Adapter=self.Adapter)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        if self.gen_lack is True:
            model_dim = width
            num_local_experts = 3
            dist_rank = 1
            self.gfmoe = tutel_moe.moe_layer(
                gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
                experts={'type': 'custom', 'module': Gen_lack_feature, 'count_per_node': num_local_experts,
                         'my_config': None},
                model_dim=model_dim, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
                seeds=(1, dist_rank + 1, 1), a2a_ffn_overlap_degree=1, use_2dh=False, batch_prioritized_routing=True,
                is_gshard_loss=False,
            )
            self.gfmoe_2 = tutel_moe.moe_layer(
                gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
                experts={'type': 'custom', 'module': Gen_lack_feature, 'count_per_node': num_local_experts,
                         'my_config': None},
                model_dim=model_dim, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
                seeds=(1, dist_rank + 1, 1), a2a_ffn_overlap_degree=1, use_2dh=False, batch_prioritized_routing=True,
                is_gshard_loss=False,
            )
            self.aux_loss_weights = 0.01

            depth_num = 1
            self.glw_RN = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.glw_RT = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.glw_NR = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.glw_NT = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.glw_TR = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.glw_TN = nn.ModuleList([Gen_lack_weight() for i in range(depth_num)])
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, cv_emb=None, modality=None):
        # save pic
        # if self.training is False:
        #     # moe_layer visual
        #     # map_image = crit[1][0].unsqueeze(1).view(list(original_shape[:-reserve_dims]))[:,1:].to(original_dtype)
        #     # map_image_ = torch.nn.functional.interpolate(map_image.reshape(map_image.shape[0], 1, 16, 8), scale_factor=16,mode='nearest')
        #     # import SimpleITK as sitk
        #     # sitk.WriteImage(sitk.GetImageFromArray(map_image_.squeeze().to("cpu").detach().numpy()), os.path.join(r'/data/object/Out_MFRNet/img/', 'map.nii'))
        #
        #     change_function(global_var)
        #     savePath = r'/data/object/Out_MFRNet/img/' + str(global_var) + '_'
        #
        #     for i in range(x.shape[0]//3):
        #         rgb_image = x[i, :, :].permute(1, 2, 0)
        #         image = 255 * (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        #         save_image = torch.cat((image[:,:,2].unsqueeze(2),image[:,:,1].unsqueeze(2),image[:,:,0].unsqueeze(2)),dim=2).cpu().numpy()
        #         cv2.imwrite(savePath + 'Image_' + str(i) + '_' + 'src_image_RGB.jpg', save_image)
        #
        #         ni_image = x[i+x.shape[0]//3, :, :].permute(1, 2, 0)
        #         image = 255 * (ni_image - ni_image.min()) / (ni_image.max() - ni_image.min())
        #         save_image = torch.cat((image[:,:,2].unsqueeze(2),image[:,:,1].unsqueeze(2),image[:,:,0].unsqueeze(2)),dim=2).cpu().numpy()
        #         cv2.imwrite(savePath + 'Image_' + str(i) + '_' + 'src_image_NI.jpg', save_image)
        #
        #         ti_image = x[i+2*x.shape[0]//3, :, :].permute(1, 2, 0)
        #         image = 255 * (ti_image - ti_image.min()) / (ti_image.max() - ti_image.min())
        #         save_image = torch.cat((image[:,:,2].unsqueeze(2),image[:,:,1].unsqueeze(2),image[:,:,0].unsqueeze(2)),dim=2).cpu().numpy()
        #         cv2.imwrite(savePath + 'Image_' + str(i) + '_' + 'src_image_TI.jpg', save_image)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb.squeeze(1)
        x = x + self.positional_embedding.to(x.dtype)

        loss_aux_list = 0.0
        B = x.shape[0]
        gen_is = 0
        # if self.gen_lack is True and B > 3:
        #     lamda = 0.5
        #     x_RNT = x.chunk(3, dim=0)
        #     if self.miss_type == 'nothing' or self.training:
        #         N_gen_R = self.gfmoe(x_RNT[1])
        #         T_gen_R = self.gfmoe(x_RNT[2])
        #         w_NR = self.glw_NR[0](x_RNT[1])
        #         w_TR = self.glw_TR[0](x_RNT[2])
        #
        #         R_gen_N = self.gfmoe(x_RNT[0])
        #         T_gen_N = self.gfmoe_2(x_RNT[2])
        #         w_RN = self.glw_RN[0](x_RNT[0])
        #         w_TN = self.glw_TN[0](x_RNT[2])
        #
        #         R_gen_T = self.gfmoe_2(x_RNT[0])
        #         N_gen_T = self.gfmoe_2(x_RNT[1])
        #         w_RT = self.glw_RT[0](x_RNT[0])
        #         w_NT = self.glw_NT[0](x_RNT[1])
        #
        #         w_R = self.softmax(torch.cat((w_NR, w_TR), dim=1))
        #         w_N = self.softmax(torch.cat((w_RN, w_TN), dim=1))
        #         w_T = self.softmax(torch.cat((w_RT, w_NT), dim=1))
        #
        #         x_R = lamda * x_RNT[0] + w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
        #             1) * T_gen_R.squeeze()
        #         x_N = lamda * x_RNT[1] + w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
        #             1) * T_gen_N.squeeze()
        #         x_T = lamda * x_RNT[2] + w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
        #             1) * N_gen_T.squeeze()
        #     else:
        #         if self.miss_type == 'r':
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             w_R = self.softmax(torch.cat((self.glw_NR[0](x_RNT[1]), self.glw_TR[0](x_RNT[2])), dim=1))
        #             x_R = w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
        #                 1) * T_gen_R.squeeze()
        #
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             x_N = lamda * x_RNT[1] + self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
        #             x_T = lamda * x_RNT[2] + self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
        #         elif self.miss_type == 'n':
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             w_N = self.softmax(torch.cat((self.glw_RN[0](x_RNT[0]), self.glw_TN[0](x_RNT[2])), dim=1))
        #             x_N = w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
        #                 1) * T_gen_N.squeeze()
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             x_R = lamda * x_RNT[0] + self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
        #             x_T = lamda * x_RNT[2] + self.softmax(self.glw_RT[0](x_RNT[0])) * R_gen_T.squeeze()
        #         elif self.miss_type == 't':
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             w_T = self.softmax(torch.cat((self.glw_RT[0](x_RNT[0]), self.glw_NT[0](x_RNT[1])), dim=1))
        #             x_T = w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
        #                 1) * N_gen_T.squeeze()
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             x_R = lamda * x_RNT[0] + self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
        #             x_N = lamda * x_RNT[1] + self.softmax(self.glw_RN[0](x_RNT[0])) * R_gen_N.squeeze()
        #         elif self.miss_type == 'rn':
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             x_R = self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
        #             x_N = self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
        #             x_T = x_RNT[2]
        #         elif self.miss_type == 'rt':
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             x_R = self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
        #             x_N = x_RNT[1]
        #             x_T = self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
        #         elif self.miss_type == 'nt':
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             x_R = x_RNT[0]
        #             x_N = self.glw_RN[0](x_RNT[0]) * R_gen_N.squeeze()
        #             x_T = self.glw_RT[0](x_RNT[0]) * R_gen_T.squeeze()
        #     x = torch.cat((x_R, x_N, x_T), dim=0)
 # NLD -> LND

        for i in range(len(self.transformer.resblocks)):
            if i==0:
                if self.gen_lack is True and B > 3:
                    gen_is = 1
                    if i > 0:
                        x = x.permute(1, 0, 2)
                    lamda = 0.5
                    x_RNT = x.chunk(3, dim=0)
                    if self.miss_type == 'nothing' or self.training:
                        N_gen_R = self.gfmoe(x_RNT[1])
                        T_gen_R = self.gfmoe(x_RNT[2])
                        w_NR = self.glw_NR[0](x_RNT[1])
                        w_TR = self.glw_TR[0](x_RNT[2])

                        R_gen_N = self.gfmoe(x_RNT[0])
                        T_gen_N = self.gfmoe_2(x_RNT[2])
                        w_RN = self.glw_RN[0](x_RNT[0])
                        w_TN = self.glw_TN[0](x_RNT[2])

                        R_gen_T = self.gfmoe_2(x_RNT[0])
                        N_gen_T = self.gfmoe_2(x_RNT[1])
                        w_RT = self.glw_RT[0](x_RNT[0])
                        w_NT = self.glw_NT[0](x_RNT[1])

                        w_R = self.softmax(torch.cat((w_NR, w_TR), dim=1))
                        w_N = self.softmax(torch.cat((w_RN, w_TN), dim=1))
                        w_T = self.softmax(torch.cat((w_RT, w_NT), dim=1))

                        x_R = lamda * x_RNT[0] + w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
                            1) * T_gen_R.squeeze()
                        x_N = lamda * x_RNT[1] + w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
                            1) * T_gen_N.squeeze()
                        x_T = lamda * x_RNT[2] + w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
                            1) * N_gen_T.squeeze()

                        # miss TIR
                        # N_gen_R = self.gfmoe(x_RNT[1])
                        # w_NR = self.glw_NR[0](x_RNT[1])
                        #
                        # R_gen_N = self.gfmoe(x_RNT[0])
                        # w_RN = self.glw_RN[0](x_RNT[0])
                        #
                        # w_R = self.softmax(torch.cat((w_NR, ), dim=1))
                        # w_N = self.softmax(torch.cat((w_RN, ), dim=1))
                        #
                        # x_R = lamda * x_RNT[0] + w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze()
                        # x_N = lamda * x_RNT[1] + w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze()
                    else:
                        # x_R = x_RNT[0]
                        # x_N = x_RNT[1]
                        if self.miss_type == 'r':
                            N_gen_R = self.gfmoe(x_RNT[1])
                            T_gen_R = self.gfmoe(x_RNT[2])
                            w_R = self.softmax(torch.cat((self.glw_NR[0](x_RNT[1]), self.glw_TR[0](x_RNT[2])), dim=1))
                            x_R = w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
                                1) * T_gen_R.squeeze()

                            T_gen_N = self.gfmoe_2(x_RNT[2])
                            N_gen_T = self.gfmoe_2(x_RNT[1])
                            x_N = lamda * x_RNT[1] + self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
                            x_T = lamda * x_RNT[2] + self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
                        elif self.miss_type == 'n':
                            R_gen_N = self.gfmoe(x_RNT[0])
                            T_gen_N = self.gfmoe_2(x_RNT[2])
                            w_N = self.softmax(torch.cat((self.glw_RN[0](x_RNT[0]), self.glw_TN[0](x_RNT[2])), dim=1))
                            x_N = w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
                                1) * T_gen_N.squeeze()
                            T_gen_R = self.gfmoe(x_RNT[2])
                            R_gen_T = self.gfmoe_2(x_RNT[0])
                            x_R = lamda * x_RNT[0] + self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
                            x_T = lamda * x_RNT[2] + self.softmax(self.glw_RT[0](x_RNT[0])) * R_gen_T.squeeze()
                        elif self.miss_type == 't':
                            R_gen_T = self.gfmoe_2(x_RNT[0])
                            N_gen_T = self.gfmoe_2(x_RNT[1])
                            w_T = self.softmax(torch.cat((self.glw_RT[0](x_RNT[0]), self.glw_NT[0](x_RNT[1])), dim=1))
                            x_T = w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
                                1) * N_gen_T.squeeze()
                            N_gen_R = self.gfmoe(x_RNT[1])
                            R_gen_N = self.gfmoe(x_RNT[0])
                            x_R = lamda * x_RNT[0] + self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
                            x_N = lamda * x_RNT[1] + self.softmax(self.glw_RN[0](x_RNT[0])) * R_gen_N.squeeze()
                        elif self.miss_type == 'rn':
                            T_gen_R = self.gfmoe(x_RNT[2])
                            T_gen_N = self.gfmoe_2(x_RNT[2])
                            x_R = self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
                            x_N = self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
                            x_T = x_RNT[2]
                        elif self.miss_type == 'rt':
                            N_gen_R = self.gfmoe(x_RNT[1])
                            N_gen_T = self.gfmoe_2(x_RNT[1])
                            x_R = self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
                            x_N = x_RNT[1]
                            x_T = self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
                        elif self.miss_type == 'nt':
                            R_gen_N = self.gfmoe(x_RNT[0])
                            R_gen_T = self.gfmoe_2(x_RNT[0])
                            x_R = x_RNT[0]
                            x_N = self.glw_RN[0](x_RNT[0]) * R_gen_N.squeeze()
                            x_T = self.glw_RT[0](x_RNT[0]) * R_gen_T.squeeze()
                    x = torch.cat((x_R, x_N, x_T), dim=0)
                    # x = torch.cat((x_R, x_N), dim=0)
                    if i > 0:
                        x = x.permute(1, 0, 2)
            if i==0:
                x = self.ln_pre(x)
                x = x.permute(1, 0, 2)
            if self.prompt_sign and self.adapter_sign:
                if i == 0:
                    x, last_prompt = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=True,
                                                                   adapter_sign=True)
                else:
                    x, last_prompt = self.transformer.resblocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                                   adapter_sign=True)
            elif self.prompt_sign and not self.adapter_sign:
                if i == 0:
                    x, last_prompt = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=True,
                                                                   adapter_sign=False)
                else:
                    x, last_prompt = self.transformer.resblocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                                   adapter_sign=False)
            elif not self.prompt_sign and self.adapter_sign:
                x = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=True)
            else:
                x = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=False)
            if getattr(self.transformer.resblocks[i], 'aux_loss') is not None and self.training:
                loss_aux_list = loss_aux_list + self.transformer.resblocks[i].aux_loss
            if self.gen_lack is True and B > 3 and gen_is == 1 and self.training:
                if self.gfmoe.l_aux > 0 and self.gfmoe_2.l_aux > 0:
                    loss_aux_list = loss_aux_list + (self.gfmoe.l_aux + self.gfmoe_2.l_aux) * self.aux_loss_weights
                # if self.gfmoe.l_aux > 0:
                #     loss_aux_list = loss_aux_list + self.gfmoe.l_aux * self.aux_loss_weights

        # if self.gen_lack is True and B > 3:
        #     lamda = 0.5
        #     x_RNT = x.chunk(3, dim=0)
        #     if self.miss_type == 'nothing' or self.training:
        #         N_gen_R = self.gfmoe(x_RNT[1])
        #         T_gen_R = self.gfmoe(x_RNT[2])
        #         w_NR = self.glw_NR[0](x_RNT[1])
        #         w_TR = self.glw_TR[0](x_RNT[2])
        #
        #         R_gen_N = self.gfmoe(x_RNT[0])
        #         T_gen_N = self.gfmoe_2(x_RNT[2])
        #         w_RN = self.glw_RN[0](x_RNT[0])
        #         w_TN = self.glw_TN[0](x_RNT[2])
        #
        #         R_gen_T = self.gfmoe_2(x_RNT[0])
        #         N_gen_T = self.gfmoe_2(x_RNT[1])
        #         w_RT = self.glw_RT[0](x_RNT[0])
        #         w_NT = self.glw_NT[0](x_RNT[1])
        #
        #         w_R = self.softmax(torch.cat((w_NR, w_TR), dim=1))
        #         w_N = self.softmax(torch.cat((w_RN, w_TN), dim=1))
        #         w_T = self.softmax(torch.cat((w_RT, w_NT), dim=1))        #
        #
        #         x_R = lamda * x_RNT[0] + w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
        #             1) * T_gen_R.squeeze()
        #         x_N = lamda * x_RNT[1] + w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
        #             1) * T_gen_N.squeeze()
        #         x_T = lamda * x_RNT[2] + w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
        #             1) * N_gen_T.squeeze()
        #     else:
        #         if self.miss_type == 'r':
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             w_R = self.softmax(torch.cat((self.glw_NR[0](x_RNT[1]), self.glw_TR[0](x_RNT[2])), dim=1))
        #             x_R = w_R[:, 0, :].unsqueeze(1) * N_gen_R.squeeze() + w_R[:, 1, :].unsqueeze(
        #                 1) * T_gen_R.squeeze()
        #
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             x_N = lamda * x_RNT[1] + self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
        #             x_T = lamda * x_RNT[2] + self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
        #         elif self.miss_type == 'n':
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             w_N = self.softmax(torch.cat((self.glw_RN[0](x_RNT[0]), self.glw_TN[0](x_RNT[2])), dim=1))
        #             x_N = w_N[:, 0, :].unsqueeze(1) * R_gen_N.squeeze() + w_N[:, 1, :].unsqueeze(
        #                 1) * T_gen_N.squeeze()
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             x_R = lamda * x_RNT[0] + self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
        #             x_T = lamda * x_RNT[2] + self.softmax(self.glw_RT[0](x_RNT[0])) * R_gen_T.squeeze()
        #         elif self.miss_type == 't':
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             w_T = self.softmax(torch.cat((self.glw_RT[0](x_RNT[0]), self.glw_NT[0](x_RNT[1])), dim=1))
        #             x_T = w_T[:, 0, :].unsqueeze(1) * R_gen_T.squeeze() + w_T[:, 1, :].unsqueeze(
        #                 1) * N_gen_T.squeeze()
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             x_R = lamda * x_RNT[0] + self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
        #             x_N = lamda * x_RNT[1] + self.softmax(self.glw_RN[0](x_RNT[0])) * R_gen_N.squeeze()
        #         elif self.miss_type == 'rn':
        #             T_gen_R = self.gfmoe(x_RNT[2])
        #             T_gen_N = self.gfmoe_2(x_RNT[2])
        #             x_R = self.softmax(self.glw_TR[0](x_RNT[2])) * T_gen_R.squeeze()
        #             x_N = self.softmax(self.glw_TN[0](x_RNT[2])) * T_gen_N.squeeze()
        #             x_T = x_RNT[2]
        #         elif self.miss_type == 'rt':
        #             N_gen_R = self.gfmoe(x_RNT[1])
        #             N_gen_T = self.gfmoe_2(x_RNT[1])
        #             x_R = self.softmax(self.glw_NR[0](x_RNT[1])) * N_gen_R.squeeze()
        #             x_N = x_RNT[1]
        #             x_T = self.softmax(self.glw_NT[0](x_RNT[1])) * N_gen_T.squeeze()
        #         elif self.miss_type == 'nt':
        #             R_gen_N = self.gfmoe(x_RNT[0])
        #             R_gen_T = self.gfmoe_2(x_RNT[0])
        #             x_R = x_RNT[0]
        #             x_N = self.glw_RN[0](x_RNT[0]) * R_gen_N.squeeze()
        #             x_T = self.glw_RT[0](x_RNT[0]) * R_gen_T.squeeze()
        #     x = torch.cat((x_R, x_N, x_T), dim=0)        #
        # if self.gen_lack is True and B > 3 and self.training:
        #     loss_aux_list = loss_aux_list + (self.gfmoe.l_aux + self.gfmoe_2.l_aux) * self.aux_loss_weights

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        if self.proj is not None:
            xproj = x @ self.proj

        return xproj, loss_aux_list


class CLIP(nn.Module):
    def __init__(self, cfg,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int,
                 w_resolution: int
                 ):
        super().__init__()
        self.Adapter = cfg.MODEL.ADA
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution * w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution=h_resolution,
                w_resolution=w_resolution,
                patch_size=vision_patch_size,
                stride_size=vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                cfg=cfg
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), Adapter=self.Adapter)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    if hasattr(attr, 'data'):
                        attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(cfg, state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:  # RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]  # 77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(cfg,
                 embed_dim,
                 image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
                 h_resolution, w_resolution
                 )
    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"],
                                                                     model.visual.positional_embedding, h_resolution,
                                                                     w_resolution,cfg)
    else:  # RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(
            state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding,
            h_resolution, w_resolution)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    try:
        print(f"Successfully load ckpt!")
        incompatibleKeys = model.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)
    except Exception as e:
        print(f"Failed loading checkpoint!")
    return model.eval()


import math


def resize_pos_embed(posemb, posemb_new, hight, width,cfg=None):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)

    ntok_new = posemb_new.shape[0]  # 129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb
