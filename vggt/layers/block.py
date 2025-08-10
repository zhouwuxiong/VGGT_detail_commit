# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # mlp_ratio = 4
        # in: [B,WH,C] -> hiden: [B,WH,4C] -> out: [B,WH,C]
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
    # input [B,HW,C]
    def forward(self, x: Tensor, pos=None) -> Tensor:
        #  带位置编码的注意力残差计算函数， 完成 归一化→注意力计算→残差连接 的标准流程
        # LayerNorm(前注意力归一化) -> MemEffAttention -> LayerScale (残差计算)
        # 前注意力归一化 可以防止输入值域波动过大，使得注意力层的梯度更平滑 ，与 pose-LN 相比 Pre-LN 通常训练更稳定，适合深层网络
        # pos ??? #todo
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))
        # 前馈网络（FFN）的残差计算函数
        # LayerNorm(后注意力归一化) -> mlp (前馈神经网络) -> LayerScale
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            # option 1：   随机丢弃 batch 残差
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        # 以一定的概率对 x 中的元素随机置 0，x = 0 处对应的后续链路的参数不再被更新，所以称为 DropPath
        elif self.training and self.sample_drop_ratio > 0.0:
            # option 2： 随机丢弃 path 路径
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            # option 3: 直接叠残差
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x

# 通过随机丢弃一部分样本的残差计算来减少计算量并实现正则化效果
# 这种实现特别适合大batch训练的场景，通过随机减少每个batch的实际计算量，既能加速训练又能获得正则化效果
def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    # 计算保留的样本数量，（至少保留1个样本）
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    # 通过torch.randperm生成随机排列并选取前sample_subset_size个索引
    # torch.randperm(n) 会生成一个从 0 到 n-1 的随机排列的整数序列，返回一个一维张量
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        # 如果提供位置编码pos，同样对位置编码进行子集选择
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    # 将输入和残差展平为2D张量 (B, WH*C)
    # x.flatten(1) 在第1维进行拼接，即 WH 维
    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    # 计算缩放因子（保持梯度期望值不变）
    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    # 数学等价于： x_flat[brange] += residual * alpha
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x) # 对最后一维进行拆分，恢复输入的 shape

def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}

# 可变长度序列的注意力计算，为不同长度的输入序列创建块对角注意力掩码
# 处理序列或图像块时），branges 是一个 批量范围索引列表，用于高效地选择和重组不同来源或长度的输入数据 ， 建立跨模态对应
# x_list = [[B,WH,C],[B2,WH2,C2],...],用于处理多模态数据输入
def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    # [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    # 等价于下面的形式
    # batch_sizes = []
    # 如果指定了数据选择索引，则只处理被索引选择的数据，否则处理所有的数据
    # if branges is not None:
    #     for b in branges:
    #         batch_sizes.append(b.shape[0])
    # else:
    #     for x in x_list:
    #         batch_sizes.append(x.shape[0])
    # 获取不同输入（例如不同视角图像）的 batch_size 大小
    batch_sizes = [b.shape[0] for b in branges]  if branges is not None else [x.shape[0] for x in x_list]
    # 生成 batch_sizes 与 list 的对应元组，所有的元组组层一个整体的 key
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))

    # 如果cache中找到没有同样形状的输入
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        # 添加不同数据类型的长度
        for b, x in zip(batch_sizes, x_list):
            # 同一种输入数据类型的 batch 具有相同的长度
            for _ in range(b):
                seqlens.append(x.shape[1])
        # fmha.BlockDiagonalMask.from_seqlens(seqlens) 是 FlashAttention 库中用于高效处理可变长度序列的关键功能，其核心作用是生成块对角注意力掩码
        # seqlens 是一个整数列表，表示每个序列的实际长度（不含填充）
        # 对于 seqlens = [2, 3] ，其输出如下，块对角注意力掩码，保证了同类型数据内部的自注意力，而不同类型的数据之间不产生自注意力（需要注意的是 batch 内部不同的样本之间的自注意力也是隔离的 ？？？#todo）
        # [[1 1 0 0 0]
        #  [1 1 0 0 0]
        #  [0 0 1 1 1]
        #  [0 0 1 1 1]
        #  [0 0 1 1 1]]
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        # 将多个张量按特定索引选择后拼接，并通过维度变换得到目标形状
        # view(1, -1, x_list[0].shape[-1])  ，保持通道数不变： x_list[0].shape[-1] ， 中间维度进行拉伸： -1 ， 扩展1维的 batch 维度： 1
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        # *x.shape[2:]：保持原始张量第2维之后的所有维度‘
        # [[B,WH,C],[B2,WH2,C2],...] -> ([1,B*WH,C],[1,B2*WH2,C2],...)
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        # [[B,WH,C],[B2,WH2,C2],...] -> [1,B*WH + B2*WH2 + ...,C] 注意不同数据类型的长度不一样，但是通道数是一样的，即 C=C1=C2=Cn
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensor


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            # 直接叠加残差
            # final_scores = attn_scores + attn_bias
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
