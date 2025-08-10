# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn

# 如果 x 是
def make_2tuple(x):
    if isinstance(x, tuple):  # 如果输入已经是元组
        assert len(x) == 2  # 确保长度为2
        return x  # 直接返回

    assert isinstance(x, int)  # 否则检查是否为整数
    return (x, x)  # 转换为(x, x)形式的元组

# [b,c,H,W] -> [b,HW,c]
class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        # 根据图像 size 和 patch size，可以分为多少各 path 块 (向下取整 )
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        #
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        # 每个 patch 编码多长的线性特征长度
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        # 对每一个 patch 做全卷积，生成 1维的编码 [1,3,518,518] -> [14×14×3,224/16,224/16] = [1,768,,16,16]
        # default: Conv2d(kernel=[1,3,518,518],embed_dim=768, kernel_size=[14,14],stride=14)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    # 这里接收到的输入为 [b,c,H,W] 的原始图像数据
    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        # x.flatten(2) 保留第0维(B)和第1维(C)，将最后两个维度(H,W)展平
        # [B, C, H, W] → [B, HW, C]
        x = x.flatten(2).transpose(1, 2)  # B HW C
        # 表示对输入张量 x 进行标准化 ，如 nn.LayerNorm、nn.BatchNorm 等
        # default: None
        x = self.norm(x)
        # 是否对序列编码反序列化，（默认不执行）
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    # 用于计算 Patch Embedding 层的理论浮点运算量
    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
