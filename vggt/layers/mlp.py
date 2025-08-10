# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn

# 多层感知机（MLP）模块 , 非线性特征变换：通过两层线性变换+激活函数增强模型表达能力
# Linear → Activation → Dropout → Linear → Dropout
# 激活函数选择：
# GELU	平滑梯度	计算量稍大
# ReLU	计算快	神经元死亡 当更新梯度过大时(例如使用了较大的学习率),权重衰减到负区间，此时激活函数的导数为0，神经元的权重不再更新，一般可通过 BatchNorm 进行区间调整
# Swish	平滑性更好	实现复杂

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)  # [B, N, D] -> [B, N, 4D]
        x = self.act(x)  # 非线性变换
        x = self.drop(x)  # 随机置零
        x = self.fc2(x)  # [B, N, 4D] -> [B, N, D]
        x = self.drop(x)
        return x
