# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py


from torch import nn

# 在训练深度神经网络时随机跳过某些层，从而提升模型的泛化能力和训练效率 ，在预测时使用所有的神经元
# 具体实现为： 训练时以一定的概率对 x 中的元素随机置 0，x = 0 处对应的后续链路的参数不再被更新，所以称为 DropPath
# drop_prob:
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    # 如果不需要dropout或处于推理模式，直接返回输入
    if drop_prob == 0.0 or not training:
        return x
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 准备与输入x兼容的形状：(batch_size, 1, 1, ...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成随机二值掩码（Bernoulli分布,伯努利分布）
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # 缩放输出以保持期望值不变，输出的期望值(x * random_tensor 运算导致)比原始输入小了(keep_prob < 0) 倍，会导致网络层间数值尺度缩放，（相当于参与更新的神经元少了，需要提高剩余神经元的贡献度，保持数值稳定）
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    # 应用掩码
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
