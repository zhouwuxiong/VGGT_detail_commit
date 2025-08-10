# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def position_grid_to_embed(
        pos_grid: torch.Tensor,
        embed_dim: int,
        omega_0: float = 100
) -> torch.Tensor:
    """
    将2D坐标网格转换为正弦余弦位置编码（Sinusoidal Positional Embeddings）

    数学原理：
    对每个坐标轴(x,y)独立生成频率递增的正弦/余弦函数组合：
        PE(pos, 2i) = sin(pos / omega_0^(2i/D))
        PE(pos, 2i+1) = cos(pos / omega_0^(2i/D))
    其中 D 是编码维度，omega_0 是基础频率系数

    Args:
        pos_grid: 2D坐标网格，形状为 (H, W, 2)，取值范围建议归一化到[-1,1]
        embed_dim: 输出嵌入的通道维度（必须为偶数）
        omega_0: 控制频率衰减速率的基础参数（默认100，值越大高频成分越多）

    Returns:
        形状为 (H, W, embed_dim) 的位置编码张量

    典型应用场景：
    - Vision Transformer 的位置编码
    - 时空动作定位中的位置感知特征
    - 图像生成任务的坐标条件输入
    """
    # 1. 输入验证与预处理
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2, f"输入坐标维度必须为2，实际得到 {grid_dim}"
    assert embed_dim % 2 == 0, f"embed_dim 必须是偶数，以便平分给x/y坐标"

    # 展平坐标网格 -> [H*W, 2]
    pos_flat = pos_grid.reshape(-1, grid_dim)  # 支持自动梯度计算

    # 2. 分别处理x/y坐标
    # 每个坐标轴分配 embed_dim//2 的维度
    emb_x = make_sincos_pos_embed(
        dim=embed_dim // 2,
        pos=pos_flat[:, 0],  # 取所有x坐标
        omega_0=omega_0
    )  # 输出形状 [1, H*W, D/2]

    emb_y = make_sincos_pos_embed(
        dim=embed_dim // 2,
        pos=pos_flat[:, 1],  # 取所有y坐标
        omega_0=omega_0
    )  # 输出形状 [1, H*W, D/2]

    # 3. 合并x/y编码并恢复空间维度
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    # 重塑为原始空间布局 [H, W, D]
    return emb.view(H, W, embed_dim)


def make_sincos_pos_embed(
        embed_dim: int,
        pos: torch.Tensor,
        omega_0: float = 100
) -> torch.Tensor:
    """
    生成基于正弦余弦函数的1D位置编码（Sinusoidal Positional Embedding）

    数学原理：
    对给定位置pos生成D维编码，其中偶数索引为sin函数，奇数索引为cos函数：
        PE(pos, 2i)   = sin(pos / omega_0^(2i/D))
        PE(pos, 2i+1) = cos(pos / omega_0^(2i/D))
    频率随维度增加呈指数衰减，形成多尺度位置表示

    Args:
        embed_dim: 输出嵌入维度（必须为偶数）
        pos: 输入位置张量，形状任意（会被展平）
        omega_0: 基础频率参数（默认100，值越大高频成分越多）

    Returns:
        形状为 (M, embed_dim) 的位置编码矩阵，其中 M = pos.numel()

    设计要点：
    1. 支持自动微分，可端到端训练
    2. 数值稳定（通过omega_0控制频率范围）
    3. 设备感知（自动匹配输入设备）

    典型应用：
    - Transformer的位置编码
    - 时空序列的位置感知特征
    - 坐标条件生成模型
    """
    # 1. 输入验证
    assert embed_dim % 2 == 0, f"embed_dim必须为偶数，当前为{embed_dim}"

    # 2. 设备与数据类型处理（特别适配Apple MPS）
    device = pos.device
    # MPS设备暂不支持float64，其他设备使用double精度保证数值稳定性
    dtype = torch.float32 if device.type == "mps" else torch.double

    # 3. 生成频率序列
    # ----------------------------------------
    # 创建 [0, 1, ..., D//2-1] 的序列，然后归一化到[0,2]
    omega = torch.arange(embed_dim // 2, dtype=dtype, device=device)
    omega /= (embed_dim / 2.0)  # 归一化到 [0, 2] 区间

    # 计算频率衰减系数：1/(omega_0^(i/(D/2)))
    # 形成几何级数，频率随i增加而降低
    omega = 1.0 / (omega_0 ** omega)  # 形状 [D/2]

    # 4. 位置与频率的外积
    # ----------------------------------------
    pos = pos.reshape(-1).to(dtype=dtype)  # 展平并转换数据类型 [M]
    # 计算所有位置与频率的组合相位 [M, D/2]
    out = torch.einsum("m,d->md", pos, omega)  # 等价于 pos[:, None] * omega[None, :]

    # 5. 生成正弦余弦编码
    # ----------------------------------------
    emb_sin = torch.sin(out)  # 奇数维度 [M, D/2]
    emb_cos = torch.cos(out)  # 偶数维度 [M, D/2]

    # 6. 合并编码并返回float32精度
    # ----------------------------------------
    # 交替拼接sin和cos：sin1,cos1, sin2,cos2, ...
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # [M, D]

    # 转换为float32节省显存（保持足够精度）
    return emb.float()


# Inspired by https://github.com/microsoft/moge


# 将 W*H 的网格编码转换为 0~1 范围内的坐标编码，保持数值稳定
def create_uv_grid(
        width: int,
        height: int,
        aspect_ratio: float = None,
        dtype: torch.dtype = None,
        device: torch.device = None
) -> torch.Tensor:
    """
    生成归一化的UV坐标网格，形状为 (width, height, 2)。
    网格坐标根据宽高比进行归一化，确保：
    - 左上角坐标为 (-x_span, -y_span)
    - 右下角坐标为 (x_span, y_span)
    所有坐标点落在单位对角线圆内（对角线长度为1）。

    参数：
        width (int): 水平方向的网格点数（如图像宽度对应的patch数）
        height (int): 垂直方向的网格点数（如图像高度对应的patch数）
        aspect_ratio (float, optional): 宽高比（width/height）。默认根据width/height自动计算
        dtype (torch.dtype, optional): 输出张量的数据类型
        device (torch.device, optional): 输出张量的设备位置

    返回：
        torch.Tensor: 形状为 (width, height, 2) 的UV坐标张量，
                     最后一维包含 (u,v) 坐标对
    """
    # 1. 计算宽高比（如果未显式提供）
    # ----------------------------------------
    if aspect_ratio is None:
        # 默认使用网格尺寸的比例作为宽高比
        aspect_ratio = float(width) / float(height)

    # 2. 计算归一化的X/Y轴跨度
    # ----------------------------------------
    # 数学原理：保持对角线长度=1的归一化
    # 对于矩形区域：对角线长度 d = sqrt((a*w)^2 + h^2)
    # 其中 a 是宽高比，w和h是网格尺寸
    # 归一化后：span_x = a / sqrt(a^2 + 1), span_y = 1 / sqrt(a^2 + 1)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5  # 计算归一化分母（对角线因子）
    span_x = aspect_ratio / diag_factor  # X轴归一化跨度（保持原始宽高比）
    span_y = 1.0 / diag_factor  # Y轴归一化跨度

    # 3. 计算线性空间的边界
    # ----------------------------------------
    # 设计目标：
    # - 网格中心始终在(0,0)
    # - 边缘对称（无论网格点数为奇偶）
    # 数学推导：
    # 对于N个点，坐标范围应为 [-(N-1)/N * span, (N-1)/N * span]
    left_x = -span_x * (width - 1) / width  # X轴左边界
    right_x = span_x * (width - 1) / width  # X轴右边界
    top_y = -span_y * (height - 1) / height  # Y轴上边界
    bottom_y = span_y * (height - 1) / height  # Y轴下边界

    # 4. 生成一维坐标序列
    # ----------------------------------------
    # 在计算出的边界之间生成等间距坐标点
    x_coords = torch.linspace(
        start=left_x,
        end=right_x,
        steps=width,  # 点数=width
        dtype=dtype,  # 继承输入数据类型
        device=device  # 保持设备一致
    )
    y_coords = torch.linspace(
        start=top_y,
        end=bottom_y,
        steps=height,
        dtype=dtype,
        device=device
    )

    # 5. 创建2D网格并堆叠为UV坐标
    # ----------------------------------------
    # torch.meshgrid 生成网格坐标矩阵
    # indexing="xy" 表示：
    # - 第一个输出（uu）沿width方向变化（水平坐标）
    # - 第二个输出（vv）沿height方向变化（垂直坐标）
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")

    # 沿最后一个维度堆叠，形成 (width, height, 2) 的张量
    # uu: 水平坐标矩阵（从左到右递增）
    # vv: 垂直坐标矩阵（从上到下递增）
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid
