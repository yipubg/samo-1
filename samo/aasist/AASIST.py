"""
AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
======================================================================================

本模块定义了 AASIST 骨干网络，是 SAMO 使用的特征提取器（backbone）。

原始代码来源: https://github.com/clovaai/aasist
版权: Copyright (c) 2021-present NAVER Corp.
许可证: MIT license

网络架构总览:
  输入: 原始音频波形 (batch, 64600) @ 16kHz

  1. SincConv 前端层     → 基于 sinc 函数的可学习滤波器组，直接处理原始波形
  2. 残差编码器 (×6)      → 逐层提取频谱-时间特征
  3. 图注意力层 (GAT-S/T) → 将频域和时域分别建模为图结构
  4. 异构融合层 (HtrgGAT) → 跨维度信息交互（频谱 ↔ 时间）
  5. 读出层               → 全局池化 + 拼接 + 全连接

  输出:
    - last_hidden: 160 维特征向量 (embedding)，用于 SAMO/OC-Softmax 损失函数
    - output:      2 分类 logits，用于 Softmax 损失函数

模型参数量: 约 280 万
"""

import json
import os
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch import Tensor


# ============================================================================
# 图注意力层 (Graph Attention Layer)
# ============================================================================
class GraphAttentionLayer(nn.Module):
    """
    图注意力层 —— 用在频域图（GAT-S）和时域图（GAT-T）上。

    核心思想:
      将输入视为一个全连接图（每个节点与其他所有节点相连），
      通过注意力机制学习节点之间的关系权重，然后加权聚合邻居节点的信息。

    输入:  x, 形状为 (batch_size, num_nodes, in_dim)
    输出:  x, 形状为 (batch_size, num_nodes, out_dim)
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # ---- 注意力映射 ----
        # att_proj: 将节点对的拼接特征投影到 out_dim 维
        self.att_proj = nn.Linear(in_dim, out_dim)
        # att_weight: 将投影后的特征压缩为单个注意力权重
        self.att_weight = self._init_new_params(out_dim, 1)

        # ---- 特征投影 ----
        # 两个投影路径：带注意力的和不带注意力的，最后相加（残差思想）
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # ---- 批归一化 ----
        self.bn = nn.BatchNorm1d(out_dim)

        # ---- 输入 Dropout（防止过拟合）----
        self.input_drop = nn.Dropout(p=0.2)

        # ---- 激活函数 ----
        # SELU: 自归一化激活函数，比 ReLU 更适合深层网络
        self.act = nn.SELU(inplace=True)

        # ---- 温度参数（控制注意力分布的"锐度"）----
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        """
        前向传播:
          1. 对输入应用 Dropout
          2. 计算注意力图（节点间关系矩阵）
          3. 用注意力图聚合邻居信息 + 自身投影 → 残差组合
          4. 批归一化 + SELU 激活
        """
        x = self.input_drop(x)

        # 计算注意力图: (batch, num_nodes, num_nodes, 1)
        att_map = self._derive_att_map(x)

        # 特征投影（注意力聚合 + 直接投影）
        x = self._project(x, att_map)

        # 批归一化 + 激活
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        """
        计算节点对的逐元素乘积。
        用于生成注意力图的输入特征。

        输入:  (batch, num_nodes, dim)
        输出:  (batch, num_nodes, num_nodes, dim)
              位置 [i,j,:] = x[:,i,:] * x[:,j,:]
        """
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)   # 扩展维度
        x_mirror = x.transpose(1, 2)                        # 转置得到所有节点对
        return x * x_mirror                                  # 逐元素乘积

    def _derive_att_map(self, x):
        """
        计算注意力图（节点间的关系权重矩阵）。

        步骤:
          1. 计算节点对的逐元素乘积
          2. 线性投影 + tanh 激活
          3. 与权重向量做内积，得到标量注意力分数
          4. 温度缩放 + softmax（沿源节点维度）

        输出: (batch, num_nodes, num_nodes, 1)
              位置 [i,j,:] 表示节点 j 对节点 i 的注意力权重
        """
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))         # 非线性变换
        att_map = torch.matmul(att_map, self.att_weight)      # 压缩为标量分数

        att_map = att_map / self.temp                         # 温度缩放
        att_map = F.softmax(att_map, dim=-2)                  # 对源节点维度做 softmax

        return att_map

    def _project(self, x, att_map):
        """
        特征投影 —— 将注意力聚合与直接投影相加（残差连接）。
        """
        # 路径1: 注意力聚合——用注意力权重加权求和邻居节点特征
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        # 路径2: 直接投影——保留节点自身的原始信息
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        """
        应用批归一化。
        需要先将 (batch, num_nodes, dim) 展平为 (batch*num_nodes, dim)，
        归一化后再恢复形状。
        """
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        """Xavier 初始化参数（适用于 tanh/SELU 激活函数）"""
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


# ============================================================================
# 异构图注意力层 (Heterogeneous Graph Attention Layer)
# ============================================================================
class HtrgGraphAttentionLayer(nn.Module):
    """
    异构（异质）图注意力层 —— 融合频谱图和时间图的信息。

    核心思想:
      频谱图（Spectral Graph）和时间图（Temporal Graph）是两种不同类型的节点，
      它们构成了一个"异构图"。HtrgGAT 通过注意力机制在这两种节点之间传递信息。

    输入有两种类型:
      - x1: 频谱节点 (spectral nodes)
      - x2: 时间节点 (temporal nodes)

    还有一个可学习的 "主节点 (master node)"，作为全局信息的中转站，
    在频谱和时间维度之间传递跨模态信息。
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # ---- 类型特定的投影（先投影再拼接）----
        self.proj_type1 = nn.Linear(in_dim, in_dim)  # 频谱节点投影
        self.proj_type2 = nn.Linear(in_dim, in_dim)  # 时间节点投影

        # ---- 注意力映射 ----
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        # ---- 注意力权重（分区域学习）----
        # 频谱→频谱、时间→时间、频谱↔时间 分别使用不同的权重
        self.att_weight11 = self._init_new_params(out_dim, 1)  # 频谱内部
        self.att_weight22 = self._init_new_params(out_dim, 1)  # 时间内部
        self.att_weight12 = self._init_new_params(out_dim, 1)  # 跨类型
        self.att_weightM = self._init_new_params(out_dim, 1)   # 主节点

        # ---- 特征投影 ----
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # ---- 批归一化 ----
        self.bn = nn.BatchNorm1d(out_dim)

        # ---- Dropout ----
        self.input_drop = nn.Dropout(p=0.2)

        # ---- 激活函数 ----
        self.act = nn.SELU(inplace=True)

        # ---- 温度 ----
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        """
        前向传播:
          1. 分别投影两种类型的节点
          2. 拼接所有节点
          3. 如果没有主节点，用所有节点的均值作为主节点
          4. 计算分块注意力图（区分同类型和跨类型边）
          5. 更新主节点
          6. 注意力聚合 + 投影
          7. 分离回两种节点类型

        参数:
            x1:     频谱节点 (batch, num_spectral_nodes, dim)
            x2:     时间节点 (batch, num_temporal_nodes, dim)
            master: 主节点   (batch, 1, dim)，如果为 None 则自动初始化

        返回:
            x1, x2, master: 更新后的频谱节点、时间节点和主节点
        """
        num_type1 = x1.size(1)  # 频谱节点数
        num_type2 = x2.size(1)  # 时间节点数

        # 分别投影
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        # 拼接所有节点为一个大的图: [频谱节点 | 时间节点]
        x = torch.cat([x1, x2], dim=1)

        # 初始化主节点（所有节点的均值）
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)

        # 计算分块注意力图
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # 更新主节点（通过注意力聚合全局信息）
        master = self._update_master(x, master)

        # 特征投影
        x = self._project(x, att_map)

        # 批归一化 + 激活
        x = self._apply_BN(x)
        x = self.act(x)

        # 分离回频谱节点和时间节点
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):
        """
        更新主节点 —— 通过注意力机制从所有节点聚合信息到主节点。
        这样主节点可以捕获全局上下文信息。
        """
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        """计算所有节点对的逐元素乘积"""
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        """计算主节点对各个节点的注意力权重"""
        att_map = x * master                           # 节点与主节点的交互
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        """
        计算分块注意力图。

        不同于 GraphAttentionLayer 的统一注意力，这里将注意力矩阵分为 4 个区域:
          - [频谱→频谱]: 频谱节点之间的内部关系
          - [时间→时间]: 时间节点之间的内部关系
          - [频谱→时间]: 跨类型的交互
          - [时间→频谱]: 跨类型的交互（对称）

        每个区域使用独立的注意力权重参数。
        """
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        # 初始化注意力矩阵（全零）
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        # 填充四个子区域
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)   # 频谱→频谱
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)   # 时间→时间
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)   # 频谱→时间
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)   # 时间→频谱（共享权重）

        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)  # 对源节点维度归一化

        return att_map

    def _project(self, x, att_map):
        """特征投影（残差组合）"""
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        """主节点投影（残差组合）"""
        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        """批归一化"""
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        """Xavier 初始化"""
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


# ============================================================================
# 图池化层 (Graph Pooling)
# ============================================================================
class GraphPool(nn.Module):
    """
    图池化层 —— 根据注意力分数选择最重要的节点（Top-K 选择）。

    核心思想:
      不是简单地平均池化，而是通过一个可学习的评分函数为每个节点打分，
      然后保留得分最高的 K% 的节点。

    参数:
        k: 保留节点的比例（0~1 之间的小数）
        in_dim: 输入特征维度
        p: Dropout 概率
    """

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()                         # 将评分映射到 [0,1]
        self.proj = nn.Linear(in_dim, 1)                    # 为每个节点打一个重要性分数
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        """
        前向传播:
          1. Dropout
          2. 计算每个节点的重要性分数
          3. 选择 Top-K 节点
          4. 用分数对保留节点加权
        """
        Z = self.drop(h)
        weights = self.proj(Z)                              # (batch, num_nodes, 1)
        scores = self.sigmoid(weights)                      # 归一化到 [0,1]
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        """
        Top-K 节点选择。

        步骤:
          1. 按分数排序，取前 K 个节点
          2. 用分数对选中的节点特征加权
          3. 收集选中的节点
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)                  # 至少保留 1 个节点
        _, idx = torch.topk(scores, n_nodes, dim=1)         # Top-K 索引
        idx = idx.expand(-1, -1, n_feat)                    # 扩展到特征维度

        h = h * scores                                       # 分数加权
        h = torch.gather(h, 1, idx)                         # 收集选中节点

        return h


# ============================================================================
# SincConv 层 —— 基于 sinc 函数的可学习滤波器
# ============================================================================
class CONV(nn.Module):
    """
    SincConv (Sinc 卷积层) —— AASIST 的第一层，直接处理原始波形。

    核心思想:
      传统卷积的第一层使用随机初始化的卷积核，而 SincConv 使用 sinc 函数
      形式的滤波器，数学上等价于一组带通滤波器。

    与普通卷积的区别:
      - 卷积核形状由 sinc 函数参数化（只有 cutoff 频率可学习）
      - 滤波器的中心频率按梅尔刻度（Mel-scale）分布，贴近人耳听觉特性
      - 参数量更少、更可解释

    输入:  原始波形 (batch, 1, 64600)
    输出:  多频带特征 (batch, 70, T)
    """

    @staticmethod
    def to_mel(hz):
        """Hz → Mel 刻度转换"""
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        """Mel → Hz 刻度转换"""
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 in_channels=1, stride=1, padding=0, dilation=1,
                 bias=False, groups=1, mask=False):
        super().__init__()

        # SincConv 要求单通道输入
        if in_channels != 1:
            raise ValueError(f"SincConv only support one input channel (here, in_channels = {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # 确保卷积核为奇数（完全对称）
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        # ---- 按梅尔刻度分布设计滤波器频带 ----
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        # 在 Mel 域等间距划分频带
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)      # 转回 Hz

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)

        # ---- 为每个频带设计理想的带通滤波器（基于 sinc 函数）----
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            # 理想低通 = sinc 函数
            hHigh = (2 * fmax / self.sample_rate) * \
                    np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * \
                   np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow                            # 带通 = 高截止低通 - 低截止低通

            # 加汉明窗减少频谱泄漏
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        """
        前向传播。

        参数:
            x:    输入波形 (batch, 1, num_samples)
            mask: 是否随机屏蔽某些频带（频域数据增强）
        """
        band_pass_filter = self.band_pass.clone().to(x.device)

        # 随机频带屏蔽（数据增强）
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0              # 屏蔽连续的 A 个频带

        # 重塑为卷积核形状: (out_channels, in_channels, kernel_size)
        self.filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


# ============================================================================
# 残差块 (Residual Block)
# ============================================================================
class Residual_block(nn.Module):
    """
    残差卷积块 —— 构成编码器的主要组件。

    结构:
      输入 → BN → SELU → Conv2d → BN → SELU → Conv2d → +输入 → MaxPool2d

    包含两个 2D 卷积层（核大小 2×3），配合批归一化和 SELU 激活。
    如果输入和输出通道数不同，通过 1×1 卷积进行维度匹配。
    """

    def __init__(self, nb_filts, first=False):
        """
        参数:
            nb_filts: [in_channels, out_channels] 格式的列表
            first:    是否为第一层（第一层不需要前置 BN）
        """
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        # 如果输入输出通道数不同，需要 1×1 卷积做维度匹配
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)
        else:
            self.downsample = False

        # 最大池化，只在时间维度上降采样
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        # 残差连接
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity

        # 时间维度下采样
        out = self.mp(out)
        return out


# ============================================================================
# AASIST 主模型
# ============================================================================
class Model(nn.Module):
    """
    AASIST 主模型 —— 端到端语音防伪检测网络。

    输入:  原始音频波形 (batch, 64600) @ 16kHz
    输出:
        - last_hidden: (batch, 160) 的特征向量 (embedding)
        - output:      (batch, 2) 的分类 logits

    数据流:
        raw audio → SincConv → 残差编码器 → GAT-S/GAT-T →
        双分支异构融合 → 全局池化 → 拼接 → 全连接 → 输出
    """

    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        # ---- 第1层: SincConv 前端 ----
        # 输入 (batch, 1, 64600) → 输出 (batch, 70, T)
        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # ---- 第2层: 残差编码器（6个残差块）----
        # 通道数变化: 70→32→32→64→64→64→64
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts[4])),     # 相同通道数，深化网络
            nn.Sequential(Residual_block(nb_filts[4])))     # 相同通道数，深化网络

        # ---- 位置编码（可学习，用于保留频域/时域的位置信息）----
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))  # 频域位置编码
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))   # 分支1 主节点
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))   # 分支2 主节点

        # ---- 第3层: 独立的频域和时域图注意力 ----
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        # ---- 第4层: 双分支异构融合 ----
        # 分支1（HtrgGAT 层对 11+12）
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # 分支2（HtrgGAT 层对 21+22）
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # ---- 图池化层（逐步减少节点数量）----
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # ---- 第5层: 读出层 ----
        # 5个 32 维特征的拼接: T_max, T_avg, S_max, S_avg, master = 5*32 = 160
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)  # 2分类输出

    def forward(self, x, Freq_aug=False):
        """
        前向传播。

        参数:
            x:        输入音频 (batch, 64600)
            Freq_aug: 是否启用频域数据增强（随机屏蔽频带）

        返回:
            last_hidden: (batch, 160) embedding 特征向量
            output:      (batch, 2) 分类 logits
        """
        # ---- 阶段1: SincConv 前端处理 ----
        x = x.unsqueeze(1)                       # (batch, 64600) → (batch, 1, 64600)
        x = self.conv_time(x, mask=Freq_aug)     # SincConv: → (batch, 70, T)
        x = x.unsqueeze(dim=1)                   # → (batch, 1, 70, T)
        x = F.max_pool2d(torch.abs(x), (3, 3))   # 取绝对值 + 池化（提取能量特征）
        x = self.first_bn(x)
        x = self.selu(x)

        # ---- 阶段2: 残差编码器 ----
        e = self.encoder(x)                      # → (batch, 64, F, T)

        # ---- 阶段3: 频域图注意力 (GAT-Spectral) ----
        # 沿时间维度取最大值 → 得到频域表示
        e_S, _ = torch.max(torch.abs(e), dim=3)  # (batch, 64, F)
        e_S = e_S.transpose(1, 2) + self.pos_S   # (batch, F, 64) + 位置编码
        gat_S = self.GAT_layer_S(e_S)            # 图注意力
        out_S = self.pool_S(gat_S)               # 图池化

        # ---- 阶段4: 时域图注意力 (GAT-Temporal) ----
        # 沿频域维度取最大值 → 得到时域表示
        e_T, _ = torch.max(torch.abs(e), dim=2)  # (batch, 64, T)
        e_T = e_T.transpose(1, 2)                # (batch, T, 64)
        gat_T = self.GAT_layer_T(e_T)            # 图注意力
        out_T = self.pool_T(gat_T)               # 图池化

        # ---- 阶段5: 双分支异构融合 ----
        # 复制主节点用于两个并行分支
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # --- 分支1 ---
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        # 残差连接 + 第二次异构融合
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug   # 残差连接
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # --- 分支2（与分支1 结构相同，参数独立）---
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # ---- 阶段6: Dropout（防止过拟合）----
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # ---- 阶段7: 双分支 max 融合 ----
        out_T = torch.max(out_T1, out_T2)   # 取两分支的最大值
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # ---- 阶段8: 全局池化 + 拼接 ----
        T_max, _ = torch.max(torch.abs(out_T), dim=1)  # 时间特征的最大池化
        T_avg = torch.mean(out_T, dim=1)                # 时间特征的平均池化
        S_max, _ = torch.max(torch.abs(out_S), dim=1)  # 频谱特征的最大池化
        S_avg = torch.mean(out_S, dim=1)                # 频谱特征的平均池化

        # 拼接 5 个特征向量 → 160 维 embedding
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        # ---- 阶段9: 分类输出 ----
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)  # → (batch, 2)

        return last_hidden, output


# ============================================================================
# 测试入口（仅用于打印模型结构和参数量）
# ============================================================================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    with open("AASIST.conf", "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    feat_model = Model(model_config)

    nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
    print("no. model params:{}".format(nb_params))

    # 打印模型结构和每层输出形状
    print(summary(feat_model, torch.randn((64, 120000)), show_input=False))

    raw = torch.randn(64, 120000)
    feat, out = feat_model(raw)
    print("feat shape", feat.shape)
    print("out shape", out.shape)
