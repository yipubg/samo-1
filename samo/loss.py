"""
SAMO 损失函数模块
================
本模块实现了语音伪造检测中两种核心的损失函数，是论文的核心理论贡献所在。

1. OCSoftmax: 单中心单类学习基线方法（对比方法）
2. SAMO: 说话人吸引力子多中心单类学习（论文提出的创新方法）

核心思想：
  单类学习（One-Class Learning）——模型只学习"什么是真实语音"，
  因为伪造方法千变万化、无法穷举。真实语音必须靠近其对应说话人的中心向量，
  伪造语音则被推离所有中心向量。

参考文献:
  - OC-Softmax: Zhang et al., "One-Class Learning Towards Synthetic Voice Spoofing Detection", SPL 2021
  - SAMO: Ding et al., "SAMO: Speaker Attractor Multi-Center One-Class Learning", ICASSP 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import uniform_hypersphere


class OCSoftmax(nn.Module):
    """
    OC-Softmax (One-Class Softmax) 损失函数
    ---------------------------------------
    这是 SAMO 论文的对比基线方法。
    只使用一个全局中心向量来代表所有真实说话人的特征分布。

    核心思想：
      真实语音（bonafide, label=0）与中心向量的余弦相似度必须大于 m_real；
      伪造语音（spoof, label=1）与中心向量的余弦相似度必须小于 m_fake。
      违反这些约束的样本会产生损失，从而推动模型学习。

    与 SAMO 的区别：
      OC-Softmax 只有 1 个中心（说话人无关），SAMO 有多个中心（说话人感知）。
    """

    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0,
                 fix_centers=True, initialize_centers="one_hot"):
        """
        参数:
            feat_dim: 特征向量的维度（与 AASIST 输出的 embedding 维度一致，默认 160）
            m_real:   真实语音的 margin。bonafide 与中心的相似度需高于此值
            m_fake:   伪造语音的 margin。spoof 与中心的相似度需低于此值
            alpha:    损失函数的缩放因子（温度参数），越大惩罚越"硬"
            fix_centers: 是否固定中心向量不参与梯度更新
            initialize_centers: 中心初始化方式
                - "one_hot": 取单位矩阵第一行作为初始中心（默认）
                - "random":  随机初始化
        """
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha

        # ---- 初始化中心向量 ----
        if initialize_centers == "one_hot":
            # 取单位矩阵的第一行，形状为 (1, feat_dim)
            # 确保初始中心是一个标准基向量
            self.center = nn.Parameter(
                torch.eye(self.feat_dim)[:1],
                requires_grad=not fix_centers
            )
        elif initialize_centers == "random":
            # 从标准正态分布随机采样
            self.center = nn.Parameter(
                torch.randn(1, self.feat_dim),
                requires_grad=not fix_centers
            )

        # Softplus: 平滑的 ReLU，用于将 margin 偏差转换为正的损失值
        # Softplus(x) = log(1 + exp(x))，当 x<0 时趋近于 0，x>0 时近似等于 x
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        前向传播：计算 OC-Softmax 损失和预测分数。

        计算步骤:
            1. 将特征向量和中心向量都归一化到单位超球面
            2. 计算余弦相似度 scores = x @ w^T
            3. 根据标签应用 margin 约束
            4. 通过 Softplus 计算损失

        参数:
            x:      特征矩阵，形状 (batch_size, feat_dim)
            labels: 标签向量，形状 (batch_size,) —— 0=bonafide, 1=spoof

        返回:
            loss:        标量损失值
            output_scores: 每个样本的预测分数（用于计算 EER）
        """
        # 步骤1: L2 归一化，将所有向量映射到单位超球面
        w = F.normalize(self.center, p=2, dim=1)   # 中心向量归一化
        x = F.normalize(x, p=2, dim=1)              # 特征向量归一化

        # 步骤2: 计算每个样本与唯一中心的余弦相似度
        # scores 形状: (batch_size, 1)
        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()  # 保存原始分数用于评估

        # 步骤3: 应用 margin 约束
        #   bonafide (label=0): adjusted = m_real - score
        #     → 如果 score > m_real，adjusted < 0，不产生损失 ✓
        #     → 如果 score < m_real，adjusted > 0，产生损失 ✗
        #   spoof (label=1): adjusted = score - m_fake
        #     → 如果 score < m_fake，adjusted < 0，不产生损失 ✓
        #     → 如果 score > m_fake，adjusted > 0，产生损失 ✗
        scores[labels == 0] = self.m_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.m_fake

        # 步骤4: Softplus(alpha * adjusted) 将偏差转换为损失
        # alpha 越大，损失函数在 margin 边界处越陡峭
        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


class SAMO(nn.Module):
    """
    SAMO (Speaker Attractor Multi-Center One-Class Learning) 损失函数
    ---------------------------------------------------------------
    论文提出的创新方法，是 loss.py 乃至整个项目的核心贡献。

    三个核心设计:
      1. Speaker Attractor（说话人吸引力子）:
         为每个说话人学习一个独立的中心向量，作为其在特征空间中的"锚点"。
         真实语音的特征会被吸引到对应说话人中心附近。

      2. Multi-Center（多中心）:
         使用多个中心向量（默认 20 个，对应训练集的 20 个说话人），
         相比 OC-Softmax 的单中心设计，能更精确地捕捉每个说话人的特征分布。

      3. One-Class Learning（单类学习）:
         模型只学习"什么是真实的"，不学习"什么是伪造的"。
         因为伪造类型无限多样，无法穷举。

    两种评分模式:
      - attractor=1 (说话人感知/SIM):
         真实语音用一对一相似度（样本 vs 其对应说话人中心）
      - attractor=0 或 2 (说话人无关/MAXSCORE):
         所有样本都用 maxscore（样本 vs 所有中心的最大相似度）
    """

    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0,
                 num_centers=20, initialize_centers="one_hot", addNegEntropy=False):
        """
        参数:
            feat_dim:     特征向量维度（与 AASIST embedding 一致，默认 160）
            m_real:       真实语音的 margin 下界
            m_fake:       伪造语音的 margin 上界
            alpha:        损失缩放因子（温度参数）
            num_centers:  中心向量数量（对应说话人数，默认 20）
            initialize_centers: 中心初始化方式
                - "one_hot": 取单位矩阵的前 num_centers 行（默认，保证初始中心正交）
                - "evenly":  在超球面上均匀采样
            addNegEntropy: 是否添加负熵正则项（鼓励中心被均匀使用）
        """
        super(SAMO, self).__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha

        # ---- 初始化多个中心向量 ----
        if initialize_centers == "one_hot":
            # 取单位矩阵的前 num_centers 行作为初始中心
            # 好处：初始中心两两正交（余弦相似度为 0），互不干扰
            # 为每个说话人提供独立的"锚点空间"，加速收敛
            self.center = torch.eye(self.feat_dim)[:self.num_centers]
        elif initialize_centers == "evenly":
            # 在 160 维超球面上均匀采样 20 个点
            # 保证初始中心在空间中分布尽可能分散
            pts = np.array(uniform_hypersphere(self.feat_dim, self.num_centers))
            self.center = torch.from_numpy(pts).float()

        self.softplus = nn.Softplus()  # 平滑版 ReLU
        self.addNegEntropy = addNegEntropy
        if self.addNegEntropy:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, labels, spk=None, enroll=None, attractor=0):
        """
        前向传播：计算 SAMO 损失和预测分数。

        计算步骤:
            1. L2 归一化特征向量和中心向量
            2. 计算余弦相似度矩阵 (batch_size × num_centers)
            3. 获取每个样本的 maxscore（与最近中心的相似度）
            4. 根据 attractor 模式计算最终分数
            5. 应用 margin 约束，通过 Softplus 计算损失

        参数:
            x:        特征矩阵，形状 (batch_size, feat_dim)
            labels:   标签向量，形状 (batch_size,) —— 0=bonafide, 1=spoof
            spk:      说话人 ID 列表，长度 batch_size
            enroll:   注册中心字典 {speaker_id: center_vector}，每个值为 (feat_dim,) 的张量
            attractor: 评分模式选择
                - 0: 说话人独立（直接用 self.center）
                - 1: 说话人感知（一对一相似度，bonafide 用对应说话人中心）
                - 2: 说话人无关（全部用 maxscore）

        返回:
            loss:         标量损失值
            final_scores: 每个样本的预测分数 (batch_size,)
        """
        # 步骤1: L2 归一化
        # 将特征向量和中心向量都归一化到单位超球面
        # 此时内积就等于余弦相似度
        x = F.normalize(x, p=2, dim=1)                                     # (batch, feat_dim)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)              # (num_centers, feat_dim)

        # 步骤2: 计算余弦相似度矩阵
        # scores[i][j] = 第i个样本与第j个中心的余弦相似度
        scores = x @ w.transpose(0, 1)                                      # (batch, num_centers)

        # 步骤3: 获取 maxscore —— 每个样本与所有中心的最大相似度
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)              # (batch, 1)

        # 步骤4: 根据 attractor 模式计算最终分数
        if attractor == 1:
            # ----- 说话人感知模式 (Speaker-Aware) -----
            # bonafide 样本：使用与"其对应说话人"中心的一对一相似度
            # 这是 SAMO 的核心创新：不仅判断"真假"，还判断"像不像这个人"
            tmp_w = torch.stack([enroll[id] for id in spk])                # (batch, feat_dim)
            tmp_w = F.normalize(tmp_w, p=2, dim=1).to(x.device)
            final_scores = torch.sum(x * tmp_w, dim=1).unsqueeze(-1)       # (batch, 1) 一对一相似度

            # 对 bonafide 样本使用一对一分数来调整 margin
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # ----- 说话人无关模式 (Speaker-Agnostic) -----
            # 所有样本都使用 maxscore
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        # 步骤5: 对 spoof 样本应用 margin
        # spoof 的分数必须低于 m_fake，否则产生损失
        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake

        # 步骤6: 计算损失
        # Softplus(alpha * adjusted) 将 margin 偏差转为正的损失
        # alpha=20 使惩罚在 margin 边界附近快速上升
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        # 可选的负熵正则项
        # 目的：鼓励 bona 样本均匀分布在各个中心上，避免所有样本只聚集到某几个中心
        if self.addNegEntropy:
            scores = self.softmax(scores[labels == 0])   # 仅对 bonafide 样本计算 softmax
            p = scores.sum(0).view(-1)                    # 每个中心被"使用"的总概率
            p /= p.sum()                                   # 归一化
            dist_loss = np.log(w.shape[0]) + (p * p.log()).sum()  # 负熵 = log(K) + Σ p*log(p)
            loss = dist_loss * 1e5 + emb_loss              # 加权组合
        else:
            loss = emb_loss

        return loss, final_scores.squeeze(1)

    def inference(self, x, labels, spk, enroll, attractor=0):
        """
        推理模式的前向传播。
        与 forward() 的区别在于：推理时能处理"没有注册中心"的说话人。

        对于在 enroll 中找不到对应中心的说话人，回退到使用 maxscore。

        参数:
            x:        特征矩阵 (batch_size, feat_dim)
            labels:   标签向量 (batch_size,)
            spk:      说话人 ID 列表
            enroll:   注册中心字典 {speaker_id: center_vector}
            attractor: 评分模式 (0/1/2)

        返回:
            loss:         标量损失值
            final_scores: 预测分数 (batch_size,)
        """
        # 归一化
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)

        # 计算相似度矩阵
        scores = x @ w.transpose(0, 1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)

        if attractor == 1:
            # 说话人感知模式：替换有注册中心的样本的分数
            final_scores = maxscores.clone()
            for idx in range(len(spk)):
                if spk[idx] in enroll:
                    # 用一对一相似度替换 maxscore
                    tmp_w = F.normalize(enroll[spk[idx]], p=2, dim=0).to(x.device)
                    final_scores[idx] = x[idx] @ tmp_w

            # 对 bonafide 应用 margin
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # 说话人无关模式
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        # 对 spoof 应用 margin
        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake

        # 计算损失
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        if self.addNegEntropy:
            scores = self.softmax(scores[labels == 0])
            p = scores.sum(0).view(-1)
            p /= p.sum()
            dist_loss = np.log(w.shape[0]) + (p * p.log()).sum()
            loss = dist_loss * 1e5 + emb_loss
        else:
            loss = emb_loss

        return loss, final_scores.squeeze(1)
