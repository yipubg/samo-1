"""
工具函数模块
============
本模块包含训练和评估过程中需要的各种辅助函数。

主要功能:
  1. 随机种子设置（保证实验可复现）
  2. 学习率调度（Cosine Annealing / Exponential Decay）
  3. 评价指标计算（EER + t-DCF + 各攻击类型分解分析）
  4. 超球面均匀采样（用于 SAMO 中心的 "evenly" 初始化）
  5. 实验结果可视化（对比不同实验的训练曲线）
"""

import os
import random
from itertools import count
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List

import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_metrics as em


# ============================================================================
# 随机种子设置
# ============================================================================

def seed_worker(worker_id):
    """
    为 DataLoader 的每个 worker 进程设置独立的随机种子。
    配合 setup_seed() 使用，确保数据加载的可复现性。

    参数:
        worker_id: DataLoader 分配的 worker 编号
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_seed(random_seed, cudnn_deterministic=True):
    """
    设置全局随机种子，确保实验完全可复现。

    固定 Python、NumPy、PyTorch 的随机数生成器，
    并设置 CUDA 确定性模式（禁用自动算法选择，确保卷积等操作的确定性）。

    参数:
        random_seed:          整数随机种子
        cudnn_deterministic:  是否启用 CUDA 确定性模式

    注意: 确定性模式可能导致性能下降，但保证每次运行结果一致。
    详见 https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False  # 禁用自动算法选择


# ============================================================================
# 学习率调度策略
# ============================================================================

def cosine_annealing(step, total_steps, lr_max, lr_min):
    """
    余弦退火 (Cosine Annealing) 学习率衰减函数。

    学习率从 lr_max 平滑衰减到 lr_min，按照余弦曲线的形状:
      lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * step / total_steps))

    参数:
        step:        当前步数
        total_steps: 总步数
        lr_max:      最大学习率（起始值）
        lr_min:      最小学习率（结束值）

    返回:
        当前步数的学习率

    使用方式:
        scheduler = LambdaLR(optimizer,
            lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, lr_min/lr_max))
    """
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    """
    指数学习率衰减（备用方案）。

    每 interval 个 epoch 将学习率乘以 lr_decay:
      new_lr = lr * (lr_decay ^ (epoch_num // interval))

    参数:
        args:       命令行参数（包含 lr_decay 和 interval）
        lr:         当前基础学习率
        optimizer:  PyTorch 优化器
        epoch_num:  当前 epoch 编号
    """
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ============================================================================
# 评价指标计算
# ============================================================================

def compute_eer_tdcf(args, cm_score_file):
    """
    计算 EER 和最小 t-DCF（串联检测代价函数）。

    这是 ASVspoof 2019 官方评测的完整指标计算流程:
      1. 加载 ASV（声纹识别）评分文件
      2. 加载 CM（防伪检测）评分文件
      3. 计算 ASV 和 CM 的 EER
      4. 以 ASV 的 EER 阈值为基准，固定 ASV 的工作点
      5. 计算 t-DCF（串联系统的综合代价）
      6. 按攻击类型分解分析（Breakdown Analysis），计算每种攻击的单独 EER

    参数:
        args:          命令行参数
        cm_score_file: CM 评分文件路径

    返回:
        eer_cm:    CM 系统的等错误率
        min_tDCF:  最小串联检测代价
    """
    # ASV 评分文件（声纹识别系统的输出）
    asv_score_file = os.path.join('scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # ---- t-DCF 参数（来自 ASVspoof 2019 官方评测方案）----
    Pspoof = 0.05  # 欺骗攻击的先验概率
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,   # 目标说话人先验概率
        'Pnon': (1 - Pspoof) * 0.01,   # 非目标说话人先验概率
        'Cmiss_asv': 1,                # ASV 误拒目标说话人的代价
        'Cfa_asv': 10,                 # ASV 误接受非目标说话人的代价（惩罚更重）
        'Cmiss_cm': 1,                 # CM 误拒目标说话人的代价
        'Cfa_cm': 10,                  # CM 误接受伪造的代价（惩罚更重）
    }

    # ---- 加载 ASV 评分 ----
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]          # target / nontarget / spoof
    asv_scores = asv_data[:, 2].astype(float)

    # ---- 加载 CM 评分 ----
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 1]         # 攻击类型标识（如 A07, A08, ...）
    cm_keys = cm_data[:, 2].astype(int)  # 标签: 0=bonafide, 1=spoof
    cm_scores = cm_data[:, 3].astype(float)

    # ---- 按类别分离 ASV 评分 ----
    tar_asv = asv_scores[asv_keys == 'target']     # 目标说话人评分
    non_asv = asv_scores[asv_keys == 'nontarget']   # 非目标说话人评分
    spoof_asv = asv_scores[asv_keys == 'spoof']     # 伪造语音评分

    # ---- 按类别分离 CM 评分 ----
    bona_cm = cm_scores[cm_keys == 0]   # 真实语音评分
    spoof_cm = cm_scores[cm_keys == 1]  # 伪造语音评分

    # ---- 计算独立系统的 EER ----
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    # ---- 以 ASV EER 阈值固定 ASV 工作点 ----
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)

    # ---- 计算 t-DCF ----
    tDCF_curve, CM_thresholds = em.compute_tDCF(
        bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model)
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    # ---- 按攻击类型分解分析 (Breakdown Analysis) ----
    # 评估模型对每种攻击方法（A07-A19）的检测能力
    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    eer_cm_lst = {}
    for attack in attack_types:
        if attack == "-":
            continue
        bona_cm = cm_scores[cm_keys == 0]
        spoof_cm = cm_scores[cm_sources == attack]

        eer_att = em.compute_eer(bona_cm, spoof_cm)[0]
        if not np.isnan(eer_att):
            eer_cm_lst[attack] = eer_att

    # ---- 输出结果到文件 ----
    output_file = "./breakdown/{}.txt".format(args.save_score)
    with open(output_file, "w") as f_res:
        f_res.write('\nCM SYSTEM\n')
        f_res.write('\tEER\t\t= {:8.9f} % '
                    '(Equal error rate for countermeasure)\n'.format(eer_cm * 100))

        f_res.write('\nTANDEM\n')
        f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))

        f_res.write('\nBREAKDOWN CM SYSTEM\n')
        for attack_type in attack_types:
            _eer = eer_cm_lst[attack_type] * 100
            f_res.write(f'\tEER {attack_type}\t\t= {_eer:8.9f} % \n')

    os.system(f"cat {output_file}")

    return eer_cm, min_tDCF


# ============================================================================
# 高维超球面均匀采样
# ============================================================================
# 参考: https://stackoverflow.com/questions/57123194
# 用于 SAMO 中心的 "evenly" 初始化方式

def int_sin_m(x: float, m: int) -> float:
    """
    计算 ∫₀ˣ sinᵐ(t) dt 的递归积分。
    用于超球面参数化中的坐标转换。
    """
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


def primes() -> Iterator[int]:
    """
    生成无限质数序列的生成器。
    质数用于构造超球面上的均匀分布点（利用质数的无理数性质避免周期性）。
    """
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:
            step = composites.pop(i)
        elif i < psq:
            yield i
            continue
        else:
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def inverse_increasing(func: Callable[[float], float], target: float,
                       lower: float, upper: float, atol: float = 1e-10) -> float:
    """
    使用二分法求单调递增函数 func 在 [lower, upper] 内的反函数值。

    即求解 func(x) = target 中的 x。
    用于超球面坐标的逆变换。
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def uniform_hypersphere(d: int, n: int) -> List[List[float]]:
    """
    在 d 维超球面上均匀生成 n 个点。

    使用质数螺旋法 (Prime Spiral Method)，通过对球面坐标的均匀采样实现。
    相比随机采样，这种方法能保证点之间的分布更加均匀，
    避免随机初始化可能导致某些中心过于靠近的问题。

    参数:
        d: 超球面维度（等于 SAMO 的 enc_dim, 默认 160）
        n: 采样点数量（等于 SAMO 的 num_centers, 默认 20）

    返回:
        points: [(d个坐标), ...] 共 n 个点的列表
    """
    assert d > 1
    assert n > 0

    # 初始化：所有点从 (1, 1, ..., 1) 开始
    points = [[1 for _ in range(d)] for _ in range(n)]

    # 二维圆上均匀分布
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)

    # 逐维扩展到高维球面
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)  # 质数偏移避免周期性
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)

    return points


# ============================================================================
# 实验结果可视化
# ============================================================================

def compare_exps(exp_dirs, root_dir, max_train_loss=None, max_dev_loss=None, eval_available=False):
    """
    对比多个实验的训练曲线。

    绘制 4 个子图:
      1. 训练损失曲线
      2. 验证损失曲线
      3. 验证 EER 曲线
      4. 测试 EER 曲线（可选）

    参数:
        exp_dirs:        实验目录名称列表
        root_dir:        所有实验目录的根路径
        max_train_loss:  y 轴上限（训练损失）
        max_dev_loss:    y 轴上限（验证损失）
        eval_available:  是否有测试 EER 数据
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    for folder in exp_dirs:
        out_fold = os.path.join(root_dir, folder)
        train_log_file = os.path.join(out_fold, "train_loss.log")
        dev_log_file = os.path.join(out_fold, "dev_loss.log")

        # 绘制训练损失
        with open(train_log_file, "r") as train_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')]
                          for line in train_log.readlines()[1:]])
            it_per_batch = int(x[:, 1].max()) + 1
            x = x[it_per_batch - 1::it_per_batch]  # 每个 epoch 取一个点
            ax1.plot(range(1, len(x) + 1), x[:, 2])
            if max_train_loss is not None:
                ax1.set_ylim([0, max_train_loss])
            ax1.set_title("Training Loss")
            ax1.grid()
            ax1.legend(exp_dirs)

        # 绘制验证损失和 EER
        with open(dev_log_file, "r") as dev_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')]
                          for line in dev_log.readlines()[1:]])
            ax2.plot(range(1, len(x) + 1), x[:, 1])
            ax2.set_title("Validation Loss")
            if max_dev_loss is not None:
                ax2.set_ylim([0, max_dev_loss])
            ax2.legend(exp_dirs)

            ax3.plot(range(1, len(x) + 1), x[:, 2])
            ax3.set_title("Validation EER")
            ax3.minorticks_on()
            ax3.grid(b=True, which='major', linestyle='-')
            ax3.grid(b=True, which='minor', linestyle=':')
            ax3.set_ylim([0, 0.01])
            ax3.legend(exp_dirs)

        # 绘制测试 EER（可选）
        if eval_available:
            with open(os.path.join(out_fold, "test_loss.log"), "r") as eval_eer:
                x = np.array([[float(i) for i in line[0:-1].split('\t')]
                              for line in eval_eer.readlines()[1:]])
                ax4.plot(range(1, len(x) + 1), x[:, 2])
                ax4.set_title("Test EER")
                ax4.minorticks_on()
                ax4.grid(b=True, which='major', linestyle='-')
                ax4.grid(b=True, which='minor', linestyle=':')
                ax4.set_ylim([0, 0.08])
                ax4.legend(exp_dirs)

    plt.show(block=True)
