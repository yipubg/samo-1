"""
评价指标模块
============
本模块实现了语音伪造检测领域的标准评价指标。

核心指标:
  1. EER (Equal Error Rate，等错误率):
     系统在某个阈值下，错误拒绝真实语音的比例 (FRR)
     恰好等于错误接受伪造语音的比例 (FAR)。
     EER 越低越好，0 表示完美分类。

  2. t-DCF (Tandem Detection Cost Function，串联检测代价函数):
     评估"防伪检测(CM) + 声纹识别(ASV)"串联系统的综合性能。
     min t-DCF < 1 说明防伪系统对串联系统有正向收益。

计算流程:
  compute_det_curve → compute_eer → obtain_asv_error_rates → compute_tDCF

参考文献:
  [1] Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem Assessment
      of Spoofing Countermeasures and Automatic Speaker Verification", Odyssey 2018
  [2] ASVspoof 2019 Evaluation Plan
"""

import sys
import numpy as np


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """
    计算 ASV（声纹识别）系统在给定阈值下的三类错误率。

    参数:
        tar_asv:       目标说话人的 ASV 评分数组
        non_asv:       非目标说话人的 ASV 评分数组
        spoof_asv:     伪造语音的 ASV 评分数组
        asv_threshold: ASV 系统的决策阈值

    返回:
        Pfa_asv:        假接率（错误接受非目标说话人）
        Pmiss_asv:      假拒率（错误拒绝目标说话人）
        Pmiss_spoof_asv: 伪造漏检率（伪造语音被 ASV 接受的比例）
    """
    # 高于阈值的非目标说话人被错误接受
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size

    # 低于阈值的目标说话人被错误拒绝
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # 低于阈值的伪造语音被正确拒绝（这里计算的是未被拒绝的比例）
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    """
    计算 DET (Detection Error Tradeoff) 曲线。

    DET 曲线展示了在不同决策阈值下，
    错误拒绝率 (FRR) 和错误接受率 (FAR) 之间的权衡关系。

    算法:
      1. 将正负样本的分数合并，按升序排序
      2. 遍历每个分数作为候选阈值
      3. 计算每个阈值下的 FRR 和 FAR

    参数:
        target_scores:    正样本（真实语音）的评分数组
        nontarget_scores: 负样本（伪造语音）的评分数组

    返回:
        frr:        各阈值下的假拒率（False Rejection Rate）
        far:        各阈值下的假接率（False Acceptance Rate）
        thresholds: 对应的决策阈值
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # 按分数升序排序
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # 累积计数计算 FRR 和 FAR
    # 分数越高 → 越可能是真实语音（本项目 label=0 为真实）
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    # FRR: 真实语音被拒绝的比例（分数低于阈值）
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # FAR: 伪造语音被接受的比例（分数高于阈值）
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    # 阈值: 排序后的分数（第一个阈值比最小分数略低）
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    计算等错误率 (Equal Error Rate, EER)。

    EER 是 FRR = FAR 时的错误率。
    实际计算中取 |FRR - FAR| 最小的点，取 FRR 和 FAR 的平均值。

    参数:
        target_scores:    正样本评分（真实语音）
        nontarget_scores: 负样本评分（伪造语音）

    返回:
        eer:       等错误率 (0-1 之间的小数)
        threshold: 对应的决策阈值
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost=False):
    """
    计算串联检测代价函数 (Tandem Detection Cost Function, t-DCF)。

    背景:
      在实际部署中，防伪系统 (CM) 和声纹识别系统 (ASV) 是串联工作的:
        语音输入 → [CM 防伪检测] → [ASV 声纹识别] → 决策
      CM 先判断语音是否为伪造，只有被判为"真实"的语音才进入 ASV。

    t-DCF 综合了 CM 和 ASV 两个系统的错误代价:
      tDCF = C1 × Pmiss_cm + C2 × Pfa_cm

    其中 C1 和 C2 由 cost_model 参数和 ASV 错误率共同决定。

    参数:
        bonafide_score_cm:  真实语音的 CM 评分
        spoof_score_cm:     伪造语音的 CM 评分
        Pfa_asv:            ASV 的假接率
        Pmiss_asv:          ASV 的假拒率
        Pmiss_spoof_asv:    伪造语音被 ASV 接受的比例
        cost_model:         代价参数字典，包含:
            - Ptar, Pnon, Pspoof: 各类用户的先验概率
            - Cmiss_asv, Cfa_asv:  ASV 的误拒和误接代价
            - Cmiss_cm, Cfa_cm:    CM 的误拒和误接代价
        print_cost:         是否打印代价参数详情

    返回:
        tDCF_norm:      归一化 t-DCF 曲线值（>1 表示 CM 无正面作用）
        CM_thresholds:  对应的 CM 决策阈值
    """
    # ---- 参数合法性检查 ----
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # 检查分数中是否有 NaN 或 Inf
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # 确保是软分数（连续值），而非硬决策（二值）
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # ---- 计算 CM 的 DET 曲线 ----
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # ---- 计算 t-DCF 的权重系数 ----
    # C1: CM 误拒的代价权重（真实语音被 CM 拦截）
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    # C2: CM 误接的代价权重（伪造语音被 CM 放行）
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evaluate tDCF with negative weights '
                 '- please check whether your ASV error rates are correctly computed?')

    # ---- 计算各阈值下的 t-DCF ----
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # 归一化: tDCF_norm > 1 表示 CM 对系统没有正面作用
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # ---- 打印代价模型详情 ----
    if print_cost:
        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')
        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds
