"""
测试结果验证与诊断工具
======================
用于验证 SAMO 模型在 VCTK 数据集上的测试结果文件，
自动检测文件格式、显示分数分布、计算混淆矩阵。

功能:
  1. 检查结果文件是否存在
  2. 自动检测分隔符并读取文件
  3. 显示真实/伪造语音的分数分布
  4. 计算混淆矩阵（基于指定阈值）
  5. 计算安全间隔（bona.min - spoof.max），评估分类效果

安全间隔越大，说明真实和伪造两类样本在分数上分离得越开，
模型的检测能力越强。如果间隔为负数，说明存在重叠区域，
某些伪造语音的分数比某些真实语音还高，这会导致分类错误。

使用方法:
  修改 result_file 变量为你的实际结果文件路径，然后运行:
  python 验证.py
"""

import pandas as pd
import numpy as np
import os

# ---- 请将此处修改为你的实际结果文件路径 ----
result_file = r'D:\PyCharm_project\samo-main\vctk_AdaIN-VC.txt'

print(f"检查文件: {os.path.abspath(result_file)}")
print(f"文件存在: {os.path.exists(result_file)}")

if not os.path.exists(result_file):
    print("文件不存在！请确认 -o 参数指定的输出文件名")
    exit()

# ---- 步骤1: 读取并显示原始内容的前5行 ----
print("\n原始文件前5行:")
with open(result_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  行{i}: {repr(line.strip())}")

# ---- 步骤2: 自动检测分隔符并读取文件 ----
print("\n尝试读取...")
try:
    # 使用空白字符（空格/制表符）作为分隔符
    df = pd.read_csv(result_file, sep=r'\s+', header=None, engine='python')
    print(f"读取成功，列数: {len(df.columns)}")
    print(f"前3行:\n{df.head(3)}")
except Exception as e:
    print(f"读取失败: {e}")
    exit()

# ---- 步骤3: 根据列数自动分配列名 ----
# 标准格式: filename, speaker, true_label, pred_label, score, file_path (6列)
# 带 mode 后缀: 额外一列 mode=with_enroll (7列)
if len(df.columns) == 6:
    df.columns = ['filename', 'speaker', 'true', 'pred', 'score', 'path']
elif len(df.columns) == 7:
    df.columns = ['filename', 'speaker', 'true', 'pred', 'score', 'path', 'extra']
else:
    print(f"未知列数: {len(df.columns)}，请检查文件格式")
    exit()

# ---- 步骤4: 数据类型转换 ----
df['true'] = pd.to_numeric(df['true'], errors='coerce')
df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# 检查是否有无效分数
nan_count = df['score'].isna().sum()
print(f"\nscore 列 NaN 数量: {nan_count}/{len(df)}")
if nan_count > 0:
    print("部分 score 为 NaN，显示前5个 NaN 行:")
    print(df[df['score'].isna()].head())

# ---- 步骤5: 按真实/伪造分离分数 ----
bona = df[df['true'] == 0]['score'].dropna()   # 真实语音的分数
spoof = df[df['true'] == 1]['score'].dropna()  # 伪造语音的分数

print(f"\n真实语音数量: {len(bona)}")
print(f"伪造语音数量: {len(spoof)}")

# ---- 步骤6: 计算分数分布和混淆矩阵 ----
if len(bona) > 0 and len(spoof) > 0:
    threshold = -13.4118  # 历史经验阈值（可根据实际数据调整）

    print(f"\n真实语音最小分数: {bona.min():.4f}")
    print(f"真实语音最大分数: {bona.max():.4f}")
    print(f"伪造语音最小分数: {spoof.min():.4f}")
    print(f"伪造语音最大分数: {spoof.max():.4f}")

    # 安全间隔: 真实最低分 - 伪造最高分
    # >0 说明两类完全分离，<0 说明存在重叠
    print(f"安全间隔 (bona.min - spoof.max): {bona.min() - spoof.max():.4f}")

    # 混淆矩阵（基于指定阈值）
    # 分数 > 阈值 → 判为真实(0)，分数 < 阈值 → 判为伪造(1)
    TP = ((df['true'] == 1) & (df['score'] < threshold)).sum()   # 伪造被正确检出
    TN = ((df['true'] == 0) & (df['score'] > threshold)).sum()   # 真实被正确通过
    FP = ((df['true'] == 0) & (df['score'] < threshold)).sum()   # 真实被误判为伪造
    FN = ((df['true'] == 1) & (df['score'] > threshold)).sum()   # 伪造被漏过

    print(f"\n混淆矩阵 (阈值 {threshold}):")
    print(f"  TP(伪造检出): {TP}")
    print(f"  TN(真实通过): {TN}")
    print(f"  FP(真实误杀): {FP}")
    print(f"  FN(伪造漏网): {FN}")
else:
    print("某类样本数量为0，无法计算间隔")
