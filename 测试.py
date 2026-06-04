"""
测试结果快速查看工具
====================
用于快速查看 VCTK 测试结果的混淆矩阵。

功能:
  读取 SAMO 测试生成的逐条结果文件 (.txt)，
  计算并显示假阳性 (FP) 和假阴性 (FN) 的数量。

这有助于快速了解模型在 VCTK 数据集上的表现:
  - FP (假阳性): 真实语音被误判为伪造的数量（误杀）
  - FN (假阴性): 伪造语音被漏过的数量（漏网）

使用方法:
  修改代码中的文件路径为你的实际结果文件路径，然后运行:
  python 测试.py
"""

import pandas as pd

# 读取逐条检测结果文件（制表符分隔）
# 文件格式: filename  speaker  true_label  pred_label  score  file_path
df = pd.read_csv('vctk_AdaIN-VC.txt', sep='\t',
                 names=['filename', 'speaker', 'true', 'pred', 'score', 'path'])

# 计算混淆矩阵
# FP: 真实语音 (true=0) 被模型判断为伪造 (pred=1)
FP = sum((df['true'] == 0) & (df['pred'] == 1))
# FN: 伪造语音 (true=1) 被模型判断为真实 (pred=0)
FN = sum((df['true'] == 1) & (df['pred'] == 0))

print(f"FP={FP}, FN={FN}")
print(f"误杀 {FP} 条真实语音，漏过 {FN} 条伪造语音")
