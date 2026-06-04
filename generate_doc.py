"""
生成 SAMO 项目架构与代码解释文档 (Word .docx)
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import os

doc = Document()

# ===== 页面设置 =====
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# ===== 样式设置 =====
style = doc.styles['Normal']
font = style.font
font.name = '微软雅黑'
font.size = Pt(11)
style.paragraph_format.line_spacing = 1.5

for level in range(1, 4):
    heading_style = doc.styles[f'Heading {level}']
    heading_style.font.name = '微软雅黑'

# ===== 封面标题 =====
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('SAMO 语音伪造检测项目\n架构与代码详解')
run.font.size = Pt(26)
run.font.bold = True
run.font.color.rgb = RGBColor(0, 51, 102)

doc.add_paragraph()
subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('——面向初学者的完整解读')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run(
    '论文: SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofing\n'
    '会议: ICASSP 2023    |    作者: Siwen Ding, You Zhang, Zhiyao Duan'
)
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(120, 120, 120)

doc.add_page_break()

# ===== 目录 =====
doc.add_heading('目录', level=1)
toc_items = [
    '1. 项目简介 —— 这个项目是做什么的？',
    '2. 论文核心思想 —— SAMO 方法通俗解释',
    '3. 项目文件结构 —— 每个文件的作用',
    '4. 核心模块详解',
    '   4.1 AASIST 骨干网络',
    '   4.2 SAMO 损失函数',
    '   4.3 OC-Softmax 损失函数（对比方法）',
    '   4.4 训练主流程 (main.py)',
    '   4.5 测试与评估 (test_vctk_samo.py)',
    '   4.6 评价指标 (eval_metrics.py)',
    '   4.7 工具函数 (utils.py)',
    '5. 数据流与处理流程',
    '6. 关键参数速查表',
    '7. 如何运行项目',
    '8. 技术要点总结',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# ===== 1. 项目简介 =====
doc.add_heading('1. 项目简介 —— 这个项目是做什么的？', level=1)

doc.add_heading('1.1 一句话概括', level=2)
doc.add_paragraph(
    'SAMO 是一个基于深度学习的语音伪造检测系统。通俗地说，它的任务是：给你一段语音，判断这段语音是 '
    '真人的声音（bonafide），还是AI合成/转换的假声音（spoof）。'
)

doc.add_heading('1.2 背景：为什么需要语音伪造检测？', level=2)
doc.add_paragraph(
    '近年来，AI语音合成技术（如TTS、语音转换VC）越来越成熟，生成的假声音几乎可以以假乱真。'
    '这带来了严重的安全隐患：诈骗分子可以用AI模仿你的声音进行电话诈骗；'
    '坏人可以通过语音伪造绕过声纹识别系统。因此，检测语音是否被伪造变得非常重要。'
)
doc.add_paragraph(
    '这个项目就是该领域的学术研究课题，发表在语音信号处理领域顶级会议 ICASSP 2023上。'
    '作者提出了一种名为 SAMO（Speaker Attractor Multi-Center One-Class Learning）的新方法，'
    '在公开基准测试集 ASVspoof 2019 LA 上取得了优异的检测性能。'
)

doc.add_heading('1.3 ASVspoof 2019 数据集简介', level=2)
doc.add_paragraph(
    'ASVspoof 2019 LA（Logical Access，逻辑访问）是语音防伪领域最权威的基准测试数据集之一。'
    '它包含三个子集：'
)
doc.add_paragraph('训练集 (Train)：25,380 条语音，来自 20 个说话人（8男12女）', style='List Bullet')
doc.add_paragraph('开发集 (Dev)：24,844 条语音，来自 10 个说话人', style='List Bullet')
doc.add_paragraph('评估集 (Eval)：71,237 条语音，来自 29 个说话人', style='List Bullet')
doc.add_paragraph(
    '检测难度：评估集中的攻击类型（A07-A19）在训练集中从未见过，因此模型需要具备对新攻击的泛化能力。'
    '本项目论文中 SAMO 在评估集上取得的 EER（等错误率）仅为 0.88%，是非常优秀的成绩。'
)

doc.add_page_break()

# ===== 2. 论文核心思想 =====
doc.add_heading('2. 论文核心思想 —— SAMO 方法通俗解释', level=1)

doc.add_heading('2.1 问题是什么？', level=2)
doc.add_paragraph(
    '语音伪造检测本质上是一个"二分类"问题：判断一段语音是真实的还是伪造的。'
    '但这里有一个关键难点：训练时你只能收集到一部分攻击类型（例如某些TTS方法），'
    '但在实际部署中，攻击者会用你从没见过的攻击方法来欺骗系统。所以，这个任务本质上要求模型 '
    '学会判断"什么是真实的"，而不是"什么是伪造的"——这就是"单类学习（One-Class Learning）"的思想。'
)

doc.add_heading('2.2 SAMO 的三个核心设计', level=2)

doc.add_heading('2.2.1 说话人吸引力子 (Speaker Attractor)', level=3)
doc.add_paragraph(
    '每个说话人的声音都有自己独特的"声纹特征"（可以参考指纹的概念）。'
    'SAMO 的核心创新之一是"说话人吸引力子（Speaker Attractor）"：'
    '为每个说话人在高维空间中学习一个向量表示（称为 attractiveness 或 中心向量），'
    '这个向量就像是该说话人声音特征的"锚点"。'
    '当一段语音被判断时，系统会计算这段语音的特征向量与对应说话人吸引力子的相似度。'
    '如果相似度很高，说明这段语音很可能来自真实的那个说话人；如果相似度很低，则可能是伪造的。'
)
doc.add_paragraph(
    '关于词汇：原文使用 "Attractor"（吸引力子）来强调其功能——在特征空间中像一个"吸引中心"，'
    '真实语音的嵌入向量会被它吸引、围绕在它周围；而伪造语音的向量则远离这些中心。'
    '虽然代码中常使用更为通用的 "center"（中心）一词，但在论文语境下 "attractor" '
    '更准确地体现了设计动机。'
)

doc.add_heading('2.2.2 多中心设计 (Multi-Center)', level=3)
doc.add_paragraph(
    '传统的单类学习方法（如 OC-Softmax）只使用一个中心向量来代表所有真实语音。'
    '但这有个问题：不同说话人的声音差异很大，用一个中心很难精确描述所有真实语音的分布。'
    'SAMO 的第二个创新是使用多个中心（Multi-Center），为每个说话人学习独立的中心向量。'
    '这样，模型可以更精细地捕捉每个说话人的声音特征，从而更好地区分真实和伪造。'
)

doc.add_heading('2.2.3 单类学习损失函数', level=3)
doc.add_paragraph(
    'SAMO 设计了一个专门的损失函数来驱动训练。其核心思想是：'
)
doc.add_paragraph(
    '真实语音（bonafide）：其特征向量应该与对应说话人的中心向量高度相似（余弦相似度 > m_real，m_real 是设定的阈值）',
    style='List Bullet'
)
doc.add_paragraph(
    '伪造语音（spoof）：其特征向量应该远离所有中心向量（与最近中心的相似度 < m_fake）',
    style='List Bullet'
)
doc.add_paragraph(
    '数学上，损失函数使用 Softplus 函数来惩罚不符合上述条件的样本。Softplus 是一个平滑版的 ReLU，'
    '当输入小于零时输出接近零（不惩罚），当输入大于零时输出近似等于输入（惩罚力度随偏差增大而增大）。'
    'scale 参数 α（默认 20）控制惩罚的"陡峭程度"，α 越大，对偏差的惩罚越严厉。'
    '训练过程就是通过反向传播不断缩小这个损失，最终让真实语音都"聚集"在各自的中心附近，'
    '而伪造语音则被"推远"。'
)

doc.add_heading('2.3 三种评分模式', level=2)
doc.add_paragraph('SAMO 支持三种不同的评分/训练策略（通过 --train_sp 或 --val_sp 参数控制）：')

doc.add_paragraph(
    '模式1 — 说话人感知 (Speaker-Aware/SIM)：对于查询语音，找到它对应说话人的中心向量，'
    '计算一对一（1-on-1）的余弦相似度作为分数。这种模式需要知道每条语音属于哪个说话人，'
    '在实际应用中通过注册（enrollment）阶段获取说话人信息。',
    style='List Bullet'
)
doc.add_paragraph(
    '模式2 — 说话人无关 (Speaker-Agnostic/MAXSCORE)：对于查询语音，计算它与所有说话人中心的相似度，'
    '取最大值作为分数。不需要事先知道语音来自哪个说话人。',
    style='List Bullet'
)
doc.add_paragraph(
    '模式0 — 说话人独立 (Speaker-Independent)：使用训练时的中心向量，不进行注册更新。',
    style='List Bullet'
)

doc.add_page_break()

# ===== 3. 项目文件结构 =====
doc.add_heading('3. 项目文件结构 —— 每个文件的作用', level=1)

file_structure = [
    ('samo-main/', '项目根目录'),
    ('├── models/samo.pt', '预训练好的 SAMO 模型权重文件（约 35MB，论文中的最优模型）'),
    ('├── requirements.txt', 'Python 依赖包列表（PyTorch、NumPy、librosa 等）'),
    ('├── README.md', '项目说明（英文，介绍如何训练和测试）'),
    ('├── LICENSE', '开源许可证'),
    ('├── build_test_sets.py', '批量构建 VCTK 测试集的工具脚本'),
    ('├── 测试.py / 验证.py', '辅助调试脚本，用于查看测试结果和验证指标'),
    ('├── samo/', '核心代码包（Python 包）'),
    ('│   ├── __init__.py', '包初始化文件'),
    ('│   ├── main.py', '★ 主入口：训练和测试的完整流程'),
    ('│   ├── loss.py', '★ SAMO 和 OC-Softmax 损失函数的实现'),
    ('│   ├── utils.py', '★ 工具函数：随机种子、学习率调度、评价指标计算'),
    ('│   ├── eval_metrics.py', '评价指标：EER、t-DCF 计算'),
    ('│   ├── search_lr.py', '学习率搜索工具（使用 LR Finder 找到最佳学习率）'),
    ('│   ├── test_vctk.py', '在 VCTK 数据集上测试模型的脚本（早期版本）'),
    ('│   ├── test_vctk_samo.py', '★ 在 VCTK 数据集上测试 SAMO 模型（增强版，含完整指标）'),
    ('│   └── aasist/', 'AASIST 骨干网络子包'),
    ('│       ├── __init__.py', '包初始化文件'),
    ('│       ├── AASIST.py', '★ AASIST 模型定义（图注意力网络）'),
    ('│       ├── AASIST.conf', 'AASIST 模型和训练的配置文件（JSON）'),
    ('│       ├── AASIST.pth', 'AASIST 官方预训练权重（用于对比）'),
    ('│       └── data_utils.py', '★ 数据加载：读取协议文件、音频预处理、数据集类'),
    ('├── protocols/', 'ASVspoof 2019 LA 实验协议文件（指定训练/验证/测试划分）'),
    ('├── scores/', 'ASV 说话人验证评分（用于计算 t-DCF 指标）'),
    ('└── vctk_lfcc_cache_clean/', 'VCTK 音频预处理缓存（LFCC 特征）'),
]

for path, desc in file_structure:
    p = doc.add_paragraph()
    run_path = p.add_run(path)
    run_path.font.name = 'Consolas'
    run_path.font.size = Pt(10)
    run_path.font.bold = path.startswith('★')
    run_desc = p.add_run(f'  —  {desc}')
    run_desc.font.size = Pt(10)

doc.add_page_break()

# ===== 4. 核心模块详解 =====
doc.add_heading('4. 核心模块详解', level=1)

# 4.1 AASIST
doc.add_heading('4.1 AASIST 骨干网络 (samo/aasist/AASIST.py)', level=2)

doc.add_paragraph(
    'AASIST（Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks）'
    '是韩国 NAVER 公司研究团队提出的语音防伪骨干网络，被 SAMO 用作特征提取器（即 backbone）。'
    '你可以把它理解为——一个专门设计用来从原始语音中提取有效特征的深度神经网络。'
)

doc.add_heading('网络结构总览', level=3)
doc.add_paragraph(
    'AASIST 是一个端到端（end-to-end）模型，直接从原始波形（raw waveform）输入，不需要手工设计声学特征。'
    '它主要包含以下几层处理：'
)

doc.add_paragraph(
    'SincConv 前端层：使用基于 sinc 函数的卷积核替代传统卷积，直接对原始波形进行频谱滤波。'
    '它的滤波频率按照梅尔（Mel）刻度分布，更贴近人耳听觉特性。输出 70 个频带的特征。',
    style='List Number'
)
doc.add_paragraph(
    '残差编码器 (Residual Encoder)：6 层残差卷积块（Residual_block），逐步提取更高层次的特征。'
    '每一层包含两个卷积层 + 批归一化 + 最大池化。通道数从 32 逐步增加到 64。',
    style='List Number'
)
doc.add_paragraph(
    '图注意力层 (Graph Attention Layer, GAT)：将频谱维度（Spectral）和时间维度（Temporal）'
    '分别建模为图结构中的"节点"，使用注意力机制学习不同频带和不同时间段之间的相互关系。',
    style='List Number'
)
doc.add_paragraph(
    '异构图注意力层 (HtrgGAT)：融合频谱和时间两个维度的信息进行交互。'
    '通过可学习的"主节点 (master node)"在频谱图和时间图之间传递信息，实现跨维度特征融合。',
    style='List Number'
)
doc.add_paragraph(
    '读出层 (Readout)：对融合后的特征进行全局最大池化和平均池化，拼接后通过全连接层输出'
    '两个值：一个 160 维的特征向量（embedding，用于 SAMO 等损失函数）和一个 2 分类的输出（用于 Softmax 损失）。',
    style='List Number'
)

doc.add_heading('关键源码解析', level=3)

doc.add_paragraph(
    'Model.forward() 方法（AASIST.py 第 531 行）是理解整个网络的核心。数据流如下：'
)
code_text = (
    '1. x.unsqueeze(1)              # (batch, 64600) → (batch, 1, 64600)\n'
    '2. self.conv_time(x)           # SincConv: (batch, 1, 64600) → (batch, 70, T)\n'
    '3. x.unsqueeze(1)              # → (batch, 1, 70, T)\n'
    '4. F.max_pool2d + BN + SELU   # 初步特征处理\n'
    '5. self.encoder(x)             # 6层残差卷积: → (batch, 64, F, T)\n'
    '6. GAT_S / GAT_T               # 频域/时域图注意力\n'
    '7. HtrgGAT × 2                 # 双分支异构融合（含跳跃连接）\n'
    '8. last_hidden = concat(T_max, T_avg, S_max, S_avg, master)\n'
    '9. output = out_layer(last_hidden)  # → (batch, 160) 和 (batch, 2)\n'
    '10. return last_hidden, output       # embedding 特征 + 分类输出'
)
p = doc.add_paragraph()
run = p.add_run(code_text)
run.font.name = 'Consolas'
run.font.size = Pt(9)

doc.add_paragraph()
doc.add_paragraph(
    '模型总参数量约 280 万。对于语音处理任务来说，这是一个中等规模的模型，'
    '可以在单个 GTX 1080 Ti（11GB 显存）上以 batch_size=23 进行训练。'
)

doc.add_heading('配置文件 (AASIST.conf) 关键参数', level=3)
params_table = doc.add_table(rows=8, cols=3, style='Light List Accent 1')
headers = ['参数', '默认值', '含义']
for i, h in enumerate(headers):
    params_table.rows[0].cells[i].text = h
    for p in params_table.rows[0].cells[i].paragraphs:
        p.runs[0].font.bold = True

config_params = [
    ('nb_samp', '64600', '输入音频样本数（约 4 秒 @16kHz）'),
    ('first_conv', '128', 'SincConv 卷积核大小'),
    ('filts', '[70, [1,32], [32,32], [32,64], [64,64]]', '各层通道数配置'),
    ('gat_dims', '[64, 32]', '图注意力层维度'),
    ('pool_ratios', '[0.5, 0.7, 0.5, 0.5]', '各池化层的保留比例'),
    ('temperatures', '[2.0, 2.0, 100.0, 100.0]', '注意力温度参数'),
    ('base_lr', '0.0001', '初始学习率'),
]
for i, (param, val, meaning) in enumerate(config_params):
    params_table.rows[i + 1].cells[0].text = param
    params_table.rows[i + 1].cells[1].text = val
    params_table.rows[i + 1].cells[2].text = meaning

doc.add_page_break()

# 4.2 SAMO 损失函数
doc.add_heading('4.2 SAMO 损失函数 (samo/loss.py)', level=2)

doc.add_paragraph(
    'loss.py 是整个项目的核心创新所在。它定义了两个损失函数类：OCSoftmax（对比基线方法）和 '
    'SAMO（论文提出的新方法），都继承自 PyTorch 的 nn.Module。'
)

doc.add_heading('SAMO 类的结构', level=3)
doc.add_paragraph(
    'SAMO 类（loss.py 第 42 行）封装了完整的 SAMO 损失函数。它的设计围绕以下几个关键组件：'
)

doc.add_paragraph(
    '中心向量矩阵 (self.center)：形状为 (num_centers × enc_dim)，即 (20 × 160) 的矩阵。'
    '每一行是一个说话人的中心向量。初始化时使用 one-hot 方式——即取单位矩阵的前 20 行，'
    '确保初始中心向量两两正交（互不干扰），为每个说话人提供独立的"锚点空间"。',
    style='List Bullet'
)
doc.add_paragraph(
    '前向传播 (forward)：输入包括特征 x、标签 labels、说话人信息 spk 和注册中心 enroll。'
    '核心逻辑是：归一化特征和中心 → 计算余弦相似度矩阵 → 根据 attractor 模式计算最终分数 → '
    '应用 margin 约束 → 通过 Softplus 计算损失。',
    style='List Bullet'
)
doc.add_paragraph(
    '推理模式 (inference)：与 forward 类似，但能处理没有注册中心的说话人（对未知说话人使用 maxscore 策略）。',
    style='List Bullet'
)

doc.add_heading('SAMO 前向传播的数学过程', level=3)
doc.add_paragraph('以 attractor=1（说话人感知模式）为例，forward 方法的计算步骤如下：')

steps = [
    '1. 归一化：x = F.normalize(x, p=2, dim=1) — 将每个特征向量归一化到单位超球面',
    '2. 归一化：w = F.normalize(self.center, p=2, dim=1) — 将中心向量也归一化',
    '3. 计算相似度：scores = x @ w^T — 得到 (batch_size × num_centers) 的余弦相似度矩阵',
    '4. 获取 maxscore：maxscores, _ = torch.max(scores, dim=1) — 每个样本与所有中心的最大相似度',
    '5. 说话人一对一分数：final_scores = sum(x * speaker_center, dim=1) — 用对应说话人中心替代 maxscore（仅对 bonafide）',
    '6. 应用 margin：',
    '   对 bonafide (label=0): adjusted = m_real - final_scores  （期望分数 > m_real，否则被惩罚）',
    '   对 spoof (label=1):    adjusted = maxscores - m_fake    （期望分数 < m_fake，否则被惩罚）',
    '7. 计算损失：loss = Softplus(α × adjusted).mean()  — α=20 控制惩罚斜率',
]
for step in steps:
    doc.add_paragraph(step, style='List Bullet')

doc.add_paragraph()
doc.add_paragraph(
    '直观理解：如果一段真实语音与其对应说话人中心的相似度为 0.9，而 m_real=0.7，'
    '那么 adjusted = 0.7 - 0.9 = -0.2，Softplus(-0.2 × 20) ≈ 0，几乎不产生损失（这是好的）。'
    '但如果相似度只有 0.3，则 adjusted = 0.7 - 0.3 = 0.4，Softplus(0.4 × 20) 会产生很大的损失，'
    '梯度会推动模型让该语音的特征更靠近其说话人中心。'
)

doc.add_heading('关键参数说明', level=3)
samo_params = doc.add_table(rows=7, cols=3, style='Light List Accent 1')
for i, h in enumerate(['参数', '默认值', '含义']):
    samo_params.rows[0].cells[i].text = h
    for p in samo_params.rows[0].cells[i].paragraphs:
        p.runs[0].font.bold = True
samo_data = [
    ('feat_dim / enc_dim', '160', '特征向量维度，即 AASIST 输出的 embedding 大小'),
    ('num_centers', '20', '中心向量数量（对应训练集中 20 个说话人）'),
    ('m_real', '0.7', '真实语音得分下界：bonafide 与中心的相似度需要高于此值'),
    ('m_fake', '0.0', '伪造语音得分上界：spoof 与中心的相似度需要低于此值'),
    ('alpha', '20.0', '损失函数的缩放因子，越大则惩罚越"硬"'),
    ('initialize_centers', 'one_hot', '中心初始化方式，one_hot 保证初始中心互为正交'),
]
for i, (param, val, meaning) in enumerate(samo_data):
    samo_params.rows[i + 1].cells[0].text = param
    samo_params.rows[i + 1].cells[1].text = val
    samo_params.rows[i + 1].cells[2].text = meaning

doc.add_page_break()

# 4.3 OC-Softmax
doc.add_heading('4.3 OC-Softmax 损失函数 (samo/loss.py)', level=2)

doc.add_paragraph(
    'OCSoftmax 类（loss.py 第 9 行）是 SAMO 的对比基线方法。它只使用一个全局中心向量 '
    '（self.center 形状为 1 × feat_dim）来代表所有真实说话人的特征分布。'
    '其计算过程与 SAMO 类似，但更简单：所有样本只与这唯一的中心向量比较。'
    '由于只有一个中心，它无法像 SAMO 那样区分不同说话人，因此被称为"说话人无关"的方法。'
    '与 SAMO 的差异：SAMO 使用 20 个中心的 one-hot 初始化，而 OC-Softmax 只用 1 个中心。'
    '在论文实验中，SAMO 在评估集上的 EER（0.88%）显著优于 OC-Softmax（约 1.5%），'
    '验证了多中心说话人感知设计的有效性。'
)

# 4.4 main.py
doc.add_heading('4.4 训练主流程 (samo/main.py)', level=2)

doc.add_paragraph(
    'main.py 是项目的顶层入口文件，包含训练 (train) 和测试 (test) 两个核心函数。'
    '文件共约 765 行，结构清晰，分为以下几个部分：'
)

doc.add_heading('init_params() — 参数解析 (第 21 行)', level=3)
doc.add_paragraph(
    '使用 argparse 定义了所有命令行参数。涵盖了数据路径、模型超参数、'
    '损失函数配置、训练策略、测试选项等约 30 个参数。'
    '同时还初始化了随机种子、创建输出目录结构。'
)

doc.add_heading('get_loader() — 数据加载 (第 154 行)', level=3)
doc.add_paragraph(
    '这是数据准备的核心函数，加载 6 个不同的 DataLoader：'
)
doc.add_paragraph(
    'trn_loader: 训练数据加载器（所有数据，含 bonafide 和 spoof）',
    style='List Bullet'
)
doc.add_paragraph(
    'trn_bona: 仅训练集中的真实语音（bonafide-only），用于计算/更新说话人中心',
    style='List Bullet'
)
doc.add_paragraph(
    'dev_loader: 验证数据加载器（目标说话人 only，由 --target 参数控制）',
    style='List Bullet'
)
doc.add_paragraph(
    'dev_enroll: 验证集的注册数据（用于提取验证集说话人的中心向量）',
    style='List Bullet'
)
doc.add_paragraph(
    'eval_loader: 评估集数据加载器',
    style='List Bullet'
)
doc.add_paragraph(
    'eval_enroll: 评估集的注册数据',
    style='List Bullet'
)

doc.add_heading('train() — 训练循环 (第 347 行)', level=3)
doc.add_paragraph('训练循环的完整流程如下：')

train_steps = [
    ('初始化模型和配置（第 352-358 行）', '加载 AASIST 配置文件，创建骨干网络 feat_model。'),
    ('加载数据（第 367-368 行）', '调用 get_loader() 获取所有 DataLoader。'),
    ('创建优化器和学习率调度器（第 380-381 行）', '使用 Adam 优化器 + Cosine Annealing 学习率衰减策略。'),
    ('初始化损失函数（第 385-396 行）', '根据选择创建 SAMO 或 OCSoftmax 损失函数实例。'),
    ('主循环开始 — 每个 epoch 的操作（第 407 行）', '以下步骤在每个 epoch 中执行：'),
    ('  a) 初始化/更新 SAMO 中心（第 421-431 行）', 'epoch=0 时使用 one-hot 初始化；之后每 3 个 epoch 用 bona 数据更新说话人中心向量。'),
    ('  b) 训练迭代（第 433-471 行）', '遍历训练数据：前向传播 → 计算损失 → 反向传播 → 更新参数。每个 batch 执行一次学习率衰减（Cosine annealing 在 step 级别）。'),
    ('  c) 验证评估（第 478-527 行）', '在验证集上计算损失和 EER。先更新验证中心（dev_enroll），然后计算每个样本的分数。'),
    ('  d) 测试评估（第 530-579 行）', '可选的评估集测试（由 --test_on_eval 控制间隔）。流程与验证相同。'),
    ('  e) 保存检查点（第 586-599 行）', '每隔 save_interval 个 epoch 保存模型权重。'),
    ('  f) 保存最佳模型（第 601-626 行）', '基于验证损失，保留性能最好的模型（early stop 机制，100 个 epoch 无改进则停止）。'),
]
for title, desc in train_steps:
    doc.add_paragraph(f'{title}: {desc}')

doc.add_heading('update_embeds() — 中心向量更新 (第 641 行)', level=3)
doc.add_paragraph(
    '这是一个关键函数。对于注册集中的每个说话人，将其所有 clean 语音通过 AASIST 提取特征 embedding，'
    '然后对这些 embedding 取平均，得到该说话人的中心向量（attractor）。这个平均向量代表了该说话人'
    '在嵌入空间中的"典型位置"。每次更新后，这些中心会被传递给 SAMO 损失函数，用于计算损失和评分。'
    '默认每 3 个 epoch 更新一次中心，以确保模型参数稳定后再更新。'
)

doc.add_heading('test() — 独立测试 (第 661 行)', level=3)
doc.add_paragraph(
    '这是一个独立的测试函数，用于加载已训练的模型并在评估集上运行。它支持三种评分方式：'
    'samo（使用说话人吸引子）、ocsoftmax（使用单中心）和 fc（使用全连接层输出）。'
    '如果指定了 --save_score 参数，会将每个样本的详细分数保存到文件，便于后续分析。'
    '测试完成后调用 compute_eer_tdcf() 计算 EER 和最小 t-DCF 指标。'
)

doc.add_page_break()

# 4.5 test_vctk_samo.py
doc.add_heading('4.5 VCTK 测试脚本 (samo/test_vctk_samo.py)', level=2)

doc.add_paragraph(
    '这是一个新增的测试脚本，用于在 VCTK（Voice Cloning Toolkit）数据集上评估预训练的 SAMO 模型。'
    'VCTK 是另一个常用的语音数据集，包含 109 个英语母语说话人的语音。这个脚本展示了如何将 SAMO '
    '迁移到新的数据集上使用。'
)

doc.add_heading('核心组件', level=3)

doc.add_paragraph(
    'VCTKRawDataset 类 (第 52 行)：自定义的 PyTorch 数据集类。扫描指定目录下的所有 .wav 文件，'
    '从目录名中提取说话人 ID（如 p225），将音频加载为单声道 16kHz 信号，并裁剪/填充到固定长度 '
    '(默认 64600 个采样点，约 4 秒)。真实语音标记为 label=0，伪造语音标记为 label=1。'
    '支持分别加载 clean 和 attack 两部分数据，可灵活组合。'
)
doc.add_paragraph(
    'VCTKTester 类 (第 125 行)：封装了模型加载和测试逻辑的核心类：'
)
doc.add_paragraph('load: 加载训练好的模型和损失模型（支持多种检查点格式）', style='List Bullet')
doc.add_paragraph(
    'compute_speaker_attractors: 使用 clean 音频为每个说话人预计算中心向量（attractor）。'
    '在 SAMO 的 with_enroll 模式下，这是测试前的必要准备步骤。处理过程为：对每个说话人的所有 '
    'clean 音频提取 embedding → 取均值 → L2 归一化，得到该说话人的最终中心向量。',
    style='List Bullet'
)
doc.add_paragraph(
    'evaluate: 在测试集上运行模型，支持两种 SAMO 评分模式：with_enroll（利用预计算的说话人中心，'
    '计算 1-on-1 相似度）和 no_enroll（使用模型自带的训练中心，计算 maxscore）。'
    '输出结果保存到文件，同时计算 EER、Accuracy、Precision、Recall、F1-Score 和混淆矩阵。',
    style='List Bullet'
)

doc.add_page_break()

# 4.6 eval_metrics.py
doc.add_heading('4.6 评价指标 (samo/eval_metrics.py)', level=2)

doc.add_paragraph(
    '这个文件实现了语音伪造检测领域的标准评价指标。'
)

doc.add_heading('EER (Equal Error Rate，等错误率)', level=3)
doc.add_paragraph(
    'EER 是衡量检测系统性能的核心指标。它的含义是：当系统调整到某个阈值时，错误拒绝真实语音的比例（FRR）'
    '恰好等于错误接受伪造语音的比例（FAR），这个相等时的错误率就是 EER。EER 越低越好，0 表示完美分类。'
    'SAMO 论文报告的 EER 为 0.88%，意味着每 1000 条语音中大约只有 9 条被错误判断。'
)
doc.add_paragraph('计算过程 (compute_eer, 第 41 行)：', style='List Bullet')
doc.add_paragraph(
    'a) compute_det_curve: 按分数排序所有样本，遍历每个分数作为候选阈值，计算对应的 FRR（假拒率）和 FAR（假接率）',
    style='List Bullet'
)
doc.add_paragraph('b) 找到 |FRR - FAR| 最小时的阈值和误差率', style='List Bullet')
doc.add_paragraph('c) EER = 该点 FRR 和 FAR 的平均值', style='List Bullet')

doc.add_heading('t-DCF (Tandem Detection Cost Function)', level=3)
doc.add_paragraph(
    't-DCF（compute_tDCF, 第 50 行）评估串联系统（CM → ASV）的性能：先经过防伪检测（CM），'
    '检测为真实的语音再进入声纹识别（ASV）系统。它将两种系统的错误代价合并为一个统一指标。'
    '参数设置来自 ASVspoof 2019 官方评测方案：Pspoof（攻击先验概率）=0.05，Cfa_cm 和 Cfa_asv '
    '（误接受代价）=10，Cmiss_cm 和 Cmiss_asv（误拒绝代价）=1，体现了对误接受更严厉的惩罚。'
    'min t-DCF 越低越好，小于 1 说明防伪系统有正向收益。'
)

doc.add_page_break()

# 4.7 utils.py
doc.add_heading('4.7 工具函数 (samo/utils.py)', level=2)

doc.add_paragraph('utils.py 包含了训练和评估过程中需要的各种辅助函数：')

utils_funcs = [
    ('setup_seed (第 24 行)', '固定随机种子，确保实验可复现。同时设置 CUDA 确定性模式，保证 GPU 运算结果一致。'),
    ('seed_worker (第 15 行)', '为 DataLoader 的每个 worker 设置独立种子，配合 setup_seed 使用。'),
    ('cosine_annealing (第 51 行)', '余弦退火学习率衰减函数。学习率从初始值平滑下降到最小值，公式为'
     ' lr = lr_min + (lr_max - lr_min) × 0.5 × (1 + cos(π × step / total_steps))。'
     '支持 "cosine2" 模式，将周期延长一倍，实现更平缓的衰减。'),
    ('adjust_learning_rate (第 57 行)', '指数学习率衰减（备用方案），每 interval 个 epoch 将学习率乘以 lr_decay。'),
    ('compute_eer_tdcf (第 67 行)', '加载 CM（防伪）和 ASV（声纹识别）两个系统的评分文件，计算 EER 和 '
     '最小 t-DCF。同时还会按攻击类型（A07-A19）分别计算每种攻击方法的 EER，输出"分解分析"（breakdown analysis），'
     '帮助了解模型对不同攻击方法的检测能力差异。'),
    ('uniform_hypersphere (第 221 行)', '在高维超球面上均匀采样 N 个点。用于 SAMO 中心的"均匀"初始化方式 '
     '(initialize_centers="evenly")。采用质数螺旋方法保证采样的均匀性。'),
    ('compare_exps (第 246 行)', '可视化函数：绘制多个实验的训练损失、验证损失、验证 EER 和测试 EER 的对比曲线。'),
]
for title, desc in utils_funcs:
    doc.add_paragraph(f'{title}: {desc}')

doc.add_page_break()

# 4.8 data_utils.py
doc.add_heading('4.8 数据处理 (samo/aasist/data_utils.py)', level=2)

doc.add_paragraph(
    '这个文件负责数据加载和预处理，由 AASIST 项目提供并在 SAMO 中进行了修改，'
    '添加了说话人信息的支持。'
)

doc.add_heading('genSpoof_list() — 协议文件解析 (第 12 行)', level=3)
doc.add_paragraph(
    '读取 ASVspoof 2019 协议文件（.trl.txt 或 .trn.txt），解析每行包含的说话人ID、音频文件名、'
    '标签（bonafide/spoof）等信息。返回四个数据结构：'
)
doc.add_paragraph('d_meta: {文件名: 标签} 字典（0=bonafide, 1=spoof）', style='List Bullet')
doc.add_paragraph('utt_list: 音频文件名列表', style='List Bullet')
doc.add_paragraph('utt2spk: {文件名: 说话人ID} 字典', style='List Bullet')
doc.add_paragraph('tag_list: 攻击类型标识列表（如 A07, A08 等）', style='List Bullet')
doc.add_paragraph(
    '当 enroll=True 时，函数读取注册协议文件（按说话人组织的文件列表），处理方式略有不同。'
)

doc.add_heading('ASVspoof2019_speaker_raw 类 — 数据集 (第 98 行)', level=3)
doc.add_paragraph(
    '继承自 PyTorch 的 Dataset 基类。__getitem__ 方法执行以下操作：'
)
doc.add_paragraph('使用 soundfile 读取 FLAC 格式的原始音频', style='List Bullet')
doc.add_paragraph(
    '训练模式：使用 pad_random() 随机裁剪（数据增强的一种方式）。如果音频超过 64600 个采样点，'
    '随机选择一个起始点截取；如果不足，重复拼接至目标长度',
    style='List Bullet'
)
doc.add_paragraph('测试模式：使用 pad() 固定起始点裁剪，保证每次测试结果一致', style='List Bullet')
doc.add_paragraph('返回: (音频tensor, 标签, 说话人ID, 文件名, 攻击类型)', style='List Bullet')

doc.add_page_break()

# ===== 5. 数据流与处理流程 =====
doc.add_heading('5. 数据流与处理流程', level=1)

doc.add_heading('5.1 训练阶段数据流', level=2)
doc.add_paragraph('下图展示了训练过程中数据在系统中的完整流动路径：')

flow_steps = [
    '1. 原始音频文件（.flac, 16kHz）→ soundfile 读取 → numpy 数组 (64600,)',
    '2. numpy 数组 → torch.Tensor → 输入 AASIST 网络',
    '3. AASIST 前向传播 → 输出两类结果：',
    '   - feat (embedding): (batch, 160) 的特征向量',
    '   - output (logits): (batch, 2) 的分类输出',
    '4. 特征向量 → SAMO 损失函数 → 计算损失值',
    '   内部步骤：特征归一化 → 与说话人中心计算余弦相似度 → 应用 margin → Softplus → 标量损失',
    '5. 损失反向传播 → 更新 AASIST 参数 (Adam 优化器)',
    '6. 定期更新说话人中心：bona 数据 → AASIST 提取特征 → 按说话人取均值 → 更新中心向量',
]
for step in flow_steps:
    doc.add_paragraph(step, style='List Number')

doc.add_heading('5.2 测试阶段数据流', level=2)
doc.add_paragraph('测试时的数据流与训练类似，但有两点关键区别：')

doc.add_paragraph(
    '模型处于 eval() 模式（关闭 Dropout 和 BatchNorm 的训练行为），且使用 torch.no_grad() '
    '关闭梯度计算以节省显存和加速推理',
    style='List Bullet'
)
doc.add_paragraph(
    '说话人中心来自注册集（eval_enroll），而非训练集。每个评估集说话人通过其注册音频提取特征后取均值得到中心',
    style='List Bullet'
)
doc.add_paragraph(
    '评分公式（attractor=1 模式）：score = normalize(feat) · normalize(speaker_center)，'
    '即特征向量与对应说话人中心的余弦相似度。分数越高表示越可能是真实语音',
    style='List Bullet'
)

doc.add_heading('5.3 音频预处理详解', level=2)
doc.add_paragraph(
    '项目中有两种音频处理模式：'
)
doc.add_paragraph(
    '训练数据加载 (data_utils.py)：直接从 FLAC 文件读取，使用 pad_random() 进行随机裁剪/填充。'
    '没有额外的特征提取步骤（如 MFCC 等），因为 AASIST 是端到端模型，直接从原始波形学习。',
    style='List Bullet'
)
doc.add_paragraph(
    'VCTK 测试数据加载 (test_vctk_samo.py)：从 WAV 文件读取，使用 librosa 重采样到 16kHz，'
    '然后将立体声转为单声道。使用固定 pad() 确保确定性测试结果。',
    style='List Bullet'
)

doc.add_page_break()

# ===== 6. 关键参数速查表 =====
doc.add_heading('6. 关键参数速查表', level=1)

doc.add_heading('6.1 训练参数 (main.py)', level=2)
train_params = doc.add_table(rows=16, cols=3, style='Light List Accent 1')
for i, h in enumerate(['参数', '默认值', '说明']):
    train_params.rows[0].cells[i].text = h
    for p in train_params.rows[0].cells[i].paragraphs:
        p.runs[0].font.bold = True
tp_data = [
    ('--seed', '10', '随机种子，保证实验可复现'),
    ('--num_epochs', '100', '训练总轮数'),
    ('--batch_size', '23', '每批次样本数（受 11GB 显存限制）'),
    ('--lr', '0.0001', '初始学习率'),
    ('--lr_min', '0.000005', '余弦退火的最小学习率'),
    ('--scheduler', 'cosine2', '学习率调度策略'),
    ('--enc_dim', '160', '特征嵌入维度'),
    ('--loss', 'samo', '损失函数选择 (softmax/ocsoftmax/samo)'),
    ('--num_centers', '20', 'SAMO 中心向量数'),
    ('--m_real', '0.7', '真实语音的 margin'),
    ('--m_fake', '0.0', '伪造语音的 margin'),
    ('--alpha', '20', '损失缩放因子'),
    ('--train_sp', '1', '训练时的 SAMO 模式 (0/1/2)'),
    ('--val_sp', '1', '验证/测试时的 SAMO 模式 (0/1/2)'),
    ('--update_interval', '3', '中心向量更新间隔（epoch）'),
]
for i, (param, val, desc) in enumerate(tp_data):
    train_params.rows[i + 1].cells[0].text = param
    train_params.rows[i + 1].cells[1].text = val
    train_params.rows[i + 1].cells[2].text = desc

doc.add_paragraph()
doc.add_heading('6.2 测试参数 (test_vctk_samo.py)', level=2)
test_params = doc.add_table(rows=8, cols=3, style='Light List Accent 1')
for i, h in enumerate(['参数', '默认值', '说明']):
    test_params.rows[0].cells[i].text = h
    for p in test_params.rows[0].cells[i].paragraphs:
        p.runs[0].font.bold = True
tp2_data = [
    ('-m / --model_path', '(必填)', '预训练模型路径，如 ./models/samo.pt'),
    ('--loss_type', 'samo', '模型类型 (samo/ocsoftmax)'),
    ('--samo_mode', 'with_enroll', 'SAMO 测试模式 (with_enroll/no_enroll)'),
    ('--vctk_clean', './VCTK-Corpus', 'VCTK 真实语音目录'),
    ('--vctk_attack', './VCTK-Corpus-attack', 'VCTK 伪造语音目录'),
    ('--max_len', '64600', '音频截取长度（采样点数）'),
    ('--batch_size', '16', '测试批次大小'),
]
for i, (param, val, desc) in enumerate(tp2_data):
    test_params.rows[i + 1].cells[0].text = param
    test_params.rows[i + 1].cells[1].text = val
    test_params.rows[i + 1].cells[2].text = desc

doc.add_page_break()

# ===== 7. 如何运行项目 =====
doc.add_heading('7. 如何运行项目', level=1)

doc.add_heading('7.1 环境准备', level=2)
doc.add_paragraph('硬件要求：1 块 GPU（如 GTX 1080 Ti），约 11GB 显存')
doc.add_paragraph('软件环境：Python 3.6+，PyTorch >= 1.6.0，CUDA 支持')

doc.add_heading('安装依赖', level=3)
p = doc.add_paragraph()
run = p.add_run('pip install -r requirements.txt')
run.font.name = 'Consolas'
run.font.size = Pt(10)

doc.add_heading('7.2 训练模型', level=2)
doc.add_paragraph('训练 SAMO 模型的命令示例：')
p = doc.add_paragraph()
run = p.add_run(
    'python3 samo/main.py -o "output_folder" -d "path_to_LA_dataset" '
    '-p "path_to_protocols" --overwrite'
)
run.font.name = 'Consolas'
run.font.size = Pt(10)
doc.add_paragraph(
    '其中：-o 指定输出目录，-d 指定 ASVspoof 2019 LA 数据集路径（包含 LA/ 子目录），'
    '-p 指定协议文件目录（项目已自带）。'
    '可通过 --loss 参数选择损失函数：softmax（标准交叉熵）、ocsoftmax（单中心单类学习）、samo（默认，多中心说话人感知）。'
)

doc.add_heading('7.3 评估预训练模型', level=2)
doc.add_paragraph('使用项目自带的预训练模型 samo.pt 进行评估：')
p = doc.add_paragraph()
run = p.add_run(
    'python3 samo/main.py --test_only --test_model "./models/samo.pt" '
    '--scoring "samo" --save_score "samo_score"'
)
run.font.name = 'Consolas'
run.font.size = Pt(10)
doc.add_paragraph('预期输出：Test EER: 0.008751418248624953（约 0.88%）')

doc.add_heading('7.4 在 VCTK 数据集上测试', level=2)
doc.add_paragraph('准备数据：')
p = doc.add_paragraph()
run = p.add_run(
    'python build_test_sets.py --vctk_root "D:\\testone\\VCTK-VC" '
    '--output_root ./test_sets --num 1500'
)
run.font.name = 'Consolas'
run.font.size = Pt(10)
doc.add_paragraph('运行测试：')
p = doc.add_paragraph()
run = p.add_run(
    'python samo/test_vctk_samo.py -m ./models/samo.pt --loss_type samo '
    '--samo_mode with_enroll --vctk_clean ./test_sets/test_adain/clean '
    '--vctk_attack ./test_sets/test_adain/spoof -o results_adain.txt'
)
run.font.name = 'Consolas'
run.font.size = Pt(10)

doc.add_page_break()

# ===== 8. 技术要点总结 =====
doc.add_heading('8. 技术要点总结', level=1)

doc.add_heading('8.1 SAMO 方法的创新点', level=2)
points = [
    '说话人感知 (Speaker-Aware)：利用说话人信息为每个说话人学习独立的中心向量，更精确地建模真实语音分布。',
    '多中心设计 (Multi-Center)：相比传统单类学习的单中心方法，多中心能更好地捕捉说话人间的差异性。',
    '动态中心更新：训练过程中定期用最新的模型参数更新说话人中心，保持中心向量的时效性。',
    '灵活的评分策略：支持说话人感知（1-on-1）和说话人无关（maxscore）两种评分方式，适应不同应用场景。',
]
for point in points:
    doc.add_paragraph(point, style='List Bullet')

doc.add_heading('8.2 为什么 EER 只有 0.88%？', level=2)
doc.add_paragraph(
    '这不是模型不够好，相反，0.88% 的 EER 在 ASVspoof 2019 LA 任务上是非常优秀的结果。'
    '这个低错误率体现了 SAMO 方法的有效性，但也需要注意：ASVspoof 2019 是一个实验室环境下的基准测试，'
    '在更复杂的真实场景（如信道噪声、压缩编码、多说话人混叠等）中，性能可能会下降。'
    '论文中报告的各攻击类型分解分析显示，对于某些特定攻击（如 A17），EER 可能高达 3% 以上，'
    '说明模型对某些攻击类型的检测能力仍然有限。'
)

doc.add_heading('8.3 项目的三个层次理解', level=2)

doc.add_paragraph(
    '硬件层：GPU 上的大规模矩阵运算。AASIST 的核心操作包括卷积（Conv1d/Conv2d）、'
    '矩阵乘法（注意力机制、相似度计算）、池化、归一化等——本质上都是 PyTorch 高效实现的线性代数运算。',
    style='List Bullet'
)
doc.add_paragraph(
    '架构层：AASIST 骨干网络 + SAMO 损失函数。端到端设计，从原始波形到最终检测结果一气呵成。'
    'AASIST 负责"提取有区分力的特征"，SAMO 负责"用正确的目标函数来训练这些特征"。',
    style='List Bullet'
)
doc.add_paragraph(
    '概念层：Single-Class Learning（单类学习）在语音防伪领域的应用。与其试图枚举所有可能的攻击类型，'
    '不如让模型深刻理解"什么是真实的人类语音"——这是 SAMO 方法论的核心哲学。',
    style='List Bullet'
)

doc.add_heading('8.4 扩展方向', level=2)
doc.add_paragraph(
    '如果你有兴趣在此项目基础上进一步学习或研究，可以考虑以下方向：'
)
doc.add_paragraph(
    '直接在本代码上运行，对比 Softmax、OC-Softmax 和 SAMO 三种损失函数的实际表现差异。',
    style='List Bullet'
)
doc.add_paragraph(
    '在 VCTK 等多个数据集上评估模型的跨库泛化能力——SAMO 论文预训练模型在 ASVspoof 2019 上训练，'
    '观察它在 VCTK 数据集上的表现能反映模型的泛化能力。',
    style='List Bullet'
)
doc.add_paragraph(
    '尝试替换骨干网络（如用 RawNet2、RawGAT-ST 等），观察不同特征提取器对性能的影响。',
    style='List Bullet'
)
doc.add_paragraph(
    '调整中心更新策略、margin 参数、中心数量等，研究这些超参数对性能的影响。',
    style='List Bullet'
)
doc.add_paragraph(
    '阅读相关研究：OC-Softmax（One-Class Learning Towards Synthetic Voice Spoofing Detection, SPL 2021）'
    '是 SAMO 的直接前驱，AASIST（Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks, 2021）'
    '提供了强大的特征提取基础。',
    style='List Bullet'
)

doc.add_paragraph()

# ===== 附录：术语表 =====
doc.add_heading('附录: 术语缩写对照表', level=1)
terms = doc.add_table(rows=18, cols=2, style='Light List Accent 1')
for i, h in enumerate(['术语/缩写', '全称与解释']):
    terms.rows[0].cells[i].text = h
    for p in terms.rows[0].cells[i].paragraphs:
        p.runs[0].font.bold = True
term_data = [
    ('SAMO', 'Speaker Attractor Multi-Center One-Class Learning（说话人吸引力子多中心单类学习）'),
    ('AASIST', 'Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'),
    ('ASV', 'Automatic Speaker Verification（自动说话人验证），即声纹识别'),
    ('CM', 'Countermeasure（防伪对策），在语音防伪任务中指区分真假的检测模块'),
    ('Bonafide', '真实的、非伪造的（指真实人类语音）'),
    ('Spoof', '伪造的、欺骗性的（指 TTS/VC 等生成的假语音）'),
    ('EER', 'Equal Error Rate（等错误率），FRR=FAR 时的错误率，越低越好'),
    ('t-DCF', 'Tandem Detection Cost Function（串联检测代价函数），评估 CM+ASV 串联系统的综合代价'),
    ('FRR / FAR', 'False Rejection Rate（假拒率）/ False Acceptance Rate（假接率）'),
    ('OC-Softmax', 'One-Class Softmax（单类 Softmax），单中心单类学习基线方法'),
    ('GAT', 'Graph Attention Network（图注意力网络），AASIST 中的核心模块之一'),
    ('SincConv', '基于 sinc 函数的卷积层，AASIST 的第一层，直接处理原始波形'),
    ('Embedding', '嵌入向量，指神经网络提取的高维特征表示（本项目为 160 维）'),
    ('Attractor', '吸引力子，即说话人中心向量，在特征空间中作为真实语音的"吸引力"锚点'),
    ('Enrollment', '注册，指用已知说话人的音频提取其特征中心向量的过程'),
    ('margin', '边界值，真实语音分数需高于此值，伪造语音分数需低于此值'),
    ('TTS / VC', 'Text-to-Speech（文本转语音）/ Voice Conversion（语音转换），两种主要的语音伪造方法'),
]
for i, (term, explanation) in enumerate(term_data):
    terms.rows[i + 1].cells[0].text = term
    terms.rows[i + 1].cells[1].text = explanation

doc.add_paragraph()

# ===== 保存文档 =====
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'SAMO项目架构与代码详解.docx')
doc.save(output_path)
print(f'文档已保存至: {output_path}')
