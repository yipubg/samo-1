"""
VCTK 数据集 SAMO 测试脚本（增强版）
==================================
本脚本用于在 VCTK 数据集上评估预训练的 SAMO 模型，输出完整的检测指标。

功能:
  1. VCTKRawDataset: 加载 VCTK 原始音频的自定义 Dataset
  2. VCTKTester:     封装模型加载、说话人 attractor 计算和评估逻辑
  3. main():         命令行入口，支持多种 SAMO 测试模式

测试模式:
  - with_enroll (推荐): 用 clean 音频预计算每个说话人的 attractor，
                        然后用 1-on-1 相似度评分
  - no_enroll:          使用模型内部训练好的中心向量，用 maxscore 评分

输出指标:
  - EER (等错误率)
  - Accuracy, Precision, Recall, F1-Score
  - 混淆矩阵 (TP/TN/FP/FN)
  - 分数分布统计
  - 逐条检测结果（保存到文件）

示例用法:
  # SAMO with enrollment（推荐）
  python test_vctk_samo.py -m ./models/samo.pt --loss_type samo --samo_mode with_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack

  # SAMO without enrollment
  python test_vctk_samo.py -m ./models/samo.pt --loss_type samo --samo_mode no_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
import re

# 导入 SAMO 相关模块
try:
    from samo.loss import SAMO
    from samo.aasist import AASIST
except ImportError:
    from loss import SAMO
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from samo.aasist import AASIST

import eval_metrics as em


# ============================================================================
# 1. 音频处理工具函数
# ============================================================================

def pad(x, max_len=64600):
    """
    固定长度填充（用于测试，保证确定性）。

    短音频重复拼接，长音频截取前 max_len 个采样点。
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    """
    随机裁剪或填充（用于训练时的数据增强）。

    长音频随机选择起始点截取，短音频重复拼接。
    """
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


# ============================================================================
# 2. VCTK 原始音频数据集
# ============================================================================

class VCTKRawDataset(Dataset):
    """
    VCTK 数据集类 —— 加载原始 WAV 音频，适配 AASIST 端到端模型。

    与 ASVspoof2019_speaker_raw 的区别:
      - 处理 WAV 格式而非 FLAC
      - 支持重采样到 16kHz
      - 支持立体声转单声道
      - 从目录名中提取说话人 ID

    参数:
        vctk_clean_dir:  真实语音目录路径
        vctk_attack_dir: 伪造语音目录路径
        max_len:         目标音频长度（采样点数）
        sr:              目标采样率（AASIST 标准为 16000）
        load_clean:      是否加载真实语音
        load_attack:     是否加载伪造语音
    """

    def __init__(self, vctk_clean_dir, vctk_attack_dir=None,
                 max_len=64600, sr=16000,
                 load_clean=True, load_attack=True):
        self.max_len = max_len
        self.sr = sr
        self.samples = []  # [(filepath, label, speaker_id), ...]

        # 加载真实语音 (label=0)
        if load_clean and vctk_clean_dir and os.path.exists(vctk_clean_dir):
            print(f"Scanning clean audio from: {vctk_clean_dir}")
            clean_samples = self._scan_vctk_directory(vctk_clean_dir, label=0)
            self.samples.extend(clean_samples)
            print(f"  Found {len(clean_samples)} clean samples")

        # 加载伪造语音 (label=1)
        if load_attack and vctk_attack_dir and os.path.exists(vctk_attack_dir):
            print(f"Scanning attacked audio from: {vctk_attack_dir}")
            attack_samples = self._scan_vctk_directory(vctk_attack_dir, label=1)
            self.samples.extend(attack_samples)
            print(f"  Found {len(attack_samples)} attacked samples")

        if len(self.samples) == 0:
            raise ValueError("未找到任何音频文件！请检查目录路径。")

        print(f"\nTotal dataset size: {len(self.samples)} samples")
        print(f"  Bonafide (0): {sum(1 for s in self.samples if s[1] == 0)}")
        print(f"  Spoof (1): {sum(1 for s in self.samples if s[1] == 1)}")

    def _scan_vctk_directory(self, root_dir, label):
        """
        递归扫描目录，找到所有 WAV 文件。

        VCTK 目录结构示例:
            VCTK-Corpus/p225/p225_001.wav
        从目录名中提取说话人 ID (如 p225)。
        """
        samples = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.wav'):
                    filepath = os.path.join(dirpath, filename)
                    # 从路径中提取说话人 ID: 匹配 p + 数字
                    match = re.search(r'p(\d+)', dirpath)
                    speaker_id = match.group(0) if match else 'unknown'
                    samples.append((filepath, label, speaker_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        加载并预处理单条音频。

        处理步骤:
          1. soundfile 读取 WAV 文件
          2. 如有需要，重采样到目标采样率 (16kHz)
          3. 立体声 → 单声道（取左右声道均值）
          4. 裁剪/填充到固定长度

        返回:
            audio_tensor: (max_len,) 的浮点张量
            filename:     音频文件名
            label:        0=真实, 1=伪造
            speaker_id:   说话人 ID
            filepath:     文件的完整路径
        """
        filepath, label, speaker_id = self.samples[idx]
        try:
            audio, sr = sf.read(filepath)

            # 重采样到 16kHz（AASIST 标准采样率）
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

            # 立体声转单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 裁剪/填充至固定长度
            audio = pad(audio, self.max_len)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            audio = np.zeros(self.max_len)

        audio_tensor = torch.from_numpy(audio).float()
        filename = os.path.basename(filepath)
        return audio_tensor, filename, label, speaker_id, filepath

    def collate_fn(self, samples):
        """
        自定义批次组装函数。

        将列表中的单个样本组合成一个批次。
        """
        audios = torch.stack([s[0] for s in samples])
        filenames = [s[1] for s in samples]
        labels = torch.tensor([s[2] for s in samples])
        speaker_ids = [s[3] for s in samples]
        filepaths = [s[4] for s in samples]
        return audios, filenames, labels, speaker_ids, filepaths


# ============================================================================
# 3. SAMO 测试引擎
# ============================================================================

class VCTKTester:
    """
    VCTK 数据集上的 SAMO 模型测试器。

    封装了:
      - 模型加载（支持多种检查点格式）
      - 说话人 attractor 预计算（with_enroll 模式）
      - 批量评估与指标计算
    """

    def __init__(self, model_path, add_loss='samo', enc_dim=160, device='cuda'):
        """
        参数:
            model_path: 预训练模型文件路径 (.pt 或 .pth)
            add_loss:   损失函数类型 ('samo' 或 'ocsoftmax')
            enc_dim:    特征嵌入维度（默认 160）
            device:     计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.add_loss = add_loss
        self.enc_dim = enc_dim

        print(f"Loading model from: {model_path}")

        # 支持多种模型保存格式
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                # 字典格式: {'feat_model': ..., 'loss_model': ...}
                if 'feat_model' in checkpoint:
                    self.model = checkpoint['feat_model'].to(self.device)
                    self.loss_model = checkpoint.get('loss_model', None)
                    if self.loss_model:
                        self.loss_model.to(self.device)
                        self.loss_model.eval()
                else:
                    self.model = checkpoint.to(self.device)
                    self.loss_model = None
            else:
                # 直接保存的模型对象
                self.model = checkpoint.to(self.device)
                self.loss_model = None

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.model.eval()
        self.use_samo = (add_loss == 'samo')
        self.speaker_attractors = None

        print(f"Model loaded. Device: {self.device}")
        print(f"Loss type: {add_loss}")

    @torch.no_grad()
    def compute_speaker_attractors(self, dataloader):
        """
        从 clean 音频为每个说话人预计算中心向量 (Speaker Attractor)。

        这是 SAMO with_enroll 模式的核心准备步骤。

        处理流程:
          1. 遍历所有 clean 音频，提取 AASIST embedding
          2. 按说话人分组累积 embedding
          3. 对每个说话人的 embedding 取均值
          4. L2 归一化得到最终的 attractor

        返回:
            attractors: {speaker_id: normalized_tensor} 字典
        """
        print("\n" + "=" * 60)
        print("Computing Speaker Attractors for SAMO (with enrollment mode)")
        print("=" * 60)

        enroll_emb_dict = {}

        for audios, filenames, labels, speaker_ids, filepaths in tqdm(dataloader, desc="Computing attractors"):
            # 只使用 clean 音频（label=0）
            mask = labels == 0
            if mask.sum() == 0:
                continue

            clean_audios = audios[mask].to(self.device)
            clean_spks = [speaker_ids[i] for i in range(len(mask)) if mask[i]]

            # 提取 AASIST embedding（160 维特征向量）
            features, _ = self.model(clean_audios)
            features = features.cpu()

            # 按说话人累积 embedding
            for s, feat in zip(clean_spks, features):
                if s not in enroll_emb_dict:
                    enroll_emb_dict[s] = []
                enroll_emb_dict[s].append(feat)

        # 对每个说话人的所有 embedding 取均值 → L2 归一化
        attractors = {}
        for spk in enroll_emb_dict:
            mean_emb = torch.stack(enroll_emb_dict[spk]).mean(dim=0)
            mean_emb = F.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            attractors[spk] = mean_emb

        print(f"Computed attractors for {len(attractors)} speakers")
        for spk in sorted(attractors.keys())[:5]:
            print(f"  {spk}: norm = {attractors[spk].norm().item():.4f}")
        print("=" * 60)

        return attractors

    @torch.no_grad()
    def evaluate(self, dataloader, output_file='vctk_results.txt',
                 save_detailed=True, samo_attractors=None):
        """
        在测试集上运行全面评估。

        评分策略（按优先级）:
          1. SAMO + attractors:    使用预计算的说话人中心，1-on-1 余弦相似度
          2. SAMO + loss_model:    使用 loss_model 中的训练中心，maxscore
          3. 其他 (OCSoftmax/FC):  使用 softmax 概率作为分数

        输出的指标包含:
          - 总体 EER、Accuracy、Precision、Recall、F1-Score
          - 混淆矩阵 (TP/TN/FP/FN)
          - 分数分布统计（均值、标准差、极值）
          - 每个说话人的详细统计

        参数:
            dataloader:      测试数据加载器
            output_file:     逐条结果输出文件路径
            save_detailed:   是否保存详细信息
            samo_attractors: 预计算的说话人中心字典（with_enroll 模式）

        返回:
            results_dict: 包含所有指标的字典
        """
        all_scores = []
        all_labels = []
        speaker_stats = {}
        bonafide_scores = []
        spoof_scores = []

        # 写入结果文件头
        with open(output_file, 'w') as f_out:
            header = "filename\tspeaker_id\ttrue_label\tpredicted_label\tscore\tfile_path"
            if self.use_samo and samo_attractors is not None:
                header += "\tmode=with_enroll"
            f_out.write(header + "\n")

            # ---- 逐批次推理 ----
            for audios, filenames, labels, speaker_ids, filepaths in tqdm(dataloader, desc="Testing"):
                audios = audios.to(self.device)
                features, outputs = self.model(audios)

                # ---- 计算每条样本的分数 ----
                if self.use_samo:
                    if samo_attractors is not None:
                        # SAMO with_enroll: 1-on-1 余弦相似度
                        batch_enrolls = []
                        for spk in speaker_ids:
                            if spk in samo_attractors:
                                batch_enrolls.append(samo_attractors[spk])
                            else:
                                # 未知说话人使用零向量（将被判定为伪造）
                                batch_enrolls.append(torch.zeros(self.enc_dim))

                        batch_enrolls = torch.stack(batch_enrolls).to(self.device)
                        batch_enrolls = F.normalize(batch_enrolls, p=2, dim=1)
                        # 内积 = 余弦相似度（向量已归一化）
                        final_scores = torch.sum(features * batch_enrolls, dim=1).cpu().numpy()
                    else:
                        # SAMO no_enroll: 使用 loss_model 的训练中心
                        if self.loss_model and hasattr(self.loss_model, 'center'):
                            w = F.normalize(self.loss_model.center, p=2, dim=1).to(self.device)
                            scores = features @ w.transpose(0, 1)    # (B, num_centers)
                            final_scores, _ = torch.max(scores, dim=1)  # maxscore
                            final_scores = final_scores.cpu().numpy()
                        else:
                            # 退回到 softmax 概率
                            probs = F.softmax(outputs, dim=1)
                            final_scores = probs[:, 0].cpu().numpy()
                else:
                    # OCSoftmax / 基础模型: 使用分类概率
                    probs = F.softmax(outputs, dim=1)
                    final_scores = probs[:, 0].cpu().numpy()

                # ---- 记录每一条样本的结果 ----
                for i in range(len(filenames)):
                    score = final_scores[i]
                    true_label = labels[i].item()
                    pred_label = 0 if score > 0.5 else 1  # 分数 > 0.5 → 真实

                    f_out.write(f"{filenames[i]}\t{speaker_ids[i]}\t{true_label}\t"
                                f"{pred_label}\t{score:.6f}\t{filepaths[i]}\n")

                    all_scores.append(score)
                    all_labels.append(true_label)

                    # 按说话人统计
                    if speaker_ids[i] not in speaker_stats:
                        speaker_stats[speaker_ids[i]] = {'bonafide': [], 'spoof': []}

                    if true_label == 0:
                        bonafide_scores.append(score)
                        speaker_stats[speaker_ids[i]]['bonafide'].append(score)
                    else:
                        spoof_scores.append(score)
                        speaker_stats[speaker_ids[i]]['spoof'].append(score)

        # ================================================================
        # 计算评估指标
        # ================================================================
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        bona_scores = all_scores[all_labels == 0]
        spoof_scores_arr = all_scores[all_labels == 1]

        # EER 和最优阈值
        eer, threshold = em.compute_eer(bona_scores, spoof_scores_arr)
        predictions = np.where(all_scores > threshold, 0, 1)  # >阈值 → 真实(0)

        # 混淆矩阵（伪造=正类 Positive）
        TP = int(np.sum((predictions == 1) & (all_labels == 1)))  # 伪造被正确检出
        TN = int(np.sum((predictions == 0) & (all_labels == 0)))  # 真实被正确通过
        FP = int(np.sum((predictions == 1) & (all_labels == 0)))  # 真实被误判为伪造
        FN = int(np.sum((predictions == 0) & (all_labels == 1)))  # 伪造被漏过

        # 四项基本指标
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # ---- 打印结果 ----
        print(f"\n{'=' * 60}")
        print(f"VCTK测试完成！结果保存至: {output_file}")
        print(f"{'=' * 60}")
        print(f"总体统计:")
        print(f"  总样本数: {len(all_labels)}")
        print(f"  真实语音 (Bonafide): {len(bona_scores)}")
        print(f"  欺骗语音 (Spoof): {len(spoof_scores_arr)}")
        print(f"  说话人数量: {len(speaker_stats)}")

        if self.use_samo and samo_attractors is not None:
            print(f"\nSAMO Mode: With Enrollment")
        elif self.use_samo:
            print(f"\nSAMO Mode: Without Enrollment")

        print(f"\n性能指标:")
        print(f"  EER (Equal Error Rate): {eer * 100:.2f}%")
        print(f"  EER Threshold: {threshold:.4f}")
        print(f"  Accuracy:  {accuracy * 100:.2f}%")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"\n  [混淆矩阵] 伪造=正类(Positive)")
        print(f"    TP(伪造检出): {TP:>5}    FP(真实误杀): {FP:>5}")
        print(f"    FN(伪造漏网): {FN:>5}    TN(真实通过): {TN:>5}")
        print(f"\n  Bonafide平均分数: {np.mean(bona_scores):.4f} ± {np.std(bona_scores):.4f}")
        print(f"  Spoof平均分数: {np.mean(spoof_scores_arr):.4f} ± {np.std(spoof_scores_arr):.4f}")
        print(f"{'=' * 60}")

        # ---- 组装结果字典 ----
        results_dict = {
            'eer': eer * 100,
            'threshold': float(threshold),
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': {
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
            },
            'sample_stats': {
                'total': int(len(all_labels)),
                'bonafide_count': int(len(bona_scores)),
                'spoof_count': int(len(spoof_scores_arr)),
                'speaker_count': int(len(speaker_stats))
            },
            'score_distribution': {
                'bonafide_mean': float(np.mean(bona_scores)),
                'bonafide_std': float(np.std(bona_scores)),
                'bonafide_min': float(np.min(bona_scores)),
                'bonafide_max': float(np.max(bona_scores)),
                'spoof_mean': float(np.mean(spoof_scores_arr)),
                'spoof_std': float(np.std(spoof_scores_arr)),
                'spoof_min': float(np.min(spoof_scores_arr)),
                'spoof_max': float(np.max(spoof_scores_arr))
            }
        }

        return results_dict


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    """
    命令行入口。

    使用流程:
      1. 创建 VCTKRawDataset（加载 clean + attack 音频）
      2. 创建 VCTKTester（加载预训练 SAMO 模型）
      3. with_enroll 模式下，预计算说话人 attractors
      4. 运行评估，输出所有指标
      5. 保存结果到 JSON 和 TXT 文件
    """
    parser = argparse.ArgumentParser(
        description='在 VCTK 数据集上测试 SAMO/AASIST 模型（原始音频）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # SAMO with enrollment（推荐用于VCTK）
  python test_vctk_samo.py -m ./models/samo.pt --loss_type samo --samo_mode with_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack

  # SAMO without enrollment
  python test_vctk_samo.py -m ./models/samo.pt --loss_type samo --samo_mode no_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack
        """
    )

    # ---- 模型参数 ----
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='预训练模型路径，如 ./models/samo.pt')
    parser.add_argument('--loss_type', type=str, default='samo',
                        choices=['samo', 'ocsoftmax'],
                        help='模型类型: samo 或 ocsoftmax')
    parser.add_argument('--samo_mode', type=str, default='with_enroll',
                        choices=['with_enroll', 'no_enroll'],
                        help='SAMO 测试模式: with_enroll(推荐) 或 no_enroll')

    # ---- 数据路径 ----
    parser.add_argument('--vctk_clean', type=str, default='./VCTK-Corpus',
                        help='VCTK 真实语音目录')
    parser.add_argument('--vctk_attack', type=str, default='./VCTK-Corpus-attack',
                        help='VCTK 伪造语音目录')
    parser.add_argument('--no_clean', action='store_true',
                        help='不加载真实语音')
    parser.add_argument('--no_attack', action='store_true',
                        help='不加载伪造语音')

    # ---- 音频处理参数 ----
    parser.add_argument('--max_len', type=int, default=64600,
                        help='音频截取长度（采样点数），默认 64600 ≈ 4秒 @16kHz')
    parser.add_argument('--enc_dim', type=int, default=160,
                        help='特征嵌入维度，默认 160')
    parser.add_argument('--sr', type=int, default=16000,
                        help='目标采样率，默认 16000 Hz')

    # ---- 运行参数 ----
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='测试批次大小')
    parser.add_argument('-o', '--output', type=str, default='vctk_results.txt',
                        help='逐条检测结果输出文件')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的 GPU 编号')

    args = parser.parse_args()

    # 参数校验
    if args.no_clean and args.no_attack:
        parser.error("不能同时跳过 clean 和 attack 数据！")
    if args.no_clean and args.samo_mode == 'with_enroll':
        parser.error("with_enroll 模式需要 clean 音频来计算 attractor！")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---- 创建测试数据集 ----
    dataset = VCTKRawDataset(
        vctk_clean_dir=args.vctk_clean if not args.no_clean else None,
        vctk_attack_dir=args.vctk_attack if not args.no_attack else None,
        max_len=args.max_len,
        sr=args.sr,
        load_clean=not args.no_clean,
        load_attack=not args.no_attack
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate_fn)

    # ---- 初始化测试器 ----
    tester = VCTKTester(
        model_path=args.model_path,
        add_loss=args.loss_type,
        enc_dim=args.enc_dim
    )

    # ---- 预计算说话人 attractors（with_enroll 模式）----
    samo_attractors = None
    if args.loss_type == 'samo' and args.samo_mode == 'with_enroll':
        # 创建一个只含 clean 音频的数据集，用于计算 attractors
        clean_dataset = VCTKRawDataset(
            vctk_clean_dir=args.vctk_clean,
            vctk_attack_dir=None,
            max_len=args.max_len,
            sr=args.sr,
            load_clean=True,
            load_attack=False
        )
        clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=clean_dataset.collate_fn)
        samo_attractors = tester.compute_speaker_attractors(clean_loader)

    # ---- 运行测试 ----
    results = tester.evaluate(dataloader, output_file=args.output,
                              samo_attractors=samo_attractors)

    # ---- 保存检测指标结果 ----
    output_base = os.path.splitext(args.output)[0]
    summary_json = output_base + '_summary.json'
    summary_txt = output_base + '_summary.txt'

    # JSON 格式（方便程序读取和后续分析）
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n指标摘要已保存至 (JSON): {summary_json}")

    # TXT 格式（方便直接查看和粘贴到报告）
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"SAMO Voice Spoofing Detection Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"EER:           {results['eer']:.2f}%\n")
        f.write(f"Threshold:     {results['threshold']:.4f}\n")
        f.write(f"Accuracy:      {results['accuracy']:.2f}%\n")
        f.write(f"Precision:     {results['precision']:.2f}%\n")
        f.write(f"Recall:        {results['recall']:.2f}%\n")
        f.write(f"F1-Score:      {results['f1']:.2f}%\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TP: {results['confusion_matrix']['TP']}\n")
        f.write(f"  TN: {results['confusion_matrix']['TN']}\n")
        f.write(f"  FP: {results['confusion_matrix']['FP']}\n")
        f.write(f"  FN: {results['confusion_matrix']['FN']}\n")
        f.write(f"\nScore Distribution:\n")
        f.write(f"  Bonafide: {results['score_distribution']['bonafide_mean']:.4f} ± "
                f"{results['score_distribution']['bonafide_std']:.4f}\n")
        f.write(f"  Spoof:    {results['score_distribution']['spoof_mean']:.4f} ± "
                f"{results['score_distribution']['spoof_std']:.4f}\n")
    print(f"指标摘要已保存至 (TXT): {summary_txt}")

    print(f"\n所有结果保存完成:")
    print(f"   逐条检测结果: {args.output}")
    print(f"   指标摘要JSON: {summary_json}")
    print(f"   指标摘要TXT:  {summary_txt}")


if __name__ == "__main__":
    main()
