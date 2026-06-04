"""
VCTK 数据集 SAMO 测试脚本（基础版）
==================================
本脚本用于在 VCTK 数据集上评估预训练的 SAMO 模型。

与 test_vctk_samo.py 的区别:
  本文件是早期版本，功能较为基础；
  test_vctk_samo.py 是增强版，包含更完整的指标计算和结果保存功能。

功能:
  1. VCTKRawDataset: 加载 VCTK 原始音频的自定义 Dataset
  2. VCTKTester:     封装模型加载、说话人 attractor 计算和评估逻辑
  3. main():         命令行入口

测试模式:
  - with_enroll (推荐): 用 clean 音频预计算每个说话人的 attractor
  - no_enroll:          使用 loss_model 中的训练中心

示例用法:
  python test_vctk.py -m ./models/samo.pt --loss_type samo --samo_mode with_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack
"""

import argparse
import os
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
    固定长度填充（用于测试）。

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
    随机裁剪或填充（用于保持一致性，测试时使用 pad()）。

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

    与 test_vctk_samo.py 中的同名类功能相同。

    参数:
        vctk_clean_dir:  真实语音目录
        vctk_attack_dir: 伪造语音目录
        max_len:         目标音频长度（采样点数）
        sr:              目标采样率
        load_clean:      是否加载真实语音
        load_attack:     是否加载伪造语音
    """

    def __init__(self, vctk_clean_dir, vctk_attack_dir=None,
                 max_len=64600, sr=16000,
                 load_clean=True, load_attack=True):
        self.max_len = max_len
        self.sr = sr
        self.samples = []

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
        """递归扫描目录，找到所有 WAV 文件，提取说话人 ID"""
        samples = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.wav'):
                    filepath = os.path.join(dirpath, filename)
                    match = re.search(r'p(\d+)', dirpath)
                    speaker_id = match.group(0) if match else 'unknown'
                    samples.append((filepath, label, speaker_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """加载并预处理单条音频"""
        filepath, label, speaker_id = self.samples[idx]

        try:
            audio, sr = sf.read(filepath)

            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = pad(audio, self.max_len)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            audio = np.zeros(self.max_len)

        audio_tensor = torch.from_numpy(audio).float()
        filename = os.path.basename(filepath)
        return audio_tensor, filename, label, speaker_id, filepath

    def collate_fn(self, samples):
        """自定义批次组装"""
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
    VCTK 数据集上 SAMO 模型的测试器（基础版）。

    支持两种模式:
      - with_enroll: 预计算说话人 attractor，1-on-1 打分
      - no_enroll:   使用 loss_model 中的训练中心，maxscore 打分
    """

    def __init__(self, model_path, add_loss='samo', enc_dim=160, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.add_loss = add_loss
        self.enc_dim = enc_dim

        print(f"Loading model from: {model_path}")

        # 兼容多种模型保存格式
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
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
                self.model = checkpoint.to(self.device)
                self.loss_model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.model.eval()
        self.use_samo = (add_loss == 'samo')
        print(f"Model loaded. Device: {self.device}")
        print(f"Loss type: {add_loss}")

    @torch.no_grad()
    def compute_speaker_attractors(self, dataloader):
        """
        从 clean 音频预计算每个说话人的中心向量 (Speaker Attractor)。

        流程: 提取 embedding → 按说话人取均值 → L2 归一化
        """
        print("\n" + "=" * 60)
        print("Computing Speaker Attractors for SAMO (with enrollment mode)")
        print("=" * 60)

        enroll_emb_dict = {}

        for audios, filenames, labels, speaker_ids, filepaths in tqdm(dataloader, desc="Computing attractors"):
            mask = labels == 0
            if mask.sum() == 0:
                continue

            clean_audios = audios[mask].to(self.device)
            clean_spks = [speaker_ids[i] for i in range(len(mask)) if mask[i]]

            features, _ = self.model(clean_audios)
            features = features.cpu()

            for s, feat in zip(clean_spks, features):
                if s not in enroll_emb_dict:
                    enroll_emb_dict[s] = []
                enroll_emb_dict[s].append(feat)

        attractors = {}
        for spk in enroll_emb_dict:
            mean_emb = torch.stack(enroll_emb_dict[spk]).mean(dim=0)
            mean_emb = F.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            attractors[spk] = mean_emb

        print(f"Computed attractors for {len(attractors)} speakers")
        print("=" * 60)
        return attractors

    @torch.no_grad()
    def evaluate(self, dataloader, output_file='vctk_results.txt',
                 save_detailed=True, samo_attractors=None):
        """
        运行评估，计算 EER 和 Accuracy。

        评分逻辑:
          - SAMO + attractors → 1-on-1 余弦相似度
          - SAMO + loss_model → maxscore
          - 其他 → softmax 概率
        """
        all_scores = []
        all_labels = []
        bonafide_scores = []
        spoof_scores = []

        with open(output_file, 'w') as f_out:
            header = "filename\tspeaker_id\ttrue_label\tpredicted_label\tscore\tfile_path"
            if self.use_samo and samo_attractors is not None:
                header += "\tmode=with_enroll"
            f_out.write(header + "\n")

            for audios, filenames, labels, speaker_ids, filepaths in tqdm(dataloader, desc="Testing"):
                audios = audios.to(self.device)
                features, outputs = self.model(audios)

                # ---- 评分逻辑 ----
                if self.use_samo:
                    if samo_attractors is not None:
                        # 1-on-1 相似度
                        batch_enrolls = []
                        for spk in speaker_ids:
                            if spk in samo_attractors:
                                batch_enrolls.append(samo_attractors[spk])
                            else:
                                batch_enrolls.append(torch.zeros(self.enc_dim))
                        batch_enrolls = torch.stack(batch_enrolls).to(self.device)
                        batch_enrolls = F.normalize(batch_enrolls, p=2, dim=1)
                        final_scores = torch.sum(features * batch_enrolls, dim=1).cpu().numpy()
                    else:
                        # maxscore
                        if self.loss_model and hasattr(self.loss_model, 'center'):
                            w = F.normalize(self.loss_model.center, p=2, dim=1).to(self.device)
                            scores = features @ w.transpose(0, 1)
                            final_scores, _ = torch.max(scores, dim=1)
                            final_scores = final_scores.cpu().numpy()
                        else:
                            probs = F.softmax(outputs, dim=1)
                            final_scores = probs[:, 0].cpu().numpy()
                else:
                    probs = F.softmax(outputs, dim=1)
                    final_scores = probs[:, 0].cpu().numpy()

                # 记录结果
                for i in range(len(filenames)):
                    score = final_scores[i]
                    true_label = labels[i].item()
                    pred_label = 0 if score > 0.5 else 1

                    f_out.write(f"{filenames[i]}\t{speaker_ids[i]}\t{true_label}\t"
                                f"{pred_label}\t{score:.6f}\t{filepaths[i]}\n")

                    all_scores.append(score)
                    all_labels.append(true_label)

                    if true_label == 0:
                        bonafide_scores.append(score)
                    else:
                        spoof_scores.append(score)

        # ---- 计算指标 ----
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        bona_scores = all_scores[all_labels == 0]
        spoof_scores_arr = all_scores[all_labels == 1]

        eer, threshold = em.compute_eer(bona_scores, spoof_scores_arr)
        predictions = np.where(all_scores > threshold, 0, 1)
        accuracy = np.mean(predictions == all_labels)

        print(f"\n{'=' * 60}")
        print(f"VCTK测试完成！结果保存至: {output_file}")
        print(f"{'=' * 60}")
        if self.use_samo and samo_attractors is not None:
            print(f"SAMO Mode: With Enrollment")
        elif self.use_samo:
            print(f"SAMO Mode: Without Enrollment")

        print(f"\n性能指标:")
        print(f"  EER: {eer * 100:.2f}%")
        print(f"  EER Threshold: {threshold:.4f}")
        print(f"  Accuracy (EER阈值): {accuracy * 100:.2f}%")
        print(f"  Bonafide: {np.mean(bona_scores):.4f} ± {np.std(bona_scores):.4f}")
        print(f"  Spoof: {np.mean(spoof_scores_arr):.4f} ± {np.std(spoof_scores_arr):.4f}")
        print(f"{'=' * 60}")

        return {
            'eer': eer * 100,
            'threshold': threshold,
            'accuracy': accuracy * 100,
            'bonafide_scores': bona_scores,
            'spoof_scores': spoof_scores_arr
        }


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    """命令行入口，与 test_vctk_samo.py 类似的参数接口"""
    parser = argparse.ArgumentParser(
        description='在 VCTK 数据集上测试 SAMO/AASIST 模型（原始音频）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_vctk.py -m ./models/samo.pt --loss_type samo --samo_mode with_enroll \\
      --vctk_clean ./VCTK-Corpus --vctk_attack ./VCTK-Corpus-attack
        """
    )

    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--loss_type', type=str, default='samo',
                        choices=['samo', 'ocsoftmax'], help='模型类型')
    parser.add_argument('--samo_mode', type=str, default='with_enroll',
                        choices=['with_enroll', 'no_enroll'], help='SAMO 测试模式')

    parser.add_argument('--vctk_clean', type=str, default='./VCTK-Corpus')
    parser.add_argument('--vctk_attack', type=str, default='./VCTK-Corpus-attack')
    parser.add_argument('--no_clean', action='store_true')
    parser.add_argument('--no_attack', action='store_true')

    parser.add_argument('--max_len', type=int, default=64600)
    parser.add_argument('--enc_dim', type=int, default=160)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-o', '--output', type=str, default='vctk_results.txt')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    if args.no_clean and args.no_attack:
        parser.error("不能同时跳过 clean 和 attack 数据！")
    if args.no_clean and args.samo_mode == 'with_enroll':
        parser.error("with_enroll 模式需要 clean 音频！")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = VCTKRawDataset(
        vctk_clean_dir=args.vctk_clean if not args.no_clean else None,
        vctk_attack_dir=args.vctk_attack if not args.no_attack else None,
        max_len=args.max_len, sr=args.sr,
        load_clean=not args.no_clean,
        load_attack=not args.no_attack
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate_fn)

    tester = VCTKTester(model_path=args.model_path, add_loss=args.loss_type,
                        enc_dim=args.enc_dim)

    samo_attractors = None
    if args.loss_type == 'samo' and args.samo_mode == 'with_enroll':
        clean_dataset = VCTKRawDataset(
            vctk_clean_dir=args.vctk_clean, vctk_attack_dir=None,
            max_len=args.max_len, sr=args.sr,
            load_clean=True, load_attack=False
        )
        clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=clean_dataset.collate_fn)
        samo_attractors = tester.compute_speaker_attractors(clean_loader)

    results = tester.evaluate(dataloader, output_file=args.output,
                              samo_attractors=samo_attractors)
    print(f"\n详细结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
