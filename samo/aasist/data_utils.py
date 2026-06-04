"""
数据处理与加载模块
================
本模块负责解析 ASVspoof 2019 LA 数据集的协议文件，并创建 PyTorch Dataset。

主要功能:
  1. genSpoof_list(): 解析协议文件，提取文件名、标签、说话人等元信息
  2. ASVspoof2019_speaker_raw: 自定义 Dataset 类，加载原始音频
  3. pad() / pad_random(): 音频长度对齐工具

协议文件格式:
  训练文件 (.trn.txt):       speaker_id  filename  system_id  attack_tag  label
  试听文件 (.trl.txt):       speaker_id  filename  system_id  attack_tag  label
  注册文件 (asv.*.trn.txt): speaker_id  filename1,filename2,...

音频格式: FLAC, 16kHz 采样率, 单声道

改编自 https://github.com/clovaai/aasist
原始作者: Hemlata Tak, Jee-weon Jung
"""

import numpy as np
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset


def genSpoof_list(dir_meta, enroll=False, train=True, target_only=False, enroll_spk=None):
    """
    解析 ASVspoof 2019 LA 协议文件。

    协议文件定义了训练/验证/测试集中每条语音的元信息:
      - 说话人 ID (speaker)
      - 音频文件名 (key)
      - 攻击类型标签 (tag, 如 A07, A08, ...)
      - 标签 (bonafide 或 spoof)

    参数:
        dir_meta:    协议文件路径（或注册文件的路径列表）
        enroll:      是否解析注册协议（注册协议格式不同——一个说话人对应多个文件）
        train:       是否为训练模式（影响标签存储方式）
        target_only: 是否仅加载目标说话人的数据
        enroll_spk:  目标说话人 ID 集合（与 target_only 配合使用）

    返回:
        d_meta:   {文件名: 标签} 字典（0=bonafide 真实, 1=spoof 伪造）
        utt_list: 音频文件名列表
        utt2spk:  {文件名: 说话人ID} 字典
        tag_list: 攻击类型标识列表（bonafide 记作 "-"）
    """
    d_meta = {}
    utt_list = []
    tag_list = []
    utt2spk = {}

    if not enroll and train:
        # ========== 训练协议文件 ==========
        # 格式: speaker_id filename system_id attack_tag label
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            spk, key, _, tag, label = line.strip().split(" ")

            if key in utt2spk:
                print("Duplicated utt error", key)

            utt2spk[key] = spk
            tag_list.append(tag)
            utt_list.append(key)
            # d_meta: 1=spoof(伪造), 0=bonafide(真实) —— 注意这里与直觉相反
            d_meta[key] = 1 if label != "bonafide" else 0

    elif not enroll and not train:
        # ========== 验证/测试试听协议文件 ==========
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            spk, key, _, tag, label = line.strip().split(" ")

            # target_only 模式: 只加载 enroll_spk 中指定的说话人的数据
            if not target_only or spk in enroll_spk:
                if key in utt2spk:
                    print("Duplicated utt error", key)

                utt2spk[key] = spk
                utt_list.append(key)
                tag_list.append(tag)
                d_meta[key] = 1 if label != "bonafide" else 0

    else:
        # ========== 注册协议文件 ==========
        # 注册文件按性别分开（female.trn.txt / male.trn.txt）
        # 格式: speaker_id filename1,filename2,filename3,...
        for dir in dir_meta:
            with open(dir, "r") as f:
                l_meta = f.readlines()

            for line in l_meta:
                tmp = line.strip().split(" ")
                spk = tmp[0]
                keys = tmp[1].split(",")  # 一个说话人可能有多条注册语音

                for key in keys:
                    if key in utt2spk:
                        print("Duplicated utt error", key)

                    utt2spk[key] = spk
                    utt_list.append(key)
                    d_meta[key] = 0       # 注册语音都是 bonafide
                    tag_list.append("-")   # 注册语音没有攻击类型

    return d_meta, utt_list, utt2spk, tag_list


def pad(x, max_len=64600):
    """
    固定长度填充（用于验证和测试）。

    如果音频比目标长度短，重复拼接至目标长度；
    如果比目标长度长，截取前 max_len 个采样点。

    参数:
        x:       一维音频 numpy 数组
        max_len: 目标长度（采样点数），默认 64600 ≈ 4秒 @16kHz

    返回:
        padded_x: 形状为 (max_len,) 的数组
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]  # 截取前 max_len 个点

    # 重复拼接至足够长度
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    """
    随机裁剪（用于训练时数据增强）。

    如果音频比目标长度长，随机选择一个起始点进行截取；
    如果比目标长度短，重复拼接至目标长度。

    参数:
        x:       一维音频 numpy 数组
        max_len: 目标长度（采样点数）

    返回:
        裁剪/填充后的音频数组，形状为 (max_len,)
    """
    x_len = x.shape[0]

    # 音频足够长 → 随机截取
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # 音频太短 → 重复填充
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class ASVspoof2019_speaker_raw(Dataset):
    """
    ASVspoof 2019 LA 数据集类（原始音频加载）。

    继承自 PyTorch Dataset，提供:
      - 从 FLAC 文件加载原始音频
      - 自动裁剪/填充到固定长度
      - 返回说话人信息（用于 SAMO 的说话人感知训练）

    输入:
        list_IDs:  音频文件名列表
        labels:    {文件名: 标签} 字典
        utt2spk:   {文件名: 说话人ID} 字典
        base_dir:  音频文件所在目录
        tag_list:  攻击类型标识列表
        train:     是否为训练模式（影响音频裁剪方式）
        cut:       目标音频长度（采样点数），默认 64600 ≈ 4秒 @16kHz
    """

    def __init__(self, list_IDs, labels, utt2spk, base_dir, tag_list, train=True, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.utt2spk = utt2spk
        self.tag_list = tag_list
        self.cut = cut
        self.train = train

    def __len__(self):
        """返回数据集大小"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        获取单条样本。

        处理流程:
          1. 用 soundfile 读取 FLAC 格式的 16kHz 原始音频
          2. 裁剪/填充到固定长度（训练时随机截取，测试时固定截取）
          3. 包装为 PyTorch Tensor

        返回:
            x_inp: 音频 tensor (cut,)
            y:     标签 (0=bonafide, 1=spoof)
            spk:   说话人 ID (如 LA_0079)
            key:   音频文件名
            tag:   攻击类型标识 (如 A07, 或 "-")
        """
        key = self.list_IDs[index]
        tag = self.tag_list[index]

        # 使用 soundfile 读取原始音频
        X, _ = sf.read(str(self.base_dir + f"flac/{key}.flac"))

        # 根据模式选择裁剪方式
        if self.train:
            X_pad = pad_random(X, self.cut)  # 训练: 随机截取（数据增强）
        else:
            X_pad = pad(X, self.cut)          # 测试: 固定截取（保证确定性）

        x_inp = Tensor(X_pad)
        y = self.labels[key]
        spk = self.utt2spk[key]

        return x_inp, y, spk, key, tag
