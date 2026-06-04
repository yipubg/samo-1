"""
学习率搜索工具
==============
使用 PyTorch LR Finder 自动寻找最佳学习率。

原理:
  LR Finder 从一个很小的学习率开始训练，逐步增大学习率，
  记录每个学习率对应的损失值。
  然后绘制"学习率 vs 损失"曲线，选择损失下降最快的区间的学习率。

使用方法:
  python search_lr.py

这有助于在正式训练前确定最优的初始学习率，避免:
  - 学习率太小 → 收敛太慢
  - 学习率太大 → 训练不稳定或发散

依赖: torch_lr_finder 库 (pip install torch_lr_finder)
"""

import json
import os

import torch
import torch.nn as nn
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from loss import SAMO
from main import init_params, get_model, get_loader, update_embeds


class MyTrainDataLoaderIter(TrainDataLoaderIter):
    """
    自定义的 DataLoader 迭代器包装器。

    由于 SAMO 的 DataLoader 返回 5 个元素 (feat, labels, spk, utt, tag)，
    而 LR Finder 默认只支持 (inputs, labels) 格式，
    因此需要重新定义如何从批次数据中提取输入和标签。

    参考: https://github.com/davidtvs/pytorch-lr-finder
    """

    def inputs_labels_from_batch(self, batch_data):
        """
        从批次数据中提取输入和标签。

        参数:
            batch_data: DataLoader 返回的 5 元组

        返回:
            inputs: 完整的批次数据（模型需要所有 5 个元素）
            labels: 标签张量
        """
        _, labels, _, _, _ = batch_data
        inputs = batch_data
        return inputs, labels


class ModelWrapper(nn.Module):
    """
    模型包装器 —— 解包批次数据中的多个元素。

    因为 SAMO 的 loss 函数需要 feat、spk 等多个输入，
    而 LR Finder 只支持标准的 (inputs) → (outputs) 格式，
    所以需要包装模型来适配接口。
    """

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, inputs):
        """
        解包输入，调用模型，返回中间结果。

        输入: (feat, labels, spk, utt, tag)
        输出: (feats, spk) —— 用于 loss 计算的中间表示
        """
        feat, labels, spk, utt, tag = inputs
        feats, fc_outputs = self.model(feat)
        return (feats, spk)


class LossFunctionWrapper(nn.Module):
    """
    损失函数包装器 —— 解包模型输出的多个元素。

    因为 SAMO 的 loss 函数需要 feats 和 spk 两个输入，
    而 LR Finder 只支持标准的 (outputs, labels) → loss 格式。
    """

    def __init__(self, loss_func, centers):
        super(LossFunctionWrapper, self).__init__()
        self.loss_func = loss_func
        self.centers = centers

    def forward(self, inputs, labels):
        """
        解包模型输出，计算 SAMO 损失。

        参数:
            inputs: (feats, spk) —— ModelWrapper 的输出
            labels: 标签张量

        返回:
            loss: 标量损失值
        """
        feats, spk = inputs
        loss, _ = self.loss_func(feats, labels, spk=spk, enroll=self.centers, spoofprint=1)
        return loss


if __name__ == '__main__':
    # ---- 环境设置 ----
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cuda = torch.cuda.is_available()
    print('Cuda device available: ', cuda)
    device = torch.device("cuda" if cuda else "cpu")

    # ---- 加载模型和数据 ----
    with open("aasist/AASIST.conf", "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    feat_model = get_model(model_config)

    nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
    print("no. model params:{}".format(nb_params))

    args = init_params()
    train_data_loader, dev_data_loader, eval_data_loader, \
    train_bona_loader, dev_enroll_loader, eval_enroll_loader, num_centers = get_loader(args)

    # ---- 创建优化器 ----
    optimizer = torch.optim.Adam(feat_model.parameters(),
                                 lr=0.0001,
                                 betas=optim_config['betas'],
                                 weight_decay=optim_config['weight_decay'],
                                 amsgrad=optim_config['amsgrad'])

    # ---- 创建 SAMO 损失和说话人中心 ----
    samo = SAMO(160, m_real=0.7, m_fake=0, alpha=20).to(device)
    centers = update_embeds(device, feat_model, train_bona_loader)

    # ---- 创建包装器（适配 LR Finder 接口）----
    trainloader_wrapper = MyTrainDataLoaderIter(train_data_loader)
    model_wrapper = ModelWrapper(feat_model)
    loss_func_wrapper = LossFunctionWrapper(samo, centers)

    # ---- 运行 LR Finder ----
    lr_finder = LRFinder(model_wrapper, optimizer, loss_func_wrapper, device='cuda')
    lr_finder.range_test(
        trainloader_wrapper,
        # 学习率从 1e-7 指数增长到 0.1
        end_lr=0.1, num_iter=1103, step_mode='exp', start_lr=1e-7
    )

    # 绘制学习率-损失曲线，并建议最佳学习率
    lr_finder.plot(suggest_lr=True)

    # 恢复模型和优化器到测试前的状态
    lr_finder.reset()
