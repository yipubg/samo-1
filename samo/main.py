"""
SAMO 训练与测试主入口
====================
本文件是项目的顶层入口，包含完整的训练和测试流程。

功能:
  1. init_params():    命令行参数解析与运行环境初始化
  2. get_loader():     创建所有数据加载器（训练/验证/评估 + 注册集）
  3. get_model():      加载 AASIST 骨干网络
  4. get_optimizer():  创建优化器和学习率调度器
  5. train():          完整的训练循环（含验证、测试、保存）
  6. update_embeds():  更新说话人中心向量（attractor）
  7. test():           独立的模型测试/评估函数

训练流程:
  初始化 → 加载数据 → 创建模型 → 创建优化器 → 创建损失函数 →
  循环每个 epoch:
    → 更新 SAMO 中心 → 训练迭代 → 验证评估 → 测试评估 → 保存模型

支持的损失函数:
  - softmax:   标准交叉熵（二分类）
  - ocsoftmax: 单中心单类学习（基线方法）
  - samo:      多中心说话人感知单类学习（论文方法）

引用:
  - AASIST 数据加载部分改编自 https://github.com/clovaai/aasist
  - OC-Softmax 部分改编自 https://github.com/yzyouzhang/AIR-ASVspoof
"""

import argparse
import json
import logging
import os.path
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
from torchcontrib.optim import SWA
from tqdm import tqdm

from aasist import AASIST
from aasist.data_utils import genSpoof_list, ASVspoof2019_speaker_raw
from loss import SAMO, OCSoftmax
from utils import setup_seed, seed_worker, cosine_annealing, adjust_learning_rate, em, compute_eer_tdcf


def init_params():
    """
    解析命令行参数并初始化运行环境。

    包含以下类别的参数:
      - 数据路径:  数据集位置、协议文件、输出目录
      - 模型参数:  特征维度
      - 训练参数:  学习率、批次大小、训练轮数、优化器设置
      - 损失参数:  损失函数类型、中心数量、margin 值
      - 测试参数:  测试模式、评分方式、保存选项
      - SAMO 参数: 说话人感知模式、中心更新策略

    返回:
        args: 包含所有配置的命名空间对象
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- 基础设置 ----
    parser.add_argument('--seed', type=int, help="随机种子（保证实验可复现）", default=10)

    # ---- 数据路径 ----
    parser.add_argument("-d", "--path_to_database", type=str, help="ASVspoof 2019 LA 数据集根目录",
                        default='/data2/sivan/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="协议文件目录",
                        default='../protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="模型和日志输出目录",
                        required=False, default='./models/try/')
    parser.add_argument("--overwrite", action='store_true', help="是否覆盖已有的输出目录")

    # ---- 模型参数 ----
    parser.add_argument("--enc_dim", type=int, help="特征嵌入向量的维度（AASIST 输出维度）", default=160)

    # ---- 训练超参数 ----
    parser.add_argument('--num_epochs', type=int, default=100, help="训练总轮数")
    parser.add_argument('--batch_size', type=int, default=23, help="每批次的样本数（受 11GB 显存限制）")
    parser.add_argument('--lr', type=float, default=0.0001, help="初始学习率")
    parser.add_argument('--lr_min', type=float, default=0.000005, help="余弦退火的最小学习率")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="指数衰减的衰减率")
    parser.add_argument('--interval', type=int, default=1, help="指数衰减的间隔（epoch）")
    parser.add_argument("--scheduler", type=str, default="cosine2",
                        choices=["cosine", "cosine2", "exp", "clr"],
                        help="学习率调度策略: cosine2=双周期余弦退火")
    parser.add_argument('--beta_1', type=float, default=0.9, help="Adam 优化器的 beta_1")
    parser.add_argument('--beta_2', type=float, default=0.999, help="Adam 优化器的 beta_2")
    parser.add_argument('--eps', type=float, default=1e-8, help="Adam 优化器的 epsilon")
    parser.add_argument("--gpu", type=str, help="使用的 GPU 编号", default="0")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader 的工作进程数")
    parser.add_argument('--dp', action='store_true', default=False, help='是否使用 DataParallel 多 GPU 训练')

    # ---- 损失函数设置 ----
    parser.add_argument('-l', '--loss', type=str, default="samo",
                        choices=["softmax", "ocsoftmax", "samo"],
                        help="损失函数选择: softmax=交叉熵, ocsoftmax=单中心基线, samo=论文方法")
    parser.add_argument('--num_centers', type=int, default=20,
                        help="SAMO 中心向量的数量（对应训练集说话人数）")
    parser.add_argument('--initialize_centers', type=str, default="one_hot",
                        choices=["randomly", "evenly", "one_hot", "uniform"],
                        help="中心向量的初始化方式: one_hot=单位矩阵行, evenly=超球面均匀采样")
    parser.add_argument('--m_real', type=float, default=0.7,
                        help="真实语音的 margin — bonafide 与中心的相似度需高于此值")
    parser.add_argument('--m_fake', type=float, default=0,
                        help="伪造语音的 margin — spoof 与中心的相似度需低于此值")
    parser.add_argument('--alpha', type=float, default=20,
                        help="损失函数的缩放因子（温度），越大惩罚越'硬'")

    # ---- 训练控制 ----
    parser.add_argument('--continue_training', action='store_true', help="从已有的检查点继续训练")
    parser.add_argument('--checkpoint', type=int, help="继续训练时，从哪个 epoch 的检查点恢复")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="是否在评估集上定期测试")
    parser.add_argument('--final_test', action='store_true',
                        help="训练结束后是否用最佳模型在测试集上评估")
    parser.add_argument('--test_interval', type=int, default=5,
                        help="每隔多少个 epoch 在验证/测试集上评估一次")
    parser.add_argument('--save_interval', type=int, default=5,
                        help="每隔多少个 epoch 保存一次模型检查点")

    # ---- 测试设置 ----
    parser.add_argument('--test_only', action='store_true', help='仅测试模式（不训练，只评估已有模型）')
    parser.add_argument("--test_model", type=str, default="./models/anti-spoofing_feat_model.pt",
                        help="测试时加载的模型路径")
    parser.add_argument("--scoring", type=str, default=None,
                        choices=["fc", "samo", "ocsoftmax"],
                        help="测试时的评分方式: fc=全连接层, samo=SAMO, ocsoftmax=单中心")
    parser.add_argument("--save_score", type=str, default=None,
                        help='保存每个样本详细分数的文件名')
    parser.add_argument('--save_center', action='store_true', help='是否将中心向量保存到日志文件')
    parser.add_argument('--one_hot', action='store_true', help='最终测试时是否使用 one-hot 中心向量')

    # ---- SAMO 特有设置 ----
    parser.add_argument('--train_sp', type=int, default=1,
                        help="训练时的说话人感知模式: 1=说话人感知(SIM), 2=说话人无关(MAXSCORE), 0=其他")
    parser.add_argument('--val_sp', type=int, default=1,
                        help="验证/测试时的说话人感知模式: 1=说话人感知, 2=说话人无关, 0=说话人独立")
    parser.add_argument('--target', type=int, default=1,
                        help='验证和评估时是否只加载目标说话人的数据')
    parser.add_argument('--update_interval', type=int, default=3,
                        help="每隔多少个 epoch 更新一次训练中心向量")
    parser.add_argument('--init_center', type=int, default=1,
                        help='是否使用 one-hot 正交嵌入初始化中心（1=是）')
    parser.add_argument('--update_stop', type=int, default=None,
                        help='训练多少个 epoch 后停止更新中心向量')
    parser.add_argument("--center_sampler", type=str, default="sequential",
                        choices=["sequential", "random"],
                        help="中心采样方式: sequential=顺序, random=随机")

    args = parser.parse_args()

    # ---- 设置 GPU ----
    if not args.dp:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 默认：整个训练过程都更新中心
    if args.update_stop is None:
        args.update_stop = args.num_epochs + 1

    # ---- 固定随机种子 ----
    setup_seed(args.seed)

    # ---- 创建输出目录 ----
    if args.continue_training:
        pass  # 继续训练时保留已有目录
    else:
        if os.path.exists(args.out_fold):
            logging.warning('{} 已存在'.format(args.out_fold))
            print("overwrite:{}".format(args.overwrite))

        # 试用目录默认覆盖
        if args.out_fold == './models/try/':
            args.overwrite = True

        os.makedirs(args.out_fold, exist_ok=args.overwrite)
        if args.overwrite:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # 检查点目录
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # 验证数据路径存在
        if not os.path.exists(args.path_to_database):
            raise RuntimeError(f'路径 {args.path_to_database} 不存在！')

        # ---- 保存训练参数 ----
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))
        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    # ---- 检测 CUDA 可用性 ----
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def get_loader(args):
    """
    创建所有数据加载器（DataLoader）。

    ASVspoof 2019 LA 数据集分为三个子集，每个子集包含：
      - 普通数据（bonafide + spoof 混合）: 用于训练/验证/测试
      - 注册数据（enroll, 仅 bonafide）: 用于提取说话人中心向量

    返回 6 个 DataLoader:
      1. trn_loader:       训练数据（随机打乱）
      2. dev_loader:       验证数据
      3. eval_loader:      评估数据
      4. trn_bona:         仅训练集真实语音（用于更新训练中心）
      5. dev_enroll:       验证集注册数据（用于提取验证中心）
      6. eval_enroll:      评估集注册数据（用于提取评估中心）

    以及:
      num_centers: [训练集说话人数, 验证集说话人数, 评估集说话人数]

    改编自 https://github.com/clovaai/aasist
    """
    database_path = args.path_to_database
    seed = args.seed
    target_only = args.target
    batch_size = args.batch_size

    # ---- 构建各子集的路径 ----
    trn_database_path = os.path.join(database_path + "LA/ASVspoof2019_LA_train/")
    dev_database_path = os.path.join(database_path + "LA/ASVspoof2019_LA_dev/")
    eval_database_path = os.path.join(database_path + "LA/ASVspoof2019_LA_eval/")

    trn_list_path = "protocols/ASVspoof2019.LA.cm.train.trn.txt"
    dev_trial_path = "protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    eval_trial_path = "protocols/ASVspoof2019.LA.cm.eval.trl.txt"

    # 注册协议文件（按性别分开）
    dev_enroll_path = ["protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
                       "protocols/ASVspoof2019.LA.asv.dev.male.trn.txt"]
    eval_enroll_path = ["protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
                        "protocols/ASVspoof2019.LA.asv.eval.male.trn.txt"]

    # ==================== 训练数据 ====================
    label_trn, file_train, utt2spk_train, tag_train = genSpoof_list(
        dir_meta=trn_list_path, enroll=False, train=True)
    trn_centers = len(set(utt2spk_train.values()))  # 统计说话人数
    print("no. training files:", len(file_train))
    print("no. training speakers:", trn_centers)

    # 训练集（含 bonafide 和 spoof）
    train_set = ASVspoof2019_speaker_raw(list_IDs=file_train,
                                         labels=label_trn,
                                         utt2spk=utt2spk_train,
                                         base_dir=trn_database_path,
                                         tag_list=tag_train,
                                         train=True)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            drop_last=True, pin_memory=True,
                            worker_init_fn=seed_worker, generator=gen)

    # 仅真实语音的训练集（用于更新说话人中心）
    # 训练集中前 2580 条为 bonafide 样本
    num_bonafide_train = 2580
    train_set_fix = ASVspoof2019_speaker_raw(list_IDs=file_train,
                                             labels=label_trn,
                                             utt2spk=utt2spk_train,
                                             base_dir=trn_database_path,
                                             tag_list=tag_train,
                                             train=False)
    trn_bona_set = Subset(train_set_fix, range(num_bonafide_train))

    if args.center_sampler == "random":
        trn_bona = DataLoader(trn_bona_set, batch_size=batch_size,
                              shuffle=True, drop_last=False, pin_memory=True,
                              worker_init_fn=seed_worker, generator=gen)
    elif args.center_sampler == "sequential":
        trn_bona = DataLoader(trn_bona_set, batch_size=int(args.batch_size),
                              shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError

    # ==================== 验证数据 ====================
    # 验证集注册数据（用于提取验证说话人中心）
    label_dev_enroll, file_dev_enroll, utt2spk_dev_enroll, tag_dev_enroll = genSpoof_list(
        dir_meta=dev_enroll_path, enroll=True, train=False)
    dev_enroll_spk = set(utt2spk_dev_enroll.values())
    dev_centers = len(dev_enroll_spk)
    print(f"no. validation enrollment files: {len(file_dev_enroll)}")
    print(f"no. validation enrollment speakers: {dev_centers}")

    dev_set_enroll = ASVspoof2019_speaker_raw(list_IDs=file_dev_enroll,
                                              labels=label_dev_enroll,
                                              utt2spk=utt2spk_dev_enroll,
                                              base_dir=dev_database_path,
                                              tag_list=tag_dev_enroll,
                                              train=False)
    dev_enroll = DataLoader(dev_set_enroll, batch_size=args.batch_size,
                            shuffle=False, drop_last=False, pin_memory=True)

    # 验证集（仅目标说话人）
    label_dev, file_dev, utt2spk_dev, tag_dev = genSpoof_list(
        dir_meta=dev_trial_path, enroll=False, train=False,
        target_only=target_only, enroll_spk=dev_enroll_spk)
    print(f"no. validation files: {len(file_dev)}")

    dev_set = ASVspoof2019_speaker_raw(list_IDs=file_dev,
                                       labels=label_dev,
                                       utt2spk=utt2spk_dev,
                                       base_dir=dev_database_path,
                                       tag_list=tag_dev,
                                       train=False)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                            shuffle=False, drop_last=False, pin_memory=True)

    # ==================== 评估数据 ====================
    # 评估集注册数据
    label_eval_enroll, file_eval_enroll, utt2spk_eval_enroll, tag_eval_enroll = genSpoof_list(
        dir_meta=eval_enroll_path, enroll=True, train=False)
    eval_enroll_spk = set(utt2spk_eval_enroll.values())
    eval_centers = len(eval_enroll_spk)
    print(f"no. eval enrollment files: {len(file_eval_enroll)}")
    print(f"no. eval enrollment speakers: {eval_centers}")

    eval_set_enroll = ASVspoof2019_speaker_raw(list_IDs=file_eval_enroll,
                                               labels=label_eval_enroll,
                                               utt2spk=utt2spk_eval_enroll,
                                               base_dir=eval_database_path,
                                               tag_list=tag_eval_enroll,
                                               train=False)
    eval_enroll = DataLoader(eval_set_enroll, batch_size=batch_size,
                             shuffle=False, drop_last=False, pin_memory=True)

    # 评估集（仅目标说话人）
    label_eval, file_eval, utt2spk_eval, tag_eval = genSpoof_list(
        dir_meta=eval_trial_path, enroll=False, train=False,
        target_only=target_only, enroll_spk=eval_enroll_spk)
    print(f"no. eval files: {len(file_eval)}")

    eval_set = ASVspoof2019_speaker_raw(list_IDs=file_eval,
                                        labels=label_eval,
                                        utt2spk=utt2spk_eval,
                                        base_dir=eval_database_path,
                                        tag_list=tag_eval,
                                        train=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size,
                             shuffle=False, drop_last=False, pin_memory=True)

    num_centers = [trn_centers, dev_centers, eval_centers]

    return trn_loader, dev_loader, eval_loader, trn_bona, dev_enroll, eval_enroll, num_centers


def get_model(model_config):
    """
    加载 AASIST 骨干网络模型。

    参数:
        model_config: 从 AASIST.conf 读取的模型配置字典

    返回:
        feat_model: AASIST 模型实例
    """
    _model = AASIST.Model
    feat_model = _model(model_config).to(args.device)
    nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
    print("no. model params:{}".format(nb_params))
    return feat_model


def get_optimizer(optim_config, feat_model, total_steps):
    """
    创建优化器和学习率调度器。

    优化器: Adam（with amsgrad）
    调度器: LambdaLR with Cosine Annealing

    参数:
        optim_config: 优化器配置字典
        feat_model:   要优化的模型
        total_steps:  总训练步数 = num_epochs × steps_per_epoch

    返回:
        optimizer:     Adam 优化器
        optimizer_swa: SWA（随机权重平均）包装器，用于模型集成
        scheduler:     Cosine Annealing 学习率调度器
    """
    optim_config['base_lr'] = args.lr
    optim_config['lr_min'] = args.lr_min

    optimizer = torch.optim.Adam(feat_model.parameters(),
                                 lr=optim_config['base_lr'],
                                 betas=optim_config['betas'],
                                 weight_decay=optim_config['weight_decay'],
                                 amsgrad=optim_config['amsgrad'])

    # Cosine Annealing 学习率调度
    if "cosine" in args.scheduler:
        if args.scheduler == "cosine2":
            total_steps = 2 * total_steps  # 双周期 → 更平缓的衰减
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step, total_steps, 1,  # lr_lambda 计算的是乘法因子
                optim_config['lr_min'] / optim_config['base_lr']))
    else:
        raise NotImplementedError(f"未知的调度器: {args.scheduler}")

    # SWA: 对训练后期的模型参数取平均，通常能提升泛化性能
    optimizer_swa = SWA(optimizer)

    return optimizer, optimizer_swa, scheduler


def train(args):
    """
    完整的训练流程。

    流程概览:
      初始化模型 → 加载数据 → 创建优化器 → 创建损失函数 →
      循环每个 epoch:
        1. 初始化/更新 SAMO 说话人中心向量
        2. 遍历训练数据：前向 → 损失 → 反向传播 → 更新参数
        3. 在验证集上评估（计算 EER）
        4. 在评估集上测试（可选，由 --test_on_eval 控制）
        5. 保存模型检查点
        6. 跟踪最佳模型（基于验证损失）
    """
    final_test = args.final_test
    torch.set_default_tensor_type(torch.FloatTensor)

    # ---- 步骤1: 加载 AASIST 骨干网络 ----
    with open("samo/aasist/AASIST.conf", "r") as f_json:
        config = json.loads(f_json.read())
    optim_config = config["optim_config"]
    feat_model = get_model(model_config=config["model_config"])

    # ---- 多 GPU 设置 ----
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and args.dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        feat_model = nn.DataParallel(feat_model)
        feat_model.to(args.device)

    # ---- 步骤2: 加载所有数据 ----
    train_data_loader, dev_data_loader, eval_data_loader, \
    train_bona_loader, dev_enroll_loader, eval_enroll_loader, num_centers = get_loader(args)

    # ---- 继续训练：从检查点恢复 ----
    if args.continue_training:
        feat_model = torch.load(
            os.path.join(args.out_fold, 'checkpoint',
                         'anti-spoofing_feat_model_%d.pt' % args.checkpoint)).to(args.device)
        loss_model = torch.load(
            os.path.join(args.out_fold, 'checkpoint',
                         'anti-spoofing_loss_model_%d.pt' % args.checkpoint)).to(args.device)

    # ---- 步骤3: 创建优化器和调度器 ----
    optimizer, optimizer_swa, scheduler = get_optimizer(
        optim_config, feat_model,
        total_steps=args.num_epochs * len(train_data_loader))

    # ---- 步骤4: 创建损失函数 ----
    # 交叉熵（用于 Softmax 模式的二分类）
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 0.1])).to(args.device)

    monitor_loss = args.loss
    if monitor_loss == "ocsoftmax":
        # OC-Softmax: 单中心单类学习
        ocsoftmax = OCSoftmax(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake,
                              alpha=args.alpha,
                              initialize_centers=args.initialize_centers).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimizer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)
    elif monitor_loss == "samo":
        # SAMO: 多中心说话人感知单类学习（论文创新方法）
        samo = SAMO(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha,
                    num_centers=args.num_centers,
                    initialize_centers=args.initialize_centers).to(args.device)
        ocsoftmax_optimizer = None
    else:
        raise NotImplementedError(f"损失函数 {monitor_loss} 尚未实现。")

    # ---- 早停 (Early Stopping) 设置 ----
    early_stop_cnt = 0
    prev_loss = 1e8
    best_epoch = 0

    # SWA 更新计数器
    n_swa_update = 0

    # ========================================================================
    # 主训练循环
    # ========================================================================
    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()

        # 用于记录 epoch 内的中间结果
        ip1_loader, idx_loader, spk_loader, utt_loader = [], [], [], []
        trainloss_dict = defaultdict(list)
        devloss_dict = defaultdict(list)
        testloss_dict = defaultdict(list)

        # 指数衰减学习率（备用方案）
        if args.scheduler == "exp":
            adjust_learning_rate(args, args.lr, optimizer, epoch_num)
        if monitor_loss == "ocsoftmax":
            adjust_learning_rate(args, args.lr, ocsoftmax_optimizer, epoch_num)

        print('\nEpoch: %d ' % (epoch_num + 1))

        # ================================================================
        # 阶段A: 初始化/更新 SAMO 说话人中心向量 (Attractors)
        # ================================================================
        if args.train_sp and monitor_loss == 'samo':
            if epoch_num == 0 and args.init_center == 1:
                # 第 0 个 epoch: one-hot 初始化
                # 训练集说话人 ID 格式: LA_0079 ~ LA_0098（共 20 人）
                spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
                tmp_center = torch.eye(args.enc_dim)[:num_centers[0]]
                train_enroll = dict(zip(spklist, tmp_center))
            elif epoch_num % args.update_interval == 0 and epoch_num < args.update_stop:
                # 每 update_interval 个 epoch 更新一次中心
                # 用最新的模型参数重新提取 bona 数据的特征，取均值作为新中心
                train_enroll = update_embeds(args.device, feat_model, train_bona_loader)

            # 将中心向量传给 SAMO 损失函数
            samo.center = torch.stack(list(train_enroll.values()))

        # ================================================================
        # 阶段B: 训练迭代
        # ================================================================
        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(train_data_loader)):
            feat = feat.to(args.device)
            labels = labels.to(args.device)

            # ---- 前向传播 ----
            feats, feat_outputs = feat_model(feat)
            # feats:       (batch, 160) 的 embedding
            # feat_outputs: (batch, 2) 的分类输出

            # ---- 计算损失 ----
            if args.train_sp:
                # SAMO 说话人感知损失
                samoloss, _ = samo(feats, labels, spk=spk, enroll=train_enroll, attractor=args.train_sp)
                feat_loss = samoloss
            elif monitor_loss == "softmax":
                feat_loss = criterion(feat_outputs, labels)
            elif monitor_loss == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                feat_loss = ocsoftmaxloss
            else:
                raise NotImplementedError

            # ---- 反向传播 ----
            if monitor_loss == "ocsoftmax":
                ocsoftmax_optimizer.zero_grad()
                ocsoftmax_optimizer.step()
            trainloss_dict[monitor_loss].append(feat_loss.item())
            optimizer.zero_grad()
            feat_loss.backward()
            optimizer.step()

            # ---- 学习率调度（步级别）----
            if args.scheduler != "exp":
                scheduler.step()

            # ---- 记录 ----
            ip1_loader.append(feats)
            idx_loader.append((labels))

            # 写入训练日志
            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(trainloss_dict[monitor_loss][-1]) + "\n")

        # 可选：保存中心向量
        if args.save_center:
            with open(os.path.join(args.out_fold, "train_enroll.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(samo.center.detach().numpy()) + "\n")

        # ================================================================
        # 阶段C: 验证评估
        # ================================================================
        feat_model.eval()
        with torch.no_grad():
            ip1_loader, idx_loader, spk_loader, score_loader = [], [], [], []

            # SAMO 验证中心向量
            if args.val_sp:
                # 用验证集注册数据更新中心
                dev_enroll = update_embeds(args.device, feat_model, dev_enroll_loader)
                samo.center = torch.stack(list(dev_enroll.values()))

            for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(dev_data_loader)):
                feat = feat.to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.val_sp:
                    # SAMO 推理（使用注册中心）
                    if args.target:
                        samoloss, score = samo(feats, labels, spk, dev_enroll, args.val_sp)
                    else:
                        samoloss, score = samo.inference(feats, labels, spk, dev_enroll, args.val_sp)
                    devloss_dict[monitor_loss].append(samoloss.item())
                elif monitor_loss == "softmax":
                    feat_loss = criterion(feat_outputs, labels)
                    score = feat_outputs[:, 0]
                    devloss_dict[monitor_loss].append(feat_loss.item())
                elif monitor_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devloss_dict[monitor_loss].append(ocsoftmaxloss.item())
                elif monitor_loss == "samo":
                    # 无注册时使用训练中心
                    samoloss, score = samo(feats, labels)
                    devloss_dict[monitor_loss].append(samoloss.item())

                ip1_loader.append(feats)
                idx_loader.append(labels)
                score_loader.append(score)

            # 计算验证 EER
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" +
                          str(np.nanmean(devloss_dict[monitor_loss])) + "\t" +
                          str(eer) + "\n")
            print("Val EER: {}".format(eer))

        # ================================================================
        # 阶段D: 评估集测试（可选）
        # ================================================================
        feat_model.eval()
        if args.test_on_eval:
            if (epoch_num + 1) % args.test_interval == 0:
                with torch.no_grad():
                    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []

                    # SAMO 评估中心向量
                    if args.val_sp:
                        eval_enroll = update_embeds(args.device, feat_model, eval_enroll_loader)
                        samo.center = torch.stack(list(eval_enroll.values()))

                    for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(eval_data_loader)):
                        if args.feat == "Raw":
                            feat = feat.to(args.device)
                        else:
                            feat = feat.transpose(2, 3).to(args.device)

                        labels = labels.to(args.device)
                        feats, feat_outputs = feat_model(feat)

                        if args.val_sp:
                            if args.target:
                                samoloss, score = samo(feats, labels, spk, eval_enroll, args.val_sp)
                            else:
                                samoloss, score = samo.inference(feats, labels, spk, eval_enroll, args.val_sp)
                            testloss_dict[monitor_loss].append(samoloss.item())
                        elif monitor_loss == "softmax":
                            feat_loss = criterion(feat_outputs, labels)
                            score = feat_outputs[:, 0]
                            testloss_dict[monitor_loss].append(feat_loss.item())
                        elif monitor_loss == "ocsoftmax":
                            ocsoftmaxloss, score = ocsoftmax(feats, labels)
                            testloss_dict[monitor_loss].append(ocsoftmaxloss.item())
                        elif monitor_loss == "samo" or monitor_loss == "samo_spoofall":
                            samoloss, score = samo(feats, labels)
                            testloss_dict[monitor_loss].append(samoloss.item())

                        ip1_loader.append(feats)
                        idx_loader.append((labels))
                        score_loader.append(score)

                    scores = torch.cat(score_loader, 0).data.cpu().numpy()
                    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

                    with open(os.path.join(args.out_fold, "test_loss.log"), "a") as log:
                        log.write(str(epoch_num) + "\t" +
                                  str(np.nanmean(testloss_dict[monitor_loss])) + "\t" + str(eer) + "\n")
                    print("Test EER: {}".format(eer))

        # ================================================================
        # 阶段E: 保存模型检查点
        # ================================================================
        if (epoch_num + 1) % args.save_interval == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            if monitor_loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif monitor_loss == "samo":
                loss_model = samo
            elif monitor_loss == "softmax":
                loss_model = None
            else:
                print("未知的损失函数类型！")
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))

        # ================================================================
        # 阶段F: 保存最佳模型（基于验证损失）+ 早停判断
        # ================================================================
        valLoss = np.nanmean(devloss_dict[monitor_loss])
        if valLoss < prev_loss:
            # 保存当前最佳模型
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            if monitor_loss in ("samo", "samo_spoofall", "ocsoftmax_IDEAL", "samo_channel"):
                loss_model = samo
            elif monitor_loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif monitor_loss == "softmax":
                loss_model = None
            else:
                print("未知的损失函数类型！")
            torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))

            prev_loss = valLoss
            early_stop_cnt = 0
            best_epoch = epoch_num
            optimizer_swa.update_swa()  # SWA 更新
            n_swa_update += 1
        else:
            early_stop_cnt += 1

        # 早停：连续 100 个 epoch 无改进则停止训练
        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 49))
            break

    # ================================================================
    # 训练结束：在测试集上最终评估
    # ================================================================
    if final_test:
        if args.save_score is None:
            args.save_score = args.out_fold[-8:]
        if args.scoring is None:
            args.scoring = monitor_loss
            if monitor_loss == "softmax":
                args.scoring == "fc"
        args.test_model = os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')
        test(args)

    print("Saving best model in epoch {}\n".format(best_epoch))


def update_embeds(device, enroll_model, loader):
    """
    更新说话人中心向量（Speaker Attractor / Embedding）。

    对于注册集中的每个说话人:
      1. 将其所有注册音频通过 AASIST 提取特征 embedding
      2. 对 embedding 取均值
      3. 返回 {speaker_id: mean_embedding} 字典

    参数:
        device:       GPU 设备
        enroll_model: AASIST 模型（eval 模式）
        loader:       注册数据的 DataLoader

    返回:
        enroll_emb_dict: {speaker_id: Tensor(feat_dim,)} 字典
    """
    enroll_emb_dict = {}
    with torch.no_grad():
        for i, (batch_x, _, spk, _, _) in enumerate(tqdm(loader)):
            batch_x = batch_x.to(device)
            batch_cm_emb, _ = enroll_model(batch_x)          # 只取 embedding，不要分类输出
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

            # 按说话人累积 embedding
            for s, cm_emb in zip(spk, batch_cm_emb):
                if s not in enroll_emb_dict:
                    enroll_emb_dict[s] = []
                enroll_emb_dict[s].append(cm_emb)

        # 对每个说话人的所有 embedding 取均值
        for spk in enroll_emb_dict:
            enroll_emb_dict[spk] = Tensor(np.mean(enroll_emb_dict[spk], axis=0))

    return enroll_emb_dict


def test(args):
    """
    独立的模型测试/评估函数。

    加载训练好的模型，在评估集上运行推理，计算 EER 和 t-DCF。

    支持三种评分方式:
      - fc (全连接层):  直接使用 AASIST 分类输出作为分数
      - ocsoftmax:      使用单中心单类方法评分
      - samo:           使用多中心说话人感知方法评分
    """
    torch.set_default_tensor_type(torch.FloatTensor)
    os.makedirs('./test_scores', exist_ok=True)
    save_path = f"./test_scores/{args.save_score}.txt"

    # 如果分数文件已存在，直接计算指标
    if os.path.exists(save_path):
        print("Calculating on existing score file...\n")
        compute_eer_tdcf(args, save_path)
    else:
        # ---- 加载模型 ----
        if args.test_model[-3:] == "pth":
            # AASIST 官方预训练权重（.pth 格式，只有 state_dict）
            with open("aasist/AASIST.conf", "r") as f_json:
                config = json.loads(f_json.read())
            feat_model = get_model(config["model_config"])
            feat_model.load_state_dict(
                torch.load(args.test_model, map_location=args.device))
        else:
            # SAMO 预训练权重（.pt 格式，包含完整模型）
            feat_model = torch.load(args.test_model).to(args.device)

        print("Model loaded : {}".format(args.test_model))
        print("Using scoring=", args.scoring,
              "  testing on val_sp=", args.val_sp,
              "using target-only=", args.target)
        print("Start evaluation...")
        feat_model.eval()

        # ---- 加载测试数据 ----
        _, _, eval_data_loader, train_bona_loader, _, eval_enroll_loader, _ = get_loader(args)

        # ---- 初始化评分模块 ----
        if args.scoring == "ocsoftmax":
            ocsoftmax = OCSoftmax(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake,
                                  alpha=args.alpha,
                                  initialize_centers=args.initialize_centers).to(args.device)
        elif args.scoring == "samo":
            samo = SAMO(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake,
                        alpha=args.alpha).to(args.device)

        with torch.no_grad():
            ip1_loader, utt_loader, idx_loader, score_loader, spk_loader, tag_loader = \
                [], [], [], [], [], []

            # ---- 准备 SAMO 中心向量 ----
            if args.scoring == "samo":
                if args.val_sp:
                    # 使用评估集注册数据提取中心
                    eval_enroll = update_embeds(args.device, feat_model, eval_enroll_loader)
                else:
                    # 使用训练中心或 one-hot 中心
                    if args.one_hot:
                        spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
                        tmp_center = torch.eye(args.enc_dim)[:20]
                        eval_enroll = dict(zip(spklist, tmp_center))
                    else:
                        eval_enroll = update_embeds(args.device, feat_model, train_bona_loader)
                samo.center = torch.stack(list(eval_enroll.values()))

            # ---- 逐批次推理 ----
            for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(eval_data_loader)):
                feat = feat.to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                # 根据评分方式计算分数
                if args.scoring == "samo":
                    if args.target:
                        _, score = samo(feats, labels, spk, eval_enroll, args.val_sp)
                    else:
                        _, score = samo.inference(feats, labels, spk, eval_enroll, args.val_sp)
                elif args.scoring == "fc":
                    if args.test_model[-3:] == "pth":
                        score = feat_outputs[:, 1]  # AASIST 预训练权重标签是反的
                    else:
                        score = feat_outputs[:, 0]  # SAMO 预训练
                elif args.scoring == "ocsoftmax":
                    _, score = ocsoftmax(feats, labels)

                ip1_loader.append(feats)
                idx_loader.append(labels)
                score_loader.append(score)
                utt_loader.extend(utt)
                spk_loader.extend(spk)
                tag_loader.extend(tag)

            # ---- 计算 EER ----
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        # ---- 保存分数 ----
        if args.save_score is not None:
            with open(save_path, "w") as fh:
                for utt, tag, score, label, spk in zip(utt_loader, tag_loader, scores, labels, spk_loader):
                    fh.write(f"{utt} {tag} {label} {score} {spk}\n")
            print(f"Scores saved to {save_path}")

        print(f"Test EER: {eer}")

        # ---- 计算 t-DCF（串联检测代价函数）----
        if args.save_score is not None:
            compute_eer_tdcf(args, save_path)


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    args = init_params()

    if args.test_only:
        # 仅测试模式
        test(args)
    else:
        # 训练模式
        train(args)
