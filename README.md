# SAMO

本仓库提供 ICASSP 2023 论文 ["SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofing"](https://arxiv.org/abs/2211.02718) 的代码实现。

## 环境准备
- 安装依赖
```
pip install -r requirements.txt
```
- 运行环境
  - 1 块 GPU: GeForce GTX 1080 Ti
  - 批量大小 23 时约需 11GB 显存
- 训练/验证/评估数据集:
  - 从[此处](https://datashare.ed.ac.uk/handle/10283/3336)下载 ASVspoof 2019 logical access 数据集
  - 将 'LA' 文件夹路径指定给参数 `--path_to_database`

## 训练
`main.py` 文件包含了 Softmax/OC-Softmax/SAMO 的训练、验证和评估步骤。

例如，训练 SAMO:
```angular2html
python3 samo/main.py -o 'path_to_output_folder' -d 'path_to_database' -p 'path_to_protocol' --overwrite
```

请查看 `main.py` 中的参数设置来指定批量大小和边际值等设置。

## 评估
评估预训练的 SAMO 模型:
```angular2html
python3 samo/main.py --test_only --test_model "./models/samo.pt" --scoring 'samo' --save_score "samo_score"
```
输出结果将显示 `Test EER: 0.008751418248624953`

## 致谢
本项目基于以下开源仓库:
- [OC-Softmax](https://github.com/yzyouzhang/AIR-ASVspoof)
- [AASIST](https://github.com/clovaai/aasist)


## 引用
```bibtex
@inproceedings{ding2023samo,
  title={SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofing},
  author={Ding, Siwen and Zhang, You and Duan, Zhiyao},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023}
}
```


## 参考文献

```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

```bibtex
@ARTICLE{zhang21one,
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters}, 
  title={One-Class Learning Towards Synthetic Voice Spoofing Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={937-941},
  doi={10.1109/LSP.2021.3076358}}
```

```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```
