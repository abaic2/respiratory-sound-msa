# respiratory-sound-msa

基于 **MVST（Multi-View Spectrogram Transformer, ICASSP 2024）** 的肺音分类实验工程，
在其上引入了 **MSA / MSAF（多尺度注意力 / 多尺度注意力融合）** 模块，并构建了一套
覆盖前端特征、骨干网络、注意力插入位置的消融实验矩阵。

> ⚠️ **本仓库基于第三方论文代码改造，二次分发请先阅读 [`NOTICE.md`](NOTICE.md)。**
> 上游 MVST 仓库未声明 License，本仓库同样**不附加任何 License**。

---

## 任务

**ICBHI 2017 呼吸音数据库**，4 分类（正常 / 啰音 crackle / 哮鸣音 wheeze / 两者兼有），
评价指标为 ICBHI 官方 Average Score = (Sensitivity + Specificity) / 2。

---

## 本人在基座上的改动

### 1. MSA / MSAF 多尺度注意力模块

实现在各变体的 `models/model_utils.py`，改写自 **AFF / iAFF（Attentional Feature Fusion）**，
接入多尺度上下文分支：

```python
context1 = AdaptiveAvgPool2d((4, 4))     # 小尺度：局部时频模式
context2 = AdaptiveAvgPool2d((8, 8))     # 中尺度
context3 = AdaptiveAvgPool2d((16, 16))   # 大尺度
global_att = AdaptiveAvgPool2d(1)        # 全局
# 上采样回原尺寸后相加，sigmoid 出权重
wei = sigmoid(local_att(x) + global_att(x) + c1 + c2 + c3)

MSA :  xo = 2 * x * wei                        # 自调制（单输入）
MSAF:  xo = 2 * x * wei + 2 * residual * (1 - wei)   # 双输入融合
```

### 2. 两个插入位置

| 位置 | 变体 | 实现 |
|---|---|---|
| **Pre-Patch** | `baseMSA/baseMSA_start*` | `PrePatchMSA`：直接在 1×128×1024 原始频谱图上增强，再做 patch embedding。尺度为 32×256 / 64×512，`fusion_weights = nn.Parameter(torch.ones(4))` 可学习加权求和，支持 `standard / learnable / adaptive` 三种跳跃连接 |
| **Post-Patch** | `baseMSA/baseMSA_end` | `MSA_small_with_skip(channels=768)`：在 AST 输出的 768 维特征上做 MSA + 残差融合 |

### 3. 肺音专用即插即用模块

`Blocks.py` 收录 5 个带论文出处的候选模块：`BellWeightedAttention`、
`BreathingCyclePerceptionTransformer`、`DiseasePatternDecompositionAttention`、
`FrequencySplitDualPathNetwork`、`VariationalInformationBottleneckEnhancer`。

---

## 目录结构

工程采用**每个实验一份完整代码副本**的组织方式（便于对照，但重复度较高，见下文）：

```
.
├── base/                原始基线（AST + patch-mix 原样）
├── basefeature/         前端声学特征消融：7 个变体
│                        Chromagram / cochleagram / CQT / gammatone /
│                        logmel / MFCC / spectrogram
├── basemodels/          骨干网络消融：6 个变体
│   ├── basecnn/         ResNet-18, GoogLeNet v1, Inception-v3, EfficientNet-B0
│   ├── basernn/         GRU / LSTM / BiLSTM（small / medium / large）
│   ├── basetransformer/ DeiT-base384, ViT-base, ViT-small, Swin-tiny
│   ├── basemamba/       Mamba
│   ├── basewhisper/     Whisper tiny / base / small
│   └── basemachinelearning/  SVM / RandomForest / KNN
├── baseMSA/             ★ MSA 插入位置消融：6 个变体
├── baseMSAF/            ★ MSAF + 双特征输入（mel+cqt / mel+mfcc / fbank+mel …）
├── basevisualization/   预处理与 AST / MSA 特征可视化：12 个入口
└── docs/
    ├── 实验矩阵.csv              33 个实验入口的结构化索引
    └── 代码盘点与实验矩阵.md      完整代码盘点报告
```

**共 33 个实验入口**（每个入口 = 一个含 `main.py` 的目录）：

| 分支 | 入口数 | 变的是什么 |
|---|---:|---|
| `base` | 1 | 原始基线 |
| `basefeature` | 7 | 前端声学特征 |
| `basemodels` | 6 | 骨干网络 |
| `baseMSA` | 6 | ★ MSA 插入位置与融合方式 |
| `baseMSAF` | 1 | ★ MSAF + 双特征输入 |
| `basevisualization` | 12 | 预处理 / 特征可视化 |

完整的入口级索引（含每个入口的代码行数、入口文件哈希、run 命令、关键开关）
见 [`docs/实验矩阵.csv`](docs/实验矩阵.csv)。

---

## 运行

```bash
pip install -r requirements.txt
```

### 1. 准备数据与预训练权重

- 从 [ICBHI 官方](https://bhichallenge.med.auth.gr/) 下载 ICBHI 2017 Challenge
  Respiratory Sound Database，放到一个数据目录下
- 下载 AudioSet 预训练的 AST 权重（**16×16 patching 版本**），放到各变体的
  `pretrained_models/` 目录

### 2. 单次训练

```bash
cd base          # 或任意变体目录
python main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 \
  --class_split lungsound --n_cls 4 --epochs 50 --batch_size 8 \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained --method ce \
  --data_folder /path/to/ICBHI_final_database
```

> `--data_folder` 的默认值指向原作者的服务器路径，**务必用参数覆盖或改代码**。
> 各变体目录下的 `run.txt` 记录了该变体对应的完整命令。

### 3. 特色开关

```bash
# MSA（Pre-Patch / Post-Patch 见变体目录归属）
--use_msa --use_skip_connection --skip_ratio 0.1 --skip_type standard

# 双特征输入
--use_dual_feature --feature_combo mel+cqt --model_size base384 --spatial_size 8
```

---

## 本地留存的结果

| 模型 | Accuracy | Sensitivity | Specificity | ICBHI Score |
|---|---:|---:|---:|---:|
| SVM | 52.36 | 13.51 | 74.86 | **44.18** |
| RandomForest | 50.83 | 13.46 | 73.72 | **43.59** |
| KNN | 46.15 | 15.80 | 63.46 | **39.63** |
| Whisper-tiny | — | 20.31 | 82.52 | **51.41** |

> 传统 ML 结果见 `basemodels/basemachinelearning/results/*/*_results.json`；
> Whisper 见 `basemodels/basewhisper/save/16/results.json`（格式 `[Specificity, Sensitivity, Score]`）。
> 所有模型敏感度都明显偏低，靠高特异度拉分——这是 ICBHI 类别极不平衡的典型表现。
>
> **MSA / MSAF 系列实验的结果未包含在本仓库中**（原记录在服务器的实验追踪平台上）。

---

## 关于代码重复度

本工程包含 **776 个 `.py` 文件，但只有 118 个是唯一内容——重复率 75.6%**
（8.86 MB → 2.16 MB）。例如 `time_warping.py` 有 49 份完全一致，
`patchmix.py`、`patchmix_cl.py`、`Blocks.py`、`astblock.py`、`save_features.py`
各有 28~33 份一字不差。

真正承载实验差异的只有四个文件：`main.py`（15 个版本）、`icbhi_util.py`（15 个版本）、
`icbhi_dataset.py`（12 个版本）、`ast.py`（8 个版本）。

这是"复制目录式版本管理"的结果，本仓库**保留原样以维持各变体的可运行性**。
如需继续扩展实验，建议改为单入口配置驱动（用 `--msa_pos / --msa_fusion / --skip_type`
这类开关替代复制目录），详见 `docs/代码盘点与实验矩阵.md` §6.3。

---

## 未包含的内容

为控制仓库体积，以下内容**未上传**（它们不属于算法代码）：

| 内容 | 体积 | 说明 |
|---|---:|---|
| `*.pth` 训练权重 | 1.0 GB | 5 个 Whisper 检查点，各 204 MB，超 GitHub 100 MB 单文件上限 |
| `*.png` 可视化图 | 158.6 MB | 178 张特征/注意力可视化图 |
| `*.pkl` 传统 ML 模型 | 46.6 MB | SVM / RF / KNN 训练产物 |
| HuggingFace 缓存 | 99.7 MB | 一个下载中断的 `.incomplete` 文件 |
| `*.wav` 数据集样本 | 3.9 MB | ICBHI 音频样本，受数据集自身使用协议约束 |
| `*.pyc` 字节码 | 6.6 MB | 可再生 |

---

## 引用

如果本仓库的代码对你的研究有帮助，请引用上游工作：

```bibtex
@inproceedings{he2024multi,
  title     = {Multi-View Spectrogram Transformer for Respiratory Sound Classification},
  author    = {He, Wentao and Yan, Yuchen and Ren, Jianfeng and Bai, Ruibin and Jiang, Xudong},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {8626--8630},
  year      = {2024},
  doi       = {10.1109/ICASSP48485.2024.10445825}
}

@inproceedings{bae23b_interspeech,
  title     = {Patch-Mix Contrastive Learning with Audio Spectrogram Transformer
               on Respiratory Sound Classification},
  author    = {Sangmin Bae and June-Woo Kim and Won-Yang Cho and Hyerim Baek
               and Soyoun Son and Byungjo Lee and Changwan Ha and Kyongpil Tae
               and Sungnyun Kim and Se-Young Yun},
  year      = {2023},
  booktitle = {INTERSPEECH 2023},
  pages     = {5436--5440},
  doi       = {10.21437/Interspeech.2023-1426},
  issn      = {2958-1796}
}
```

数据集请引用 ICBHI 2017 Challenge 官方论文（Rocha et al., *Physiological Measurement*, 2019）。
