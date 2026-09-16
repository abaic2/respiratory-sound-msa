# NOTICE — 来源与授权说明

本仓库**不是从零开始的原创工程**，而是基于第三方研究代码改造而来。二次分发前请阅读本文件。

---

## 1. 代码来源

### 1.1 主要基座：MVST（ICASSP 2024）

本仓库的目录结构、数据管线、训练框架、AST 骨干集成，以及以下核心设计，
来自 **Multi-View Spectrogram Transformer (MVST)**：

> He, Wentao; Yan, Yuchen; Ren, Jianfeng; Bai, Ruibin; Jiang, Xudong.
> *Multi-View Spectrogram Transformer for Respiratory Sound Classification.*
> ICASSP 2024, pp. 8626–8630. DOI: `10.1109/ICASSP48485.2024.10445825`

- 原作者仓库（经核查，**两个仓库均未声明任何 License**）：
  - `https://github.com/Huake-EZhou/MVST`
  - `https://github.com/wentaoheunnc/MVST`
- 原始论文思路：将梅尔频谱按 256×1 / 128×2 / 64×4 / 32×8 / 16×16 五种 patch 形状切分，
  各自过独立 Transformer 编码器，再经门控融合（gated fusion）加权后分类。

**授权状态：未声明 License。** 按 GitHub 的默认规则，未声明 License 的仓库
"保留所有权利"（all rights reserved）。这意味着**严格来说，本仓库并不具备
再分发该部分代码的明示许可**。

### 1.2 上游框架：patch-mix 对比学习（INTERSPEECH 2023）

训练框架的下述组件来自 **Patch-Mix Contrastive Learning**：

- `method/patchmix.py`、`method/patchmix_cl.py`（`PatchMixLoss` / `PatchMixConLoss`）
- `--method supcon`、`--negative_pair`、`--target_type grad_block|grad_flow` 等参数
- `--ma_update` / `--ma_beta` 滑动平均更新、swanlab 实验记录集成

> Sangmin Bae, June-Woo Kim, Won-Yang Cho, Hyerim Baek, Soyoun Son, Byungjo Lee,
> Changwan Ha, Kyongpil Tae, Sungnyun Kim, Se-Young Yun.
> *Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
> Respiratory Sound Classification.* INTERSPEECH 2023, pp. 5436–5440.
> DOI: `10.21437/Interspeech.2023-1426`

- 作者仓库：`https://github.com/raymin0223/patch-mix_contrastive_learning`
- **授权状态：MIT License。** 该部分可以自由使用与再分发，须保留版权声明。

### 1.3 数据集

实验使用 **ICBHI 2017 Challenge Respiratory Sound Database**，受其自身使用协议约束：

> Rocha B, Filos D, Mendes L, et al. *A respiratory sound database for the development
> of automated classification algorithms.* Physiological Measurement, 2019.

数据集音频样本（`*.wav`）**未包含在本仓库中**。

---

## 2. 本仓库的改动

在上述基座上新增/改造的部分：

1. **MSA / MSAF 多尺度注意力模块**（各变体 `models/model_utils.py`）——
   结构改写自 AFF / iAFF（Attentional Feature Fusion），接入 4×4 / 8×8 / 16×16
   三个尺度上下文分支，并提供 `standard / learnable / adaptive` 三种跳跃连接。
2. **两个插入位置**：`PrePatchMSA`（patch embedding 之前，作用于原始频谱图）
   与 `MSA_small_with_skip`（作用于 AST 输出的 768 维特征）。
3. **`Blocks.py`** 中 5 个肺音专用即插即用模块。
4. **消融实验组织**：33 个实验入口（前端特征 7 / 骨干网络 6 / MSA 6 / MSAF 1 /
   可视化 12 / 基线 1）。

---

## 3. 关于 License

**本仓库刻意不附加任何 License。**

原因：仓库主体是未声明 License 的第三方论文代码，附加 MIT / Apache 等许可协议
等同于替原作者做出授权声明，这是不恰当的。MIT 覆盖的仅限 §1.2 所列的
patch-mix 部分。

**使用者须知**：

- 用于学习、研究、复现实验：通常没有问题，但**发表或再分发时应引用 §1 的两篇论文**
- 用于商业用途或公开再分发：**请先联系 §1.1 的原作者取得许可**
- 如需引用本仓库的 MSA / MSAF 改动，请同时引用 §1.1 的 MVST 论文

---

## 4. 声明

本仓库作者与 §1 中任一篇论文的作者无隶属关系，也未获得其对本仓库的背书。
上述来源信息基于公开的仓库与论文元数据整理，如原作者认为本仓库的使用方式不妥，
请联系仓库所有者删除。
