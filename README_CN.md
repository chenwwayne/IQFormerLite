# 📡 IQFormerLite

> **IQFormerLite** 是一个基于 PyTorch 的轻量化自动调制识别（Automatic Modulation Classification, AMC）框架，面向无线电 IQ 信号分类与边缘端实时部署研究。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Ready-ee4c2c)
![Task](https://img.shields.io/badge/Task-Automatic%20Modulation%20Classification-green)
![Datasets](https://img.shields.io/badge/Datasets-RML2016.10a%20%7C%20RML2016.10b-orange)

📘 英文版：[README.md](README.md)

📄 论文：[IQFormerLite: A hardware-efficient framework for automatic modulation recognition](https://ftp.nowpublishers.com/ijicc/article-abstract/doi/10.1108/IJICC-02-2026-0175/1397768/IQFormerLite-a-hardware-efficient-framework-for?redirectedFrom=fulltext)<br>
DOI：[10.1108/IJICC-02-2026-0175](https://doi.org/10.1108/IJICC-02-2026-0175)

---

## 🧠 从原始 IQ 信号到边缘端推理

IQFormerLite 使用 Dynamic Fusion Embedding、可学习 KAN 滤波器组（Learnable KAN Filterbank，LKF）和大卷积核编码器处理原始 I/Q 序列。在保留长程上下文建模与自适应频谱提取能力的同时，模型将循环结构和较重的注意力路径替换为更适合边缘 NPU 编译的算子。

![IQFormerLite 网络架构](assets/paper/overview.png)

最终得到的模型可以用于 RadioML2016.10A 和 RadioML2016.10B 的实时自动调制识别实验。论文同时评估识别精度与实际部署效率，而不是只用参数量定义模型是否轻量。

## 📈 IQFormerLite 为什么更轻量

下面的参数量–准确率对比展示了模型的设计目标：在两个数据集上，IQFormerLite 使用显著更少的参数，同时保持接近 IQFormer 的识别精度。

![RadioML2016.10A 与 RadioML2016.10B 上的准确率–参数量对比](assets/paper/param_vs_acc_bubble.png)

| 数据集 | IQFormerLite 总体准确率 | 参数量 | FLOPs |
|---|---:|---:|---:|
| RadioML2016.10A | 63.18 ± 0.22% | 0.13 M | 32.06 M |
| RadioML2016.10B | 65.51 ± 0.13% | 0.13 M | 32.06 M |

与 IQFormer 相比，IQFormerLite 的参数量从约 0.35 M 降至 0.13 M，减少约 63.8%，同时保持具有竞争力的总体准确率。

---

## ✨ 项目亮点

- 面向实时自动调制识别实验的轻量级 IQFormerLite 主干网络。
- 支持 IQ-only、STFT 融合、卷积滤波器组特征以及 KAN/LKF 滤波特征。
- 支持在 RML2016.10a 与 RML2016.10b 上进行训练与评估。
- 集成 IQFormer、MCFormer、AMCNet、MCLDNN、PET-CGDNN、FEA-T128、FEA-T1024 等对比模型。
- 提供多随机种子实验、LKF 消融、卷积核扫描、结果汇总与绘图脚本。

下面的 SNR 曲线展示了轻量化设计保留性能的区间：在中高信噪比条件下，IQFormerLite 的准确率逐渐接近表现更强的基线模型。

| RadioML2016.10A | RadioML2016.10B |
|---|---|
| ![RadioML2016.10A 分 SNR 准确率](assets/paper/rml2016_10a_sota_accuracy.png) | ![RadioML2016.10B 分 SNR 准确率](assets/paper/rml2016_10b_sota_accuracy.png) |

---

## 🧭 目录结构

```text
.
|-- main.py                         # 训练、验证、测试主入口
|-- train.sh                        # 示例批量训练脚本
|-- dataset/                        # RadioML 数据加载与处理工具
|-- model/                          # IQFormerLite 与基线模型定义
|-- utils/                          # 训练循环、模型报告、绘图与辅助函数
|-- scripts/                        # 实验运行与结果汇总脚本
|-- notebook/                       # 分析与绘图工具
|-- assets/paper/                   # README 中嵌入的论文图片
|-- logs/                           # 运行日志、TensorBoard、混淆矩阵、t-SNE
|-- save_models/                    # 训练得到的模型权重
`-- rknn/                           # 边缘端/NPU 部署与 RKNN 相关文件
```

> ⚠️ `logs/`、`save_models/`、TensorBoard 事件文件以及本地实验输出体积可能很大。
> `results/`、`paper/`、`.trae/` 等本地目录已明确排除，不再同步到 Gitee。

---

## 🛠️ 环境配置

本项目基于 PyTorch 实现。若需要 GPU 训练，请安装与 CUDA/驱动匹配的 PyTorch 版本。

### 核心依赖

- Python 3.9+
- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- h5py
- tensorboardX
- tqdm
- timm

### 示例安装

```bash
conda create -n iqformerlite python=3.11 -y
conda activate iqformerlite

# 请先根据本机 CUDA/驱动版本安装合适的 PyTorch。
pip install torch torchvision torchaudio

# 安装常用工具库。
pip install numpy pandas scipy scikit-learn matplotlib seaborn h5py tensorboardX tqdm timm
```

---

## 📦 数据集准备

实验使用 DeepSig RadioML 2016 系列数据集：

- `RML2016.10a.pkl`
- `RML2016.10b.dat`

请将数据文件放入 `dataset/`：

```text
dataset/
|-- RML2016.10a.pkl
`-- RML2016.10b.dat
```

也可以通过 `--database_path` 指定自定义数据集目录。

支持的数据集选项：

- `2016.10a`
- `2016.10b`

---

## 🚀 快速开始

### 1. 快速验证

正式训练前，建议先运行一个短流程验证数据、模型和环境是否正常。

```bash
cd /Users/chenww/develop/IQFormerLite/code

python main.py \
  --database_path ./dataset \
  --database_choose 2016.10a \
  --model IQFormerLite \
  --aux_mode kan \
  --kernel_size 31 \
  --grid_size 4 \
  --grid_range -2 2 \
  --batch_size 1024 \
  --num_epochs 1 \
  --dry_run \
  --comment IQFormerLite_dry_run
```

### 2. 训练 IQFormerLite

```bash
python main.py \
  --database_path ./dataset \
  --database_choose 2016.10a \
  --model IQFormerLite \
  --aux_mode kan \
  --kernel_size 31 \
  --grid_size 4 \
  --grid_range -2 2 \
  --batch_size 1024 \
  --num_epochs 100 \
  --seed 1234 \
  --comment IQFormerLite_seed1234
```

### 3. 运行示例脚本

运行前请先检查并修改 `train.sh` 中的 `DATABASE_PATH`、`DATABASE_CHOOSE`、`CUDA_VISIBLE_DEVICES` 和训练参数。

```bash
bash train.sh
```

---

## ⚙️ 主要参数

| 参数 | 说明 |
|---|---|
| `--database_path` | 数据集目录 |
| `--database_choose` | 数据集名称：`2016.10a` 或 `2016.10b` |
| `--model` | 模型名称 |
| `--aux_mode` | 辅助特征模式：`none`、`stft`、`conv`、`kan` |
| `--band_k` | 频带数量 |
| `--kernel_size` | KAN/LKF 滤波核大小 |
| `--grid_size` | KAN 网格大小 |
| `--grid_range` | KAN 网格范围，例如 `-2 2` |
| `--lkf_variant` | LKF 消融变体 |
| `--batch_size` | 训练 batch size |
| `--eval_batch_size` | 验证/测试 batch size |
| `--num_epochs` | 最大训练 epoch 数 |
| `--lr` | 学习率 |
| `--seed` | 随机种子 |
| `--model_path` | 可选模型权重路径 |
| `--dry_run` | 使用小数据子集快速验证 |
| `--report_only` | 只生成模型报告并退出 |
| `--skip_post_test_artifacts` | 写出 `Test_ACC.csv` 后跳过混淆矩阵和 t-SNE |

支持模型：

```text
IQFormerLite, IQFormer, MCFormer, AMCNET, MCLDNN, PETCGDNN, FEA_T128, FEA_T1024
```

推荐的 IQFormerLite 配置：

```text
--model IQFormerLite --aux_mode kan --kernel_size 31 --grid_size 4 --grid_range -2 2
```

这些 LKF 参数并非随意设置：网格大小和网格范围扫描最终选取 `grid_size=4`、`grid_range=[-2, 2]` 以及 `kernel_size=31` 作为论文结果中的配置。

![LKF 网格大小与网格范围消融结果](assets/paper/lkf_grid_ablation.png)

---

## 📊 输出文件

每次运行会生成如下形式的 run tag：

```text
model_<dataset>_<epochs>_<batch_size>_<lr>_<comment>
```

运行输出：

```text
logs/<run_tag>/
|-- Train_Epoch.csv                 # 每个 epoch 的训练集分 SNR 准确率
|-- Val_Epoch.csv                   # 每个 epoch 的验证集分 SNR 准确率
|-- Test_ACC.csv                    # 测试集分 SNR 准确率及 Avg 汇总行
|-- Test_mod_SNR.csv                # 不同调制类别在各 SNR 下的准确率
|-- model_report.txt                # 模型复杂度与结构报告
|-- confusionMatrix/                # 混淆矩阵
`-- tsne/                           # t-SNE 可视化结果
```

最优权重保存路径：

```text
save_models/<run_tag>/weight.pt
```

> ✅ `Test_ACC.csv` 是结果汇总脚本最关键的输入文件。

---

## 🚀 边缘端部署

部署目标是 Orange Pi 5 Plus 上的 Rockchip RK3588 NPU。该硬件友好的推理路径是模型设计目标的一部分：除 PyTorch 精度外，评测还关注编译模型的延迟、吞吐、内存占用、NPU 利用率和量化后的准确率。

<img src="assets/paper/orangepi5.png" alt="Orange Pi 5 Plus 部署平台" width="50%">

在论文评估的 INT8 RKNN 协议下，IQFormerLite 达到 **0.157 ms/样本** 的延迟和 **6,387.84 samples/s** 的吞吐，配套 RKNN 验证准确率为 **63.11%**。

---

## 🧾 模型报告

`main.py` 会在训练前写出模型报告：

```text
logs/<run_tag>/model_report.txt
```

仅生成模型报告：

```bash
python main.py \
  --database_path ./dataset \
  --database_choose 2016.10a \
  --model IQFormerLite \
  --aux_mode kan \
  --report_only \
  --comment IQFormerLite_report
```

---

## 📌 推荐工作流

1. **准备数据**  
   将 `RML2016.10a.pkl` 和/或 `RML2016.10b.dat` 放入 `dataset/`。

2. **快速验证**  
   使用 `--dry_run` 确认环境、模型和数据集路径正常。

3. **训练模型**  
   运行 `main.py` 或 `scripts/` 下的实验脚本。

4. **检查结果**  
   优先查看 `logs/<run_tag>/Test_ACC.csv` 和 `model_report.txt`。

5. **汇总结果**  
   使用对应的 `scripts/summarize_*.py` 脚本。

---

## 📄 许可证

项目许可证见 `LICENSE`。第三方数据集、依赖库、模型实现与部署工具链可能有各自独立的许可证与使用条款。
