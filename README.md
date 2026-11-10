# MSTAR ANN / SNN / SEW-ResNet-18

MSTAR 十类 SOC：17° 训练 2747 张，15° 测试 2425 张；单通道 128×128，seed=0，SNN 为 T=4。全部分类指标使用最终权重、FP32 和完整测试集。

## 结果

延迟单位为 **ms/图，batch size=1**，在本机单张 NVIDIA L20Z 上测得。

| 模型 | Accuracy | Macro-F1 | LIF 保留率 | 平均延迟 | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN ResNet-18 | 98.9278% | 98.9044% | — | 1.9864 | 1.4587 | 4.5600 |
| SNN ResNet-18，post-add LIF | 97.4021% | 97.3755% | 100.0000% | 7.0699 | 5.1375 | 16.0199 |
| SEW-ResNet-18，本机重训 | 97.9794% | 97.9282% | 100.0000% | 7.7866 | 5.6525 | 17.7157 |
| SEW 剪枝 λ=1e-5，本机重训 | 95.8763% | 95.7298% | 1.4759% | 8.7897 | 6.3727 | 20.1023 |
| SEW-ResNet-18，上传权重 | 98.1856% | 98.0824% | 100.0000% | 7.7576 | 5.6412 | 17.5276 |
| SEW 剪枝 λ=5e-6，上传权重 | 96.1649% | 96.0894% | 2.7434% | 8.7889 | 6.3660 | 20.0163 |
| SEW 剪枝 λ=1e-5，上传权重 | 95.2165% | 95.1890% | 1.5431% | 8.7903 | 6.3683 | 19.8949 |
| SEW 剪枝 λ=2e-5，上传权重 | 89.5670% | 89.4382% | 0.8134% | 8.7827 | 6.3669 | 19.9368 |

计时使用 FP32 eager 模式、单 CPU 线程并固定 CPU 核；每个模型预热 100 次，在相同的 256 张分层测试图片上重复 5 轮，每张图的模型执行顺序打乱并交错运行。每次 CUDA 同步，计入完整前向、SNN 的四步编码与时间读出、状态重置；不计读图、预处理和 CPU→GPU 传输。均值/P50/P95 包含共享宿主的调度波动，不是 batch=64 吞吐量折算。CUDA Event 时间和每轮均值另存于 JSON/CSV。

ANN / post-add SNN 为 11,175,370 个参数；SEW 为 11,221,540 个网络参数，差异来自 512→100→10 的 boost 分类头。post-add SNN 有 17 个 LIF 模块、753,664 个空间单元；SEW 有 20 个模块、811,008 个空间单元。剪枝版额外包含 811,008 个掩码参数。

掩码剪枝关闭空间神经元输出，卷积张量尺寸保持不变，保留率不代表参数压缩率或实测加速比。SEW 训练 100 epochs；剪枝从头训练掩码 100 epochs，再固定二值掩码微调 20 epochs。原 post-add SNN 使用 SpikingJelly torch 后端，SEW 使用 CuPy 后端，延迟比较包含各自实现差异。

## 完整指标

- [完整 JSON](results/analysis/metrics.json)：分类指标、混淆矩阵、参数/神经元数量、SOP 估计、BS=1 延迟统计、训练配置和预处理。
- [汇总 CSV](results/analysis/summary.csv)；[逐类别 CSV](results/analysis/per_class.csv)。
- [混淆矩阵](results/analysis/confusion_matrices.png)、[ROC 曲线](results/analysis/roc_curves.png)、[PR 曲线](results/analysis/precision_recall_curves.png)；目录中同时提供 SVG。

指标包括 Accuracy、Top-3/Top-5、Balanced accuracy、Macro precision/recall/F1、Weighted F1、ROC-AUC macro/micro、mAP/AP、κ、MCC、Log loss、Brier 和 ECE；逐类包含 Precision、Recall、F1、Specificity、ROC-AUC 与 AP。ROC/AP 按 one-vs-rest 计算，mAP 为分类 AP 的类别均值；ECE 使用 15 个等宽置信度分箱，Brier 对十类求和后对样本取均值。SOP 仅统计脉冲驱动卷积，不含静态 stem 和 fc，也不是实测能耗。

## 复现

数据使用 [MSTAR SOC 镜像](https://github.com/omae11/MSTAR-dataset)，固定版本 `d99007290d20ce52933c8bf0f8c3f461acd522ee`，结构为 `soc/{train,test}/{class}/*.jpg`。这是 JPEG 镜像上的实验；SOC 逐类数量对应 [ASIR-Net 表 2](https://doi.org/10.3390/rs13153029)。

```bash
pip install -r requirements-analysis.txt -r pruning/requirements.txt

# ANN / post-add LIF SNN 训练
CUDA_VISIBLE_DEVICES=0 python train.py --data /path/to/soc --arch resnet18 --seed 0 --output outputs/ann
CUDA_VISIBLE_DEVICES=0 python train.py --data /path/to/soc --arch spiking_resnet18 --time-steps 4 --seed 0 --output outputs/snn

# 完整指标与 BS=1 计时；37 为本机使用的 CPU 核，可按机器调整
CUDA_VISIBLE_DEVICES=0 python benchmark.py --data /path/to/soc/test --cpu-core 37 \
  --uploaded-dir /path/to/uploaded_checkpoints

# 仅更新 BS=1 计时，保留已有分类指标
CUDA_VISIBLE_DEVICES=0 python benchmark.py --data /path/to/soc/test --cpu-core 37 \
  --uploaded-dir /path/to/uploaded_checkpoints --latency-only
```

`benchmark.py` 默认加载 `checkpoints/` 中四组本机权重；`--uploaded-dir` 可选，用于额外加载 `sew_resnet18_mstar_{dense,5e-6,1e-5,2e-5}/checkpoint_final.pth`。本次新增 SEW checkpoint 暂不提交远端，后续单独打 zip；旧 ANN / post-add SNN 权重仍在 `checkpoints/`。

SEW 的训练、剪枝及独立评估命令见 [pruning/README.md](pruning/README.md)。代码来自 [xyshi2000/MSTAR-SNN](https://github.com/xyshi2000/MSTAR-SNN/tree/c3e696a48ccc2bcf0896efa546de0c0a2dbfc05a)，保留其模型和剪枝配方。SEW 与旧 ANN/post-add SNN 的 bias/norm 衰减、cosine 最低学习率和 `drop_last` 设置有差异。
