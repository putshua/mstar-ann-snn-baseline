# SEW ResNet18 训练与神经元剪枝

模型和剪枝流程来自 [xyshi2000/MSTAR-SNN](https://github.com/xyshi2000/MSTAR-SNN)，固定版本 `c3e696a48ccc2bcf0896efa546de0c0a2dbfc05a`。

本地适配包括 SpikingJelly 旧版 `np.int` 的 NumPy 兼容处理、可跳过逐通道掩码绘图的开关，以及包导入支持。SEW 残差结构、100 维 boost 读出、LIF、掩码、损失和训练配方保留原实现。

未剪枝 SEW-ResNet-18 为 **11,221,540** 个参数、**811,008** 个 LIF 单元；其中 20 个 LIF 模块。原 ANN ResNet-18 为 11,175,370 个参数，差异主要来自 512→100→10 的 boost 读出。

掩码剪枝保留卷积张量尺寸；需分别报告原网络参数、掩码参数、剩余神经元、SOP 估计及实测延迟。不能用零掩码比例直接代替模型参数压缩率或实测加速比。

从仓库根目录运行：

```bash
pip install -r pruning/requirements.txt

# 未剪枝 SEW 对照，100 epochs
CUDA_VISIBLE_DEVICES=0 python pruning/main.py \
  --config pruning/configs/image_classification/sew_resnet18_mstar_train.yaml \
  --data-path /path/to/soc --output-dir outputs/sew_dense --skip-mask-plots

# 从头训练掩码 100 epochs，再固定二值掩码微调 20 epochs
CUDA_VISIBLE_DEVICES=0 python pruning/main.py \
  --config pruning/configs/image_classification/sew_resnet18_mstar_prune.yaml \
  --data-path /path/to/soc --output-dir outputs/sew_pruned --skip-mask-plots

# 独立评估训练得到的 checkpoint
CUDA_VISIBLE_DEVICES=0 python pruning/main.py --test-only \
  --resume outputs/sew_pruned/checkpoint_final.pth --resume-type test \
  --data-path /path/to/soc --output-dir outputs/sew_pruned_eval --skip-mask-plots

# 加载本地精简权重；未剪枝同理，替换 checkpoint 路径
CUDA_VISIBLE_DEVICES=0 python pruning/main.py --test-only \
  --resume checkpoints/sew_resnet18_pruned.pt --resume-type test \
  --data-path /path/to/soc --output-dir outputs/sew_pruned_eval --skip-mask-plots
```

采用同一十类 SOC、单通道 128×128、T=4、seed=0。保持上游配方的 bias/norm 不衰减、cosine 最低学习率为 0、训练 `drop_last=True`；这些细节与旧 ANN/post-add SNN 基线不完全相同。

新 SEW 和剪枝 checkpoint 使用上游 `config`、`mstar_metadata`、`model` 格式；原 ANN/post-add SNN 权重仍使用根目录 `evaluate.py`，两种结构分别加载。

本机结果（最终模型，2425 张 SOC 测试图）：未剪枝 **97.9794%**，剪枝 **95.8763%**。剪枝后保留 11,970 / 811,008 个空间 LIF 单元，准确率比这组未剪枝配方低 2.1031 个百分点。分类指标、参数统计、混淆矩阵和 batch=1 延迟见 [完整结果](../results/analysis/metrics.json)。新增 SEW checkpoint 暂留本机，后续可单独打包分发。
