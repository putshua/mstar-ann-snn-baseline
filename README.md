# MSTAR ANN / SEW 推理基线

MSTAR 十类 SOC：17° 训练 2747 张、15° 测试 2425 张；单通道 128×128，FP32、BS=1、SNN T=4。保留 ResNet-18 既有对照，并新增本机 ResNet-34 的神经元/性能结果。

## ResNet-34：神经元与性能

本机 GPU 3（L20Z）从头训练，SOC、seed=0。ANN 训练 100 epochs；SEW 训练掩码 100 epochs，再固定二值掩码微调 20 epochs，T=4。分类和计时均使用最终权重、FP32、BS=1；CUDA Event 计时 5 个 run，时间为 avg ± std（ms/图）。

| 模型 | 神经元/激活总数 | 保留数 | 保留率 | Accuracy | Macro-F1 | 推理时间 avg ± std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN ResNet-34 | 1,163,264 | 1,163,264 | 100.0000% | 98.9278% | 98.8939% | 4.8804 ± 1.5444 |
| SEW-ResNet-34，λ=2e-5 | 1,220,608 | 5,977 | 0.4897% | 85.8969% | 85.3944% | 11.2500 ± 0.3842 |
| SEW-ResNet-34，λ=1e-5 | 1,220,608 | 12,078 | 0.9895% | 94.5567% | 94.5637% | 16.1066 ± 6.4382 |

λ=1e-5 保留 12,078 / 1,220,608 个 LIF 单元，即 **0.9895%**，满足约 1% 的目标。ANN 统计 ReLU 空间激活位置，SEW 统计 LIF 空间单元，均不乘 batch 或 T。ANN 有 33 个激活位置，SEW 有 36 个 LIF 模块（包括下采样支路）。

ANN 网络参数 21,283,530；SEW 网络参数 21,329,700，另有 1,220,608 个 mask 参数。此处是原始 masked 推理，卷积张量尺寸未缩小，未对 ResNet-34 追加融合/稀疏性能实验。两组 SNN 的计时波动不同，保留原始五次均值及 std，不据此判断两个 λ 的速度优劣。

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --data /path/to/soc --arch resnet34 \
  --epochs 100 --seed 0 --output outputs/ann34

CUDA_VISIBLE_DEVICES=3 python pruning/main.py \
  --config pruning/configs/image_classification/sew_resnet18_mstar_prune.yaml \
  --model sew_resnet34 --penalty-lmbda 1e-5 --seed 0 \
  --data-path /path/to/soc --output-dir outputs/sew34 --skip-mask-plots --no-save-latest

# 单个新 checkpoint：评估一次指标，计时 5 次
CUDA_VISIBLE_DEVICES=3 python benchmark_inference.py --data /path/to/soc/test \
  --sew-checkpoint /path/to/sew_resnet34.pt --output outputs/sew34_eval --evaluate
```

ANN 和 SEW 可分别用 `--ann-checkpoint` / `--sew-checkpoint`，或同时提供以比较一对权重。两者均支持 ResNet-18/34。新增权重仍留本机，不提交远端。

## ResNet-18 既有结果

| 模型 / 版本 | Accuracy | Macro-F1 | 推理图参数元素 | 推理时间 avg ± std（ms/图） |
| --- | ---: | ---: | ---: | ---: |
| ann/baseline | 98.9278% | 98.9044% | 11,175,370 | 1.4998 ± 0.0582 |
| ann/fold_bn | 98.9278% | 98.9044% | 11,170,570 | 1.0436 ± 0.0387 |
| sew_dense/baseline | 98.1031% | 98.0134% | 11,221,540 | 5.8518 ± 0.2067 |
| sew_dense/bn_lif | 98.1031% | 98.0134% | 11,211,940 | 1.9113 ± 0.0791 |
| pruned_5e-6/baseline | 96.3299% | 96.2426% | 12,032,548 | 6.5824 ± 0.2233 |
| pruned_5e-6/bn_lif | 96.3299% | 96.2426% | 11,211,940 | 1.9147 ± 0.0679 |
| pruned_5e-6/sparse | 96.2474% | 96.1715% | 4,634,596 | 1.6699 ± 0.0649 |
| pruned_1e-5/baseline | 95.3402% | 95.3142% | 12,032,548 | 6.5954 ± 0.2301 |
| pruned_1e-5/bn_lif | 95.3402% | 95.3142% | 11,211,940 | 1.9195 ± 0.0739 |
| pruned_1e-5/sparse | 95.1753% | 95.1455% | 2,982,628 | 1.5305 ± 0.0614 |
| pruned_2e-5/baseline | 89.6082% | 89.4968% | 12,032,548 | 6.5931 ± 0.2379 |
| pruned_2e-5/bn_lif | 89.6082% | 89.4968% | 11,211,940 | 1.9205 ± 0.0736 |
| pruned_2e-5/sparse | 89.4845% | 89.4076% | 1,793,188 | 1.5256 ± 0.0546 |

完整分类指标、逐类指标和混淆矩阵：[results/metrics.json](results/metrics.json)；汇总：[results/summary.csv](results/summary.csv)。avg 为 5 个 run 均值的平均，std 为这 5 个值的样本标准差（ddof=1），不表示逐图波动。当前时间沿用已完成的五次进程测量均值；每个版本的五个值及实际 timer 见 JSON，不重新测量。

`baseline` 为原始图；ANN 的 `fold_bn` 折叠 Conv-BN；SEW 的 `bn_lif` 融合四步 LIF、冻结 BN 仿射、mask 和残差；`sparse` 进一步裁掉全零中间通道，并仅计算活跃输出坐标上的卷积。稀疏点积改变浮点累加顺序，准确率可能略变。参数列统计推理图参数元素，不含 BN 缓存和 mask buffer；原始 SEW 网络均为 11,221,540 个参数。

## 测试

后续默认只计时：预热 20 次，CUDA Event 计时 5 个 run，每个 run 连续执行 100 次 BS=1 前向后同步，报告每次前向平均时间。不含 IO、预处理和 H2D；每次前向包含完整 T=4 读出及状态重置。分类指标复用当前 JSON。更换权重或需要重新计算分类指标时，加 `--evaluate`，只评估完整测试集一次。

```bash
pip install -r requirements-analysis.txt -r pruning/requirements.txt

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py \
  --data /path/to/soc/test \
  --uploaded-dir /path/to/uploaded_checkpoints
```

上传目录使用 `sew_resnet18_mstar_{dense,5e-6,1e-5,2e-5}/checkpoint_final.pth`。新增 checkpoint 暂留本机，后续单独打 zip。

融合实现见 [inference.py](inference.py)，参考 [spformers Triton LIF](https://github.com/putshua/spformers/blob/f5bfa1d9fd6e13c61802929c175397f57746ca39/spformers/neurons/_triton/triton_lif.py) 并适配 SpikingJelly 的 `v+=(x-v)/tau` 与 `>=` 阈值。SEW 训练/剪枝命令见 [pruning/README.md](pruning/README.md)。数据使用 [MSTAR SOC 镜像](https://github.com/omae11/MSTAR-dataset) 的 `d99007290d20ce52933c8bf0f8c3f461acd522ee` 版本。
