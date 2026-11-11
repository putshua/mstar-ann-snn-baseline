# MSTAR ANN / SEW 推理基线

MSTAR 十类 SOC：17° 训练 2747 张、15° 测试 2425 张；默认单通道 128×128，另列完整图像 192×192 实验。FP32、BS=1、SNN T=4。保留 ResNet-18 既有对照，并新增本机 ResNet-34 的神经元/性能结果。

## 最新普通 PyTorch 推理与 Params

以下五组统一为 `model.eval()` + `torch.no_grad()`、FP32、BS=1，不使用 CUDA Graph 或额外算子融合。计时前确认 GPU 3 无其他进程；训练使用 GPU 2，推理使用 GPU 3。预热 100 次，`time.perf_counter()` 前后 CUDA 同步，5 个 run，每个 run 100 次前向取平均；std 为五个均值的样本标准差。包含 SNN 完整 T=4 和状态重置，排除 IO、预处理及 H2D。网络 Params 不含 mask。

| 模型 | 输入 | λ | 网络 Params | Acc | 分类 mAP | GSOPs / dense GFLOPs | 推理 avg ± std（ms） |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ANN ResNet-34 | 128×128 | — | 21,283,530 | 98.9278% | 99.9529% | — / 2.340956 | 2.3194 ± 0.0074 |
| SEW-ResNet-34 | 128×128 | 1e-5 | 21,329,700 | 94.5567% | 98.8235% | 0.000927 / 9.287123 | 12.3320 ± 0.9228 |
| SEW-ResNet-34 | 192×192 | 1e-6 | 21,329,700 | 97.6495% | 99.7459% | 0.022000 / 20.895515 | 12.5079 ± 0.3945 |
| SEW-ResNet-34 | 192×192 | 3e-6 | 21,329,700 | 97.8557% | 99.5416% | 0.005286 / 20.895515 | 11.9863 ± 0.4223 |
| SEW-ResNet-18 | 192×192 | 6e-6 | 11,221,540 | 97.0309% | 99.6995% | 0.001355 / 10.023879 | 7.4056 ± 1.0692 |

GSOPs 是有效脉冲卷积操作估计（完整测试集、T=4，排除静态 stem 和分类器）；dense GFLOPs 按 Conv/Linear 每 MAC 两次浮点操作计算，SNN 包含完整 T=4、静态 stem 只执行一次，排除 BN/LIF/mask。当前 masked 实现仍执行完整密集卷积，因此不能把 GSOPs 当成实际 dense GPU FLOPs。

SEW-34 的 128、λ=1e-5 保留 12,078 / 1,220,608 个神经元，即自身 0.9895%。新 SEW-18 标准宽度、192 输入、λ=6e-6 保留 16,847 / 1,824,768 个神经元，即自身 0.9232%，有效 SOPs 为 0.0013551 G/图；SOC、seed=0、T=4，GPU 2 从头训练掩码 100 epochs，再固定掩码微调 20 epochs。

上表 SNN 时间已替换为与稀疏 kernel 同场进行的普通推理重测。其他历史模型保留其原计时方法，具体以每个模型 JSON 内的 timer 为准。

## 稀疏 kernel：同场普通推理对照

复用 `inference.py` 既有 active-output Triton 稀疏卷积、BN/LIF/mask/残差融合及全零中间通道裁剪。支持 SEW-18/34 和 128/192 输入。GPU 3 无其他进程，FP32、BS=1、T=4，`no_grad()`，无 CUDA Graph；普通和 kernel 各预热 100 次，5 个 run，每次 100 个前向，`perf_counter` 前后 CUDA 同步。输入是固定一张已在 GPU 的图片，时间不含文件读取、预处理或 H2D；Acc/mAP 另在完整 2425 张测试集上评估一次。

| 模型 / 输入 / λ | 普通时间 avg ± std（ms） | sparse 时间 avg ± std（ms） | 加速 | sparse Acc | sparse mAP | 执行图 Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SEW-34 / 128 / 1e-5 | 12.3320 ± 0.9228 | 2.3220 ± 0.6719 | 5.31× | 94.4742% | 98.7690% | 2,670,436 |
| SEW-34 / 192 / 1e-6 | 12.5079 ± 0.3945 | 2.8026 ± 0.4158 | 4.46× | 97.6082% | 99.7536% | 12,180,772 |
| SEW-34 / 192 / 3e-6 | 11.9863 ± 0.4223 | 2.5157 ± 0.0130 | 4.76× | 97.8557% | 99.5391% | 7,765,156 |
| SEW-18 / 192 / 6e-6 | 7.4056 ± 1.0692 | 1.5735 ± 0.0879 | 4.71× | 97.0309% | 99.6884% | 3,177,316 |

前两组 sparse Acc 分别降低 0.0825、0.0412 个百分点，后两组 Acc 不变；稀疏点积改变浮点累加顺序，少量脉冲阈值或 logits 排序可能变化。执行图 Params 不含 mask，因裁掉全零中间通道并将 BN 仿射移入 buffer，可少于原始网络 Params。逻辑神经元保留数量沿用同一个冻结掩码。优化行中的参考 FLOPs/SOPs 属于原始冻结模型，不代表稀疏 kernel 实际运算量。

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark_inference.py --data /path/to/soc/test \
  --sew-checkpoint /path/to/checkpoint.pt --sew-modes sparse \
  --evaluate-optimized --output /path/to/existing_results
```

首次没有 baseline 指标时用 `--evaluate`；已有 baseline、首次测 kernel 时用 `--evaluate-optimized`；两者已有指标后去掉评估选项，只计时五次。

## SEW-ResNet-34：完整图像 192×192

标准宽度，完整原图双线性缩放到 192×192（antialias），SOC、seed=0、T=4。训练统计量 mean=0.0297217593、std=0.0337017143。掩码训练 100 epochs，再固定二值掩码微调 20 epochs；最终权重、FP32、BS=1 评估。下表时间已替换为上述最新普通 PyTorch 五次计时。

| λ | Acc | 分类 mAP | 保留 LIF / 原始 LIF | 自身保留率 | 时间 avg ± std（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-6 | 97.6495% | 99.7459% | 77,460 / 2,746,368 | 2.8205% | 12.5079 ± 0.3945 |
| 3e-6 | 97.8557% | 99.5416% | 30,930 / 2,746,368 | 1.1262% | 11.9863 ± 0.4223 |

网络参数 21,329,700，另有 2,746,368 个训练 mask 参数。相同标准宽度的 128 输入有 1,220,608 个 LIF，192 输入增加为 2.25 倍；自身 1% 预算约为 27,464 个。λ=3e-6 自然收敛到 1.1262%，接近此目标，未强制裁剪；相对 λ=1e-6，保留神经元减少 60.07%，Acc 提高 0.2062 个百分点，分类 mAP 降低 0.2043 个百分点。

完整 dense Conv/Linear 计算为 20.8955 GFLOPs/图；λ=1e-6、3e-6 的有效 SOPs 分别约为 0.0220004、0.00529 GSOPs/图（完整测试集、T=4、BS=64，排除静态 stem 和分类器），精确值见结果 JSON。当前掩码仍执行完整卷积，SOPs 不代表实际 dense GPU 运算量。

这组使用完整图像，旧 128 输入使用中心裁剪，因此视野也发生变化。原图最大约 193×192，较小图像需要插值放大。

```bash
CUDA_VISIBLE_DEVICES=3 python pruning/main.py \
  --config pruning/configs/image_classification/sew_resnet18_mstar_prune.yaml \
  --model sew_resnet34 --input-size 192 --input-mode resize_full \
  --penalty-lmbda 3e-6 --seed 0 --data-path /path/to/soc \
  --output-dir outputs/sew34_192 --skip-mask-plots --no-save-latest --print-freq 0
```

## SEW-ResNet-18 宽度 ×4：神经元剪枝

四个 stage 通道为 256、512、1024、2048；SOC、seed=0、T=4，从头训练掩码 100 epochs，固定二值掩码微调 20 epochs。λ 自然决定保留数量，未强制设定神经元预算。

| 模型 | 网络 Params | Acc | 分类 mAP | 保留 LIF / 原始 LIF | 保留率 | BS=1 时间 avg ± std（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SEW-ResNet-34，λ=1e-5 | 21.330M | 94.5567% | 98.8235% | 12,078 / 1,220,608 | 0.9895% | 12.3320 ± 0.9228 |
| SEW-ResNet-18 ×4，λ=1e-5 | 178.776M | 92.9897% | 97.3621% | 8,668 / 3,244,032 | 0.2672% | 6.3975 ± 0.0886 |
| SEW-ResNet-18 ×4，λ=1e-6 | 178.776M | 97.8144% | 99.6565% | 69,374 / 3,244,032 | 2.1385% | 6.5752 ± 0.3876 |
| SEW-ResNet-18 ×4，λ=2e-6 | 178.776M | 97.5670% | 99.5488% | 37,211 / 3,244,032 | 1.1471% | 6.8758 ± 0.5835 |

×4 版本另有 3,244,032 个训练 mask 参数。λ=1e-5 时，8,668 个保留 LIF 相当于 ANN ResNet-34 激活数的 0.7451%；与 SEW-ResNet-34 的保留预算不同。λ 降到 1e-6 后 Acc 提高到 97.8144%，但自身保留率为 2.1385%，尚未达到自身约 1%（约 32,440 个）的目标。本表 SEW-34 为最新普通 PyTorch 计时，×4 版本保留此前计时，不跨口径判断速度。

λ=2e-6 自然收敛到自身 1.1471%，接近 1% 目标，未强制裁剪掩码。相对 λ=1e-6，保留神经元减少 46.36%，Acc 降低 0.2474 个百分点，分类 mAP 降低 0.1077 个百分点。

×4 版本完整 dense Conv/Linear 计算均为 70.9714 GFLOPs/图（T=4，2 FLOPs/MAC）。λ=1e-5、1e-6、2e-6 的有效脉冲卷积估计分别为 0.0007356、0.0280571、0.0100617 GSOPs/图，按完整测试集、T=4、BS=64 估计，排除静态 stem 和分类器。当前仍执行完整卷积加掩码，GSOPs 不代表实际 dense GPU 运算量。分类指标为最终权重、FP32、BS=1，计时预热 20 次，5 个 run，每次 100 个前向，包含状态重置。

```bash
CUDA_VISIBLE_DEVICES=3 python pruning/main.py \
  --config pruning/configs/image_classification/sew_resnet18_mstar_prune.yaml \
  --model sew_resnet18_w4 --penalty-lmbda 2e-6 --seed 0 \
  --data-path /path/to/soc --output-dir outputs/sew18_w4_lambda2e-6 \
  --skip-mask-plots --no-save-latest --print-freq 0
```

## ResNet-34：神经元与性能

本机 GPU 3（L20Z）从头训练，SOC、seed=0。ANN 训练 100 epochs；SEW 训练掩码 100 epochs，再固定二值掩码微调 20 epochs，T=4。分类使用最终权重、FP32、BS=1。ANN 和 λ=1e-5 时间已更新为最新普通 PyTorch 五次计时；λ=2e-5 保留此前 CUDA Event 计时，具体 timer 见 JSON。

| 模型 | 神经元/激活总数 | 保留数 | 保留率 | Accuracy | Macro-F1 | 推理时间 avg ± std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN ResNet-34 | 1,163,264 | 1,163,264 | 100.0000% | 98.9278% | 98.8939% | 2.3194 ± 0.0074 |
| SEW-ResNet-34，λ=2e-5 | 1,220,608 | 5,977 | 0.4897% | 85.8969% | 85.3944% | 11.2500 ± 0.3842 |
| SEW-ResNet-34，λ=1e-5 | 1,220,608 | 12,078 | 0.9895% | 94.5567% | 94.5637% | 12.3320 ± 0.9228 |

λ=1e-5 保留 12,078 / 1,220,608 个 LIF 单元，即 **0.9895%**，满足约 1% 的目标。ANN 统计 ReLU 空间激活位置，SEW 统计 LIF 空间单元，均不乘 batch 或 T。ANN 有 33 个激活位置，SEW 有 36 个 LIF 模块（包括下采样支路）。

ANN 网络参数 21,283,530；SEW 网络参数 21,329,700，另有 1,220,608 个 mask 参数。此处是原始 masked 推理，卷积张量尺寸未缩小，未对 ResNet-34 追加融合/稀疏性能实验。保留每组五次均值及 std，未统一重测的 λ=2e-5 不参与最新速度比较。

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

后续默认只计时：普通 PyTorch `eval()` + `no_grad()`，预热 100 次，`time.perf_counter()` 前后 CUDA 同步，计时 5 个 run，每个 run 连续执行 100 次 BS=1 前向，报告每次前向平均时间。不含 IO、预处理和 H2D；每次前向包含完整 T=4 读出及状态重置。分类指标复用当前 JSON。更换权重或需要重新计算分类指标时，加 `--evaluate`，只评估完整测试集一次。训练与推理使用不同 GPU，推理前确认卡无其他计算进程。

```bash
pip install -r requirements-analysis.txt -r pruning/requirements.txt

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py \
  --data /path/to/soc/test \
  --uploaded-dir /path/to/uploaded_checkpoints
```

上传目录使用 `sew_resnet18_mstar_{dense,5e-6,1e-5,2e-5}/checkpoint_final.pth`。新增 checkpoint 暂留本机，后续单独打 zip。

融合实现见 [inference.py](inference.py)，参考 [spformers Triton LIF](https://github.com/putshua/spformers/blob/f5bfa1d9fd6e13c61802929c175397f57746ca39/spformers/neurons/_triton/triton_lif.py) 并适配 SpikingJelly 的 `v+=(x-v)/tau` 与 `>=` 阈值。SEW 训练/剪枝命令见 [pruning/README.md](pruning/README.md)。数据使用 [MSTAR SOC 镜像](https://github.com/omae11/MSTAR-dataset) 的 `d99007290d20ce52933c8bf0f8c3f461acd522ee` 版本。
