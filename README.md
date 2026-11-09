# MSTAR ANN / SNN ResNet-18

MSTAR 十类 SOC：17° 训练 2747 张，15° 测试 2425 张。
输入为单通道 128×128 中心裁剪；seed=0，随机初始化，100 epochs。

| 模型 | Accuracy | Macro-F1 | 错误数 |
| --- | ---: | ---: | ---: |
| ANN ResNet-18 | 98.9278% | 98.9044% | 26 / 2425 |
| SNN ResNet-18，LIF，T=4 | 97.4021% | 97.3755% | 63 / 2425 |

两者均为 11,175,370 个可训练参数。SNN 含 17 个 LIF 层，128×128 输入下共 753,664 个 LIF 单元；tau=2，threshold=1，ATan 代理梯度。同一静态图输入 4 步，对 logits 取时间平均，每个 batch 重置膜电位。

训练配方：AdamW，lr=0.001，weight_decay=0.0001，batch=64，cosine 调度，AMP，反射填充后的轻微平移增强；固定使用最后一轮评估测试集。单张 NVIDIA L20Z 上运行，PyTorch 2.9.0+cu130 / torchvision 0.24.0+cu130 / SpikingJelly 0.0.0.0.14。

## 文件

- `train.py`、`snn.py`、`mstar_data.py`、`evaluate.py`：训练、模型、预处理及评估代码。
- `checkpoints/ann_resnet18.pt`、`checkpoints/snn_resnet18_lif_t4.pt`：两份训练权重。
- `results/ann.json`、`results/snn.json`：实验配置、测试指标及混淆矩阵。

## 运行

安装与本机 CUDA 匹配的 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
```

使用 [MSTAR SOC 镜像](https://github.com/omae11/MSTAR-dataset) 的 `soc/` 目录，结构为 `soc/{train,test}/{class}/*.jpg`。本次使用版本 `d99007290d20ce52933c8bf0f8c3f461acd522ee`。

```bash
# ANN
CUDA_VISIBLE_DEVICES=0 python train.py --data /path/to/soc \
  --arch resnet18 --seed 0 --output outputs/ann

# SNN：LIF，4 个时间步
CUDA_VISIBLE_DEVICES=0 python train.py --data /path/to/soc \
  --arch spiking_resnet18 --time-steps 4 --lif-tau 2 \
  --seed 0 --output outputs/snn

# 加载权重评估；ANN 同理，更换 checkpoint 路径即可
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoints/snn_resnet18_lif_t4.pt \
  --data /path/to/soc/test --output outputs/snn_eval
```

输出目录须不存在。`SOC` 的逐类样本数对应 [ASIR-Net 论文表 2](https://doi.org/10.3390/rs13153029)；本项目使用 JPEG 镜像及自己的预处理，不声称逐文件复现该论文。
