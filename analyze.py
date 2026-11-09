"""Re-evaluate both published checkpoints and generate the README metrics report."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torchvision import datasets

from mstar_data import transforms_for
from train import build_model, loader, reset_model_state

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {'ann': 'ann_resnet18.pt', 'snn': 'snn_resnet18_lif_t4.pt'}
NAMES = {'ann': 'ANN ResNet-18', 'snn': 'SNN ResNet-18 (LIF, T=4)'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@torch.inference_mode()
def collect_scores(kind, data, output, device, batch_size):
    checkpoint = ROOT / 'checkpoints' / CHECKPOINTS[kind]
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    ds = datasets.ImageFolder(data, transforms_for(cfg['mean'], cfg['std']))
    if ds.classes != cfg['classes'] or len(ds) != 2425:
        raise ValueError('Expected the published 2425-image, ten-class SOC test split')
    model = build_model(cfg['arch'], False, cfg['stem'], cfg.get('time_steps', 4),
                        cfg.get('lif_tau', 2.0)).to(device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    logits, labels = [], []
    start = time.perf_counter()
    for index, (x, y) in enumerate(loader(ds, batch_size, 0, cuda=device == 'cuda')):
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
        reset_model_state(model)
        if index % 10 == 0:
            print(f'{kind}: {sum(map(len, labels))}/{len(ds)} images', flush=True)
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    probabilities = torch.from_numpy(logits).double().softmax(1).numpy()
    paths = np.asarray([str(Path(p).relative_to(data)) for p, _ in ds.samples])
    fingerprint = hashlib.sha256()
    for (p, _), relative in zip(ds.samples, paths):
        fingerprint.update(relative.encode() + b'\0' + hashlib.sha256(Path(p).read_bytes()).digest())
    metadata = {'checkpoint': f'checkpoints/{CHECKPOINTS[kind]}',
                'checkpoint_sha256': sha256(checkpoint), 'epoch': ckpt['epoch'],
                'dataset_source': cfg.get('dataset_source'),
                'test_fingerprint_sha256': fingerprint.hexdigest(),
                'mean': cfg['mean'], 'std': cfg['std'], 'device': device,
                'evaluation_host': platform.node(),
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'inference_dtype': 'float32; softmax computed in float64',
                'batch_size': batch_size, 'inference_seconds': time.perf_counter() - start}
    np.savez_compressed(output / f'{kind}_scores.npz', logits=logits,
                        probabilities=probabilities, labels=labels, paths=paths,
                        classes=np.asarray(ds.classes), metadata=json.dumps(metadata))


def calculate(kind, scores):
    y, p = scores['labels'], scores['probabilities']
    classes = scores['classes'].tolist()
    pred = p.argmax(1)
    onehot = np.eye(len(classes), dtype=int)[y]
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probabilities')
    cm = metrics.confusion_matrix(y, pred, labels=np.arange(len(classes)))
    original = json.loads((ROOT / 'results' / f'{kind}.json').read_text())
    if not np.array_equal(cm, original['test_metrics']['confusion_matrix']):
        raise ValueError(f'{kind}: reloaded confusion matrix differs from published results')
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y, pred, labels=np.arange(len(classes)), zero_division=0)
    aucs = np.asarray([metrics.roc_auc_score(onehot[:, i], p[:, i]) for i in range(len(classes))])
    aps = metrics.average_precision_score(onehot, p, average=None)
    confidence = p.max(1)
    correct = pred == y
    # Fifteen equal-width bins, [0, 1/15], (1/15, 2/15], ..., (14/15, 1].
    bin_id = np.minimum(np.ceil(confidence * 15).astype(int), 15) - 1
    bin_id = np.maximum(bin_id, 0)
    ece = sum(np.mean(bin_id == b) * abs(correct[bin_id == b].mean() - confidence[bin_id == b].mean())
              for b in range(15) if np.any(bin_id == b))
    tp = cm.diagonal()
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = len(y) - tp - fp - fn
    summary = {
        'accuracy': metrics.accuracy_score(y, pred),
        'top3_accuracy': metrics.top_k_accuracy_score(y, p, k=3, labels=np.arange(len(classes))),
        'balanced_accuracy': metrics.balanced_accuracy_score(y, pred),
        'macro_precision': precision.mean(), 'macro_recall': recall.mean(), 'macro_f1': f1.mean(),
        'weighted_f1': np.average(f1, weights=support),
        'roc_auc_macro_ovr': aucs.mean(),
        'roc_auc_micro_ovr': metrics.roc_auc_score(onehot.ravel(), p.ravel()),
        'map_macro': aps.mean(), 'ap_micro': metrics.average_precision_score(onehot, p, average='micro'),
        'cohen_kappa': metrics.cohen_kappa_score(y, pred), 'mcc': metrics.matthews_corrcoef(y, pred),
        'log_loss': metrics.log_loss(y, p, labels=np.arange(len(classes))),
        'brier_multiclass': np.mean(np.sum((p - onehot) ** 2, axis=1)), 'ece_15_bins': float(ece),
    }
    rows = [{'class': name, 'support': int(support[i]), 'precision': float(precision[i]),
             'recall': float(recall[i]), 'f1': float(f1[i]), 'specificity': float(tn[i] / (tn[i] + fp[i])),
             'roc_auc_ovr': float(aucs[i]), 'ap_ovr': float(aps[i])} for i, name in enumerate(classes)]
    errors = [(int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes))
              for j in range(len(classes)) if i != j and cm[i, j]]
    errors.sort(key=lambda x: (-x[0], x[1], x[2]))
    return {'model': NAMES[kind], 'n': len(y), 'correct': int(correct.sum()), 'errors': int((~correct).sum()),
            'classes': classes, 'summary': {k: float(v) for k, v in summary.items()},
            'per_class': rows, 'confusion_matrix': cm.tolist(),
            'largest_confusions': [{'true': a, 'predicted': b, 'count': n} for n, a, b in errors[:5]],
            'matches_published_confusion_matrix': True,
            'provenance': json.loads(str(scores['metadata']))}


def save_figure(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    svg = output / f'{name}.svg'
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def plot_confusion(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
    for col, kind in enumerate(CHECKPOINTS):
        cm = np.asarray(results[kind]['confusion_matrix'])
        for row in range(2):
            values = cm if row == 0 else cm / cm.sum(1, keepdims=True)
            ax = axes[row, col]
            im = ax.imshow(values, cmap='Blues', vmin=0, vmax=274 if row == 0 else 1)
            for (i, j), value in np.ndenumerate(values):
                text = str(cm[i, j]) if row == 0 else f'{value:.1%}'
                ax.text(j, i, text, ha='center', va='center', fontsize=7.5,
                        color='white' if value > (137 if row == 0 else .5) else '#24344a')
            ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel='Predicted class', ylabel='True class',
                   title=NAMES[kind] + (' | count' if row == 0 else ' | row normalized'))
            ax.set_xticklabels(results[kind]['classes'], rotation=45, ha='right')
            ax.set_yticklabels(results[kind]['classes'])
            fig.colorbar(im, ax=ax, shrink=.75)
    save_figure(fig, output, 'confusion_matrices')


def plot_curves(results, scores, output):
    colors = plt.get_cmap('tab10').colors
    roc_fig, roc_axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    pr_fig, pr_axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    for col, kind in enumerate(CHECKPOINTS):
        p, y = scores[kind]['probabilities'], scores[kind]['labels']
        onehot = np.eye(10, dtype=int)[y]
        for i, row in enumerate(results[kind]['per_class']):
            fpr, tpr, _ = metrics.roc_curve(onehot[:, i], p[:, i], drop_intermediate=False)
            # Independent consistency check against the rank-based AUC metric.
            assert abs(metrics.auc(fpr, tpr) - row['roc_auc_ovr']) < 1e-12
            for ax in roc_axes[:, col]:
                ax.plot(fpr, tpr, color=colors[i], lw=1.3,
                        label=f"{row['class']} ({row['roc_auc_ovr']:.5f})")
            precision, recall, _ = metrics.precision_recall_curve(onehot[:, i], p[:, i])
            for ax in pr_axes[:, col]:
                ax.step(recall, precision, where='post', color=colors[i], lw=1.3,
                        label=f"{row['class']} ({row['ap_ovr']:.5f})")
        fpr, tpr, _ = metrics.roc_curve(onehot.ravel(), p.ravel())
        for ax in roc_axes[:, col]:
            ax.plot(fpr, tpr, color='black', lw=1.8, ls='--',
                    label=f"micro ({results[kind]['summary']['roc_auc_micro_ovr']:.5f})")
            ax.set(xlabel='False positive rate', ylabel='True positive rate')
            ax.grid(alpha=.2)
        roc_axes[0, col].plot([0, 1], [0, 1], ls=':', color='gray', lw=1)
        roc_axes[0, col].set(title=NAMES[kind] + ' | ROC (OvR)', xlim=(0, 1), ylim=(0, 1.005))
        roc_axes[0, col].legend(loc='lower right', fontsize=8, title='Class (AUC)')
        roc_axes[1, col].set(title='Low false positive region (zoom)', xlim=(0, .03), ylim=(.90, 1.001))
        precision, recall, _ = metrics.precision_recall_curve(onehot.ravel(), p.ravel())
        for ax in pr_axes[:, col]:
            ax.step(recall, precision, where='post', color='black', lw=1.8, ls='--',
                    label=f"micro ({results[kind]['summary']['ap_micro']:.5f})")
            ax.set(xlabel='Recall', ylabel='Precision')
            ax.grid(alpha=.2)
        pr_axes[0, col].set(title=NAMES[kind] + ' | Precision–recall (OvR)',
                            xlim=(0, 1), ylim=(0, 1.005))
        pr_axes[0, col].legend(loc='lower left', fontsize=8, title='Class (AP)')
        pr_axes[1, col].set(title='High recall and precision region (zoom)',
                            xlim=(.9, 1), ylim=(.9, 1.001))
    save_figure(roc_fig, output, 'roc_curves')
    save_figure(pr_fig, output, 'precision_recall_curves')


def markdown(results):
    rows = [
        ('Accuracy ↑', 'accuracy', True), ('Top-3 accuracy ↑', 'top3_accuracy', True),
        ('Balanced accuracy / Macro recall ↑', 'balanced_accuracy', True),
        ('Macro precision ↑', 'macro_precision', True), ('Macro F1 ↑', 'macro_f1', True),
        ('Weighted F1 ↑', 'weighted_f1', True), ('ROC-AUC macro (OvR) ↑', 'roc_auc_macro_ovr', False),
        ('ROC-AUC micro (OvR) ↑', 'roc_auc_micro_ovr', False), ('mAP / Macro AP (OvR) ↑', 'map_macro', False),
        ('Micro AP (OvR) ↑', 'ap_micro', False), ("Cohen's κ ↑", 'cohen_kappa', False),
        ('MCC ↑', 'mcc', False), ('Log loss ↓', 'log_loss', False),
        ('Multiclass Brier score ↓', 'brier_multiclass', False), ('ECE (15 bins) ↓', 'ece_15_bins', True)]
    text = ['## 完整测试指标', '',
            '使用已发布的第 100 epoch 权重，重新推理全部 **2425 张 SOC 测试图片**。灰度、128×128 中心裁剪、训练集均值/标准差与原实验一致；无测试增强，不重新训练或选择权重。两组预测的混淆矩阵均与原实验逐项一致。', '',
            f"本次评估通过 keeper 在单张 **{results['ann']['provenance']['gpu']}** 上运行；节点和软件版本记录于 [metrics.json](results/analysis/metrics.json)。", '',
            '| 指标 | ANN ResNet-18 | SNN ResNet-18，LIF T=4 |', '| --- | ---: | ---: |']
    for label, key, percent in rows:
        values = [results[k]['summary'][key] for k in CHECKPOINTS]
        formatted = [f'{100*v:.4f}%' if percent else f'{v:.6f}' for v in values]
        text.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    text.extend([f"| 错误数 | {results['ann']['errors']} / 2425 | {results['snn']['errors']} / 2425 |", '',
                 '### 指标口径', '',
                 '- 每张图片的十类分数为 FP32 推理 logits 的 float64 softmax。SNN 先对 4 个时间步的 logits 求均值，再做 softmax；每个 batch 重置膜电位。',
                 '- ROC 与 PR 使用 **one-vs-rest (OvR)**：每类分别作为正类，其余九类作为负类，遍历连续分数阈值。macro AUC 为十个类别 AUC 的算术平均；micro 将 2425×10 个标签/分数展开计算。',
                 '- **mAP 是图像分类的 macro AP**：按 [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) 计算各类 AP，再等权平均。AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ，使用召回增量加权，未采用梯形积分。此处不涉及目标检测的 IoU、mAP@0.5 或 mAP@0.5:0.95。',
                 '- Macro 对十类等权；Weighted F1 按各类测试样本数加权。单标签十分类中 micro precision、micro recall、micro F1 均等于 accuracy。',
                 '- Specificity = TN/(TN+FP)，按 OvR 定义；κ 与 MCC 衡量离散预测的一致性。Log loss 使用自然对数。Brier = meanᵢ Σ꜀ (pᵢ꜀−yᵢ꜀)²，范围 [0, 2]，不除以类别数或 2。',
                 '- ECE 使用 top-1 置信度的 15 个等宽分箱，对各箱 |准确率−平均置信度| 按样本比例加权。所有概率均未做温度缩放或测试集校准。',
                 '- 本表来自单次 seed=0 实验，未估计多次训练的方差。', '',
                 '### 混淆矩阵', '',
                 '纵轴为真实类别，横轴为预测类别；上排为计数，下排按真实类别归一化，每行和为 1。', '',
                 '![ANN 与 SNN 的混淆矩阵](results/analysis/confusion_matrices.png)', '',
                 '### ROC 曲线', '',
                 '每种模型给出十类 OvR ROC 与 micro ROC；下排放大 FPR ≤ 0.03、TPR ≥ 0.90 区域。图例括号内为 AUC。', '',
                 '![ANN 与 SNN 的 ROC 曲线](results/analysis/roc_curves.png)', '',
                 '### Precision–Recall 曲线', '',
                 '图例括号内为 AP；mAP 是十类 AP 的平均值。下排放大 Recall ≥ 0.90、Precision ≥ 0.90 区域。', '',
                 '![ANN 与 SNN 的 PR 曲线](results/analysis/precision_recall_curves.png)', '',
                 '### 各类别指标', '', 'Precision、Recall、F1、Specificity 均以百分比展示；AUC 和 AP 范围为 [0, 1]。', ''])
    for kind in CHECKPOINTS:
        text.extend([f'#### {NAMES[kind]}', '',
                     '| 类别 | 测试数 | Precision | Recall | F1 | Specificity | ROC-AUC | AP |',
                     '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in results[kind]['per_class']:
            vals = [f"{r[k]*100:.4f}%" for k in ('precision', 'recall', 'f1', 'specificity')]
            text.append(f"| {r['class']} | {r['support']} | " + ' | '.join(vals) + f" | {r['roc_auc_ovr']:.6f} | {r['ap_ovr']:.6f} |")
        text.extend(['', '主要混淆（真实 → 预测）：' + '；'.join(
            f"{r['true']} → {r['predicted']}：{r['count']} 张" for r in results[kind]['largest_confusions']) + '。', ''])
    delta = (results['ann']['summary']['accuracy'] - results['snn']['summary']['accuracy']) * 100
    text.extend([f"ANN 的 accuracy 比 SNN 高 **{delta:.4f} 个百分点**，少错 {results['snn']['errors']-results['ann']['errors']} 张。ROC-AUC、AP 衡量类别分数的排序能力，argmax 后的准确率和 F1 衡量最终分类结果，需结合阅读。", '',
                 '### 数据与复现', '',
                 '- [完整指标与溯源信息](results/analysis/metrics.json)；[逐类别 CSV](results/analysis/per_class.csv)。',
                 '- 原始预测分数：[ANN](results/analysis/ann_scores.npz)、[SNN](results/analysis/snn_scores.npz)。包含相对图片路径、真实标签、类别顺序、logits、softmax 概率和权重 SHA-256；使用 `numpy.load(..., allow_pickle=False)` 读取。',
                 '- 矢量图：[混淆矩阵](results/analysis/confusion_matrices.svg)、[ROC](results/analysis/roc_curves.svg)、[PR](results/analysis/precision_recall_curves.svg)。', '',
                 '```bash', 'pip install -r requirements-analysis.txt',
                 'python analyze.py --data /path/to/soc/test --device auto', '',
                 '# 直接使用已发布的分数重新生成指标和图表，无需推理',
                 'python analyze.py --reuse-scores', '```', ''])
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'analysis')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--reuse-scores', action='store_true')
    args = parser.parse_args()
    if not args.reuse_scores and args.data is None:
        parser.error('--data is required unless --reuse-scores is supplied')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    args.output.mkdir(parents=True, exist_ok=True)
    results, scores = {}, {}
    for kind in CHECKPOINTS:
        if not args.reuse_scores:
            collect_scores(kind, args.data, args.output, device, args.batch_size)
        with np.load(args.output / f'{kind}_scores.npz', allow_pickle=False) as saved:
            scores[kind] = {key: saved[key] for key in saved.files}
        results[kind] = calculate(kind, scores[kind])
        print(kind, json.dumps(results[kind]['summary']), flush=True)
    for key in ('labels', 'paths', 'classes'):
        assert np.array_equal(scores['ann'][key], scores['snn'][key]), key
    assert results['ann']['provenance']['test_fingerprint_sha256'] == results['snn']['provenance']['test_fingerprint_sha256']
    environment = {key: importlib.metadata.version(key) for key in
                   ('torch', 'torchvision', 'spikingjelly', 'numpy', 'scikit-learn', 'matplotlib')}
    environment['python'] = platform.python_version()
    save_json(args.output / 'metrics.json', {'environment': environment, 'models': results})
    with (args.output / 'per_class.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['model', *results['ann']['per_class'][0]], lineterminator='\n')
        writer.writeheader()
        for kind in CHECKPOINTS:
            for row in results[kind]['per_class']:
                writer.writerow({'model': kind, **row})
    plot_confusion(results, args.output)
    plot_curves(results, scores, args.output)
    report = markdown(results)
    # The standalone Markdown and README use the same generated content.
    (args.output / 'report.md').write_text(report.replace('(results/analysis/', '('))
    readme = ROOT / 'README.md'
    current = readme.read_text().split('<!-- BEGIN GENERATED METRICS -->')[0].rstrip()
    readme.write_text(current + '\n\n<!-- BEGIN GENERATED METRICS -->\n' + report + '<!-- END GENERATED METRICS -->\n')
    print(f'Report written to {args.output}', flush=True)


if __name__ == '__main__':
    main()
