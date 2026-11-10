"""Classification metrics and figure output shared by benchmark.py."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

NAMES = {'ann': 'ANN ResNet-18', 'snn': 'SNN ResNet-18 (LIF, T=4)'}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def calculate(kind, scores):
    y, p = scores['labels'], scores['probabilities']
    classes = scores['classes'].tolist()
    pred = p.argmax(1)
    onehot = np.eye(len(classes), dtype=int)[y]
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probabilities')
    cm = metrics.confusion_matrix(y, pred, labels=np.arange(len(classes)))
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
    return {'model': NAMES.get(kind, kind), 'n': len(y), 'correct': int(correct.sum()), 'errors': int((~correct).sum()),
            'classes': classes, 'summary': {k: float(v) for k, v in summary.items()},
            'per_class': rows, 'confusion_matrix': cm.tolist(),
            'largest_confusions': [{'true': a, 'predicted': b, 'count': n} for n, a, b in errors[:5]],
            'provenance': json.loads(str(scores['metadata']))}


def save_figure(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    svg = output / f'{name}.svg'
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
