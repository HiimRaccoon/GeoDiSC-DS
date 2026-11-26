"""Support-aware metrics over the complete classifier label space."""
import math
import statistics

import torch


def rank_logits(logits):
    logits = torch.as_tensor(logits)
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('logits must be a finite two-dimensional tensor')
    return torch.argsort(logits, dim=1, descending=True, stable=True)


def compute_metrics(y_true, logits, groups):
    scores = torch.as_tensor(logits)
    ranking = rank_logits(scores)
    labels = torch.as_tensor(y_true, device=scores.device)
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError('labels must be integer class IDs')
    labels = labels.to(dtype=torch.long)
    if labels.ndim != 1 or len(labels) != len(scores) or not len(labels) or scores.shape[1] != 124:
        raise ValueError('expected nonempty labels and [N,124] logits')
    if (labels < 0).any() or (labels >= 124).any():
        raise ValueError('label outside classifier space')
    matrix = torch.bincount(labels * 124 + ranking[:, 0], minlength=124 * 124).reshape(124, 124)
    classes = []
    for c in range(124):
        tp = int(matrix[c, c]); support = int(matrix[c].sum())
        fp = int(matrix[:, c].sum()) - tp; fn = support - tp
        classes.append({'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
                        'evaluable': support > 0,
                        'f1': 2 * tp / (2 * tp + fp + fn) if support else None,
                        'recall': tp / support if support else None})
    supported = [c for c in classes if c['evaluable']]
    n = len(labels)
    group_results = {}
    for name, ids in groups.items():
        if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 or i >= 124 for i in ids):
            raise ValueError('invalid group class IDs')
        subset = [classes[i] for i in ids if classes[i]['evaluable']]
        group_results[name] = {'num_evaluable_classes': len(subset), 'support': sum(c['support'] for c in subset),
                               'macro_f1': statistics.mean(c['f1'] for c in subset) if subset else None,
                               'macro_recall': statistics.mean(c['recall'] for c in subset) if subset else None}
    return {'confusion_matrix': matrix.tolist(), 'per_class': classes, 'groups': group_results,
            'num_evaluable_classes': len(supported), 'support': n,
            'macro_f1': statistics.mean(c['f1'] for c in supported),
            'balanced_accuracy': statistics.mean(c['recall'] for c in supported),
            'weighted_f1': sum(c['support'] * c['f1'] for c in supported) / n,
            'top1_accuracy': int(matrix.diag().sum()) / n,
            'top5_accuracy': float((ranking[:, :5] == labels[:, None]).any(dim=1).sum()) / n}


def aggregate_seeds(records):
    if len(records) != 3 or {r['seed'] for r in records} != {42, 123, 2026}:
        raise ValueError('exactly the three required seeds are required')
    values = [r['value'] for r in records]
    if all(v is None for v in values):
        return {'mean': None, 'std': None}
    if any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('incomplete or invalid seed values')
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values)}


def format_aggregate(aggregate):
    if aggregate['mean'] is None:
        return 'N/A'
    return f"{aggregate['mean'] * 100:.2f} ± {aggregate['std'] * 100:.2f}"


def format_percentage(value):
    if value is None:
        return 'N/A'
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid metric')
    return f'{value * 100:.2f}'
