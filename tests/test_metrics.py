import pytest
import torch
from geobench.metrics import rank_logits, compute_metrics, aggregate_seeds, format_aggregate


def logits_for(labels):
    x = torch.zeros((len(labels), 124))
    for i, label in enumerate(labels):
        x[i, label] = 1
    return x


def test_ties_rank_by_class_id():
    assert rank_logits(torch.zeros((1, 124)))[0, :5].tolist() == [0, 1, 2, 3, 4]


def test_support_and_full_label_metrics():
    result = compute_metrics([0, 0, 1], logits_for([0, 2, 1]), {'head': [0], 'medium': [1], 'extreme-tail': [2]})
    assert result['macro_f1'] == pytest.approx(5 / 6)
    assert result['balanced_accuracy'] == pytest.approx(3 / 4)
    assert result['weighted_f1'] == pytest.approx(7 / 9)
    assert result['top1_accuracy'] == pytest.approx(2 / 3)
    assert result['per_class'][2]['f1'] is None
    assert result['per_class'][2]['fp'] == 1
    assert result['groups']['extreme-tail']['macro_f1'] is None


def test_group_counts_include_other_groups():
    result = compute_metrics([0, 1], logits_for([0, 0]), {'head': [0]})
    assert result['groups']['head']['macro_f1'] == pytest.approx(2 / 3)


def test_aggregate_requires_exact_seeds():
    records = [{'seed': 42, 'value': .1}, {'seed': 123, 'value': .2}, {'seed': 2026, 'value': .3}]
    value = aggregate_seeds(records)
    assert value['mean'] == pytest.approx(.2)
    assert value['std'] == pytest.approx(.1)
    assert format_aggregate(value) == '20.00 ± 10.00'
    with pytest.raises(ValueError):
        aggregate_seeds(records[:2])


def test_top5_ties_and_invalid_logits():
    scores = torch.zeros((1, 124))
    scores[0, :4] = 1
    assert rank_logits(scores)[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert compute_metrics([5], scores, {})['top5_accuracy'] == 0
    scores[0, 0] = float('nan')
    with pytest.raises(ValueError):
        compute_metrics([5], scores, {})


def test_unavailable_seed_values_are_not_missing_values():
    unavailable = [{'seed': seed, 'value': None} for seed in (42, 123, 2026)]
    assert aggregate_seeds(unavailable) == {'mean': None, 'std': None}
    unavailable[0]['value'] = .2
    with pytest.raises(ValueError):
        aggregate_seeds(unavailable)


def test_single_value_presentation():
    from geobench.metrics import format_percentage
    assert format_percentage(.123456) == '12.35'
    assert format_percentage(None) == 'N/A'


def test_invalid_labels_do_not_silently_truncate():
    with pytest.raises(ValueError):
        compute_metrics([0.5], logits_for([0]), {})
