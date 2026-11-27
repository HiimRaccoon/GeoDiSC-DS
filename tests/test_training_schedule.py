import pytest
from geobench.training import learning_rate, steps_per_epoch


def test_step_schedule_and_partial_batch():
    assert steps_per_epoch(129) == 2
    assert learning_rate(0, 2) == pytest.approx(3e-5)
    assert learning_rate(2, 2) == pytest.approx(3e-4)
    assert learning_rate(40, 2) == pytest.approx(1e-6)


def test_invalid_schedule_inputs_rejected():
    with pytest.raises(ValueError): steps_per_epoch(0)
    with pytest.raises(ValueError): learning_rate(41, 2)
