"""Confirmed shared classifier learning-rate schedule."""
import math


def steps_per_epoch(anchors):
    if type(anchors) is not int or anchors < 1:
        raise ValueError('anchor count must be positive')
    return (anchors + 127) // 128


def learning_rate(step, steps):
    if type(step) is not int or type(steps) is not int or steps < 1 or not 0 <= step <= 20 * steps:
        raise ValueError('invalid optimizer step')
    epoch = step / steps
    if epoch < 1:
        return 3e-4 * (0.1 + 0.9 * epoch)
    return 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + math.cos(math.pi * (epoch - 1) / 19))
