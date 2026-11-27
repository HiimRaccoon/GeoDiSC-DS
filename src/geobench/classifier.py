"""Shared EfficientNet-B0 construction from verified local weights."""
import torch
from torch import nn
from .artifacts import sha256_file


def build_classifier(weights_path, expected_sha256, *, seed):
    if type(seed) is not int or seed not in (42, 123, 2026):
        raise ValueError('unsupported training seed')
    if sha256_file(weights_path) != expected_sha256:
        raise ValueError('classifier weights hash mismatch')
    import torchvision.models
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torchvision.models.efficientnet_b0(weights=None)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=.2), nn.Linear(in_features, 124))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model.float()
