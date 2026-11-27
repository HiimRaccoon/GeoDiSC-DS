import torch
from torch import nn
import pytest
from geobench.classifier import build_classifier


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 2)
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(2, 1000))
    def forward(self,x): return self.classifier(self.features(x))


def test_classifier_uses_verified_weights_and_124_outputs(tmp_path, monkeypatch):
    import torchvision.models
    from geobench.artifacts import sha256_file
    monkeypatch.setattr(torchvision.models, 'efficientnet_b0', lambda weights=None: TinyNet())
    path=tmp_path/'weights.pt'; torch.save(TinyNet().state_dict(),path)
    model=build_classifier(path,sha256_file(path),seed=42)
    assert model(torch.zeros(1,2)).shape==(1,124)
    other=build_classifier(path,sha256_file(path),seed=42)
    assert torch.equal(model.classifier[1].weight,other.classifier[1].weight)
    assert all(p.requires_grad for p in model.parameters())


def test_classifier_rejects_mismatched_weights(tmp_path):
    path=tmp_path/'weights'; path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash'):
        build_classifier(path,'0'*64,seed=42)
