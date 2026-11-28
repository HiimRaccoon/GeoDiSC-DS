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


def test_initializations_are_frozen_per_seed_with_runtime_dependencies(tmp_path, monkeypatch):
    import json
    import torchvision.models
    from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock
    from geobench.classifier import create_initializations
    monkeypatch.setattr(torchvision.models, 'efficientnet_b0', lambda weights=None: TinyNet())
    weights=tmp_path/'models/efficientnet'; token=claim_writer(weights,'classifier-weights'); stage=staging_path(weights,token)
    torch.save(TinyNet().state_dict(),stage/'weights.pt')
    model_lock=publish_lock(weights,{'identity':'classifier-weights','files':['weights.pt'],'dependencies':{},'claim':token})
    runtime=tmp_path/'runtime';token=claim_writer(runtime,'shared-runtime');stage=staging_path(runtime,token)
    (stage/'runtime.json').write_text(json.dumps({'weights':'models/efficientnet/weights.pt','weights_sha256':sha256_file(weights/'weights.pt')}))
    runtime_lock=publish_lock(runtime,{'identity':'shared-runtime','files':['runtime.json'],'dependencies':{'models/efficientnet/lock.json':sha256_file(model_lock)},'artifact_root':tmp_path,'claim':token})
    index=create_initializations(tmp_path,owner_token='fixture')
    assert len(verify_lock(index,boundary=tmp_path)['dependencies']) == 3
    for seed in (42,123,2026):
        path=tmp_path/f'initializations/seed_{seed}/lock.json'
        lock=verify_lock(path,boundary=tmp_path)
        assert lock['identity']==f'initialization:{seed}'
        assert lock['dependencies']['runtime/lock.json']==sha256_file(runtime_lock)
    assert create_initializations(tmp_path,owner_token='fixture')==index


def test_initialization_cli_requires_persistent_owner(tmp_path,monkeypatch):
    from geobench.cli import main
    monkeypatch.delenv('GEOBENCH_OWNER_TOKEN',raising=False)
    with pytest.raises(ValueError,match='owner token'):
        main(['initialize','--root',str(tmp_path)])
