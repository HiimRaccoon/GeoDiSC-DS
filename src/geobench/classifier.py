"""Shared EfficientNet-B0 construction from verified local weights."""
import torch


def probe_gpu(root):
    import os
    from .probes import run_probe
    return run_probe(root,'classifier',owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN'))
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


def create_initializations(root, *, owner_token):
    """Freeze one shared initial state per official seed from a verified runtime."""
    import hashlib
    import io
    import json
    from pathlib import Path
    from .artifacts import _inside, active_writer, claim_writer, staging_path, publish_lock, verify_lock, complete_temporary
    root = Path(root).resolve()
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    runtime_path = root/'runtime/lock.json'
    runtime = verify_lock(runtime_path,boundary=root)
    if runtime['identity'] != 'shared-runtime' or 'runtime.json' not in runtime['files']:
        raise ValueError('invalid shared runtime lock')
    metadata = json.loads((runtime_path.parent/'runtime.json').read_text(encoding='utf-8'))
    model_lock_path = root/'models/efficientnet/lock.json'
    model_lock = verify_lock(model_lock_path,boundary=root)
    if model_lock['identity'] != 'classifier-weights' or runtime['dependencies'].get('models/efficientnet/lock.json') != sha256_file(model_lock_path):
        raise ValueError('runtime classifier model dependency mismatch')
    weight_path = _inside(root,metadata['weights'])
    if (not weight_path.is_relative_to(model_lock_path.parent)
            or model_lock['files'].get(weight_path.relative_to(model_lock_path.parent).as_posix()) != metadata['weights_sha256']):
        raise ValueError('runtime weights are not covered by model lock')
    paths = []
    with active_writer(root):
        for seed in (42,123,2026):
            target = root/'initializations'/f'seed_{seed}'
            identity = f'initialization:{seed}'
            token_value = hashlib.sha256((identity+'\0'+owner_token).encode()).hexdigest()
            token = claim_writer(target,identity,token=token_value,resume=(target/'.writer').exists())
            try:
                stage = staging_path(target,token)
                model = build_classifier(weight_path,metadata['weights_sha256'],seed=seed)
                buffer=io.BytesIO(); torch.save(model.state_dict(),buffer)
                complete_temporary(stage/'initial_state.pt',buffer.getvalue())
                descriptor={'seed':seed,'runtime_sha256':sha256_file(runtime_path),
                            'weights_sha256':metadata['weights_sha256'],'initial_state_sha256':sha256_file(stage/'initial_state.pt')}
                complete_temporary(stage/'initialization.json',(json.dumps(descriptor,sort_keys=True,separators=(',', ':'))+'\n').encode())
                lock=publish_lock(target,{'identity':identity,'files':['initial_state.pt','initialization.json'],
                    'dependencies':{'runtime/lock.json':sha256_file(runtime_path)},'artifact_root':root,'claim':token})
                paths.append(lock)
            finally:
                token.close()
        index_root = root/'initializations'
        index_token_value = hashlib.sha256(('initializations\0'+owner_token).encode()).hexdigest()
        index_token = claim_writer(index_root, 'initializations', token=index_token_value,
                                   resume=(index_root/'.writer').exists())
        try:
            stage = staging_path(index_root,index_token)
            dependencies = {path.relative_to(root).as_posix():sha256_file(path) for path in paths}
            index = {'seeds':[42,123,2026], 'locks':dependencies}
            complete_temporary(stage/'index.json',(json.dumps(index,sort_keys=True,separators=(',', ':'))+'\n').encode())
            return publish_lock(index_root,{'identity':'initializations','files':['index.json'],
                                'dependencies':dependencies,'artifact_root':root,'claim':index_token})
        finally:
            index_token.close()
