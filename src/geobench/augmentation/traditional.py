"""E1 image transformations with confirmed numeric and ordering rules."""
import math
import random

import torch
import torch.nn.functional as F

ORDER = ('brightness', 'contrast', 'blur', 'noise', 'rotation')
RANGES = ((.9, 1.1), (.9, 1.1), (.1, .6), (1 / 255, 3 / 255), (-3., 3.))


def select_transforms(seed):
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(ORDER)), rng.choice((1, 2))))


def sample_transforms(seed):
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(ORDER)), rng.choice((1, 2))))
    return [(ORDER[i], rng.uniform(*RANGES[i])) for i in indexes]


def apply_transforms(image, transforms, *, generator=None):
    x = torch.as_tensor(image)
    if x.ndim != 3 or x.shape[0] != 3 or not x.is_floating_point() or not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('expected finite RGB float CHW image in [0,1]')
    x = x.to(torch.float32).clone()
    names = [name for name, _ in transforms]
    if not 1 <= len(names) <= 2 or any(name not in ORDER for name in names):
        raise ValueError('one or two known transforms required')
    indexes = [ORDER.index(name) for name in names]
    if indexes != sorted(set(indexes)):
        raise ValueError('transforms must be unique and in confirmed order')
    for name, value in transforms:
        low, high = RANGES[ORDER.index(name)]
        if not low <= value <= high:
            raise ValueError('transform parameter outside confirmed range')
        if name == 'brightness':
            x = x * value
        elif name == 'contrast':
            mean = x.mean()
            x = mean + value * (x - mean)
        elif name == 'blur':
            coords = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
            kernel = torch.exp(-coords.square() / (2 * value * value))
            kernel = kernel / kernel.sum()
            weight = torch.outer(kernel, kernel).expand(3, 1, 5, 5)
            x = F.conv2d(F.pad(x.unsqueeze(0), (2, 2, 2, 2), mode='reflect'), weight, groups=3).squeeze(0)
        elif name == 'noise':
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x + noise * value
        else:
            radians = value * math.pi / 180
            height, width = x.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(height, device=x.device, dtype=x.dtype), torch.arange(width, device=x.device, dtype=x.dtype), indexing='ij')
            dx, dy = xx - (width - 1) / 2, yy - (height - 1) / 2
            source_x = math.cos(radians) * dx + math.sin(radians) * dy + (width - 1) / 2
            source_y = -math.sin(radians) * dx + math.cos(radians) * dy + (height - 1) / 2
            grid = torch.stack(((source_x + .5) * 2 / width - 1, (source_y + .5) * 2 / height - 1), dim=-1)
            x = F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='reflection', align_corners=False).squeeze(0)
        x = x.clamp(0, 1)
    return x


def quantize(image):
    x = torch.as_tensor(image)
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('invalid image values')
    return torch.floor(x * 255 + .5).to(torch.uint8)


def e1_policy():
    """Capture the exact local code and backends used for a generation pool."""
    import platform
    import numpy
    import PIL
    import torchvision
    from .. import images
    from ..artifacts import sha256_file
    from . import jobs, quality
    return {'code_sha256': sha256_file(__file__),
            'dependency_code_sha256': {module.__name__: sha256_file(module.__file__) for module in (jobs, quality, images)},
            'backend': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'numpy': numpy.__version__,
                        'pillow': PIL.__version__, 'rng': 'python.random.Random',
                        'rotation': 'torch.grid_sample.bilinear.reflection.align_corners_false',
                        'encoding': 'Pillow RGB PNG'}}


def generate_e1(root, *, dataset_root=None, owner_token=None):
    """Build the shared E1 additions from verified real-training membership."""
    import csv
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from ..artifacts import active_writer, claim_writer, staging_path, publish_lock, verify_lock, sha256_file, complete_temporary, _inside
    from ..images import decode, decoded_pixel_sha256
    from .jobs import source_for_slot, seed_for
    from .quality import audit_candidate, HoldoutCollision, validate_candidate_tensor
    from ..config import verified_configuration

    root = Path(root).resolve()
    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError('owner token required')
    data_lock = root / 'splits/lock.json'
    e0_lock = root / 'pools/E0/pool_lock.json'
    verify_lock(data_lock, boundary=root)
    e0 = verify_lock(e0_lock, boundary=root)
    if e0['dependencies'].get('splits/lock.json') != sha256_file(data_lock):
        raise ValueError('E0 data dependency mismatch')
    descriptor = json.loads((e0_lock.parent/'pool.json').read_text(encoding='utf-8'))
    if descriptor.get('manifest_sha256') != sha256_file(root/'splits/train.csv'):
        raise ValueError('E0 manifest mismatch')
    if dataset_root is None:
        readiness = root/'deployment/data_readiness/lock.json'
        verify_lock(readiness, boundary=root)
        dataset_root = json.loads((readiness.parent/'V01.json').read_text(encoding='utf-8'))['dataset_root']
    dataset_root = Path(dataset_root).resolve(strict=True)
    def read_csv(name):
        with (root/'splits'/name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    train, val, test, groups = (read_csv(name) for name in ('train.csv','val.csv','test.csv','class_groups.csv'))
    source_by_class = {}
    for row in train:
        if row['split'] != 'train' or row['is_synthetic'].lower() != 'false':
            raise ValueError('invalid real training membership')
        source_by_class.setdefault(int(row['class_id']), []).append(row)
    for group in groups:
        class_id, count = int(group['class_id']), int(group['n_train'])
        if len(source_by_class.get(class_id, [])) != count:
            raise ValueError('training class count mismatch')
    for row in train:
        path = _inside(dataset_root, row['relative_path'])
        if sha256_file(path) != row['sha256'] or decoded_pixel_sha256(path) != row['decoded_pixel_sha256']:
            raise ValueError('training source content mismatch')
    configuration = verified_configuration()
    train_files = {row['sha256'] for row in train}
    train_pixels = {row['decoded_pixel_sha256'] for row in train}
    holdout_files = {row['sha256'] for row in val + test}
    holdout_pixels = {row['decoded_pixel_sha256'] for row in val + test}
    target = root/'augmentation/E1'
    identity = 'e1-generation'
    token_value = hashlib.sha256(('e1\0'+owner_token).encode()).hexdigest()
    with active_writer(root):
        token = claim_writer(target, identity, token=token_value, resume=(target/'.writer').exists())
        try:
            stage = staging_path(target, token)
            complete_temporary(stage/'configuration.json', (json.dumps(configuration,sort_keys=True,separators=(',', ':'))+'\n').encode())
            complete_temporary(stage/'policy.json', (json.dumps(e1_policy(),sort_keys=True,separators=(',', ':'))+'\n').encode())
            accepted_files, accepted_pixels, rows, events = set(), set(), [], []
            ledger = stage/'candidates.jsonl'
            if ledger.is_symlink():
                raise ValueError('candidate history cannot be a symlink')
            history = ledger.read_bytes() if ledger.exists() else b''
            cursor = 0
            def record(event):
                nonlocal history, cursor
                events.append(event)
                content = (json.dumps(event, sort_keys=True, separators=(',', ':'))+'\n').encode()
                saved = history[cursor:cursor+len(content)]
                if not content.startswith(saved):
                    raise ValueError('candidate history mismatch')
                if len(saved) < len(content):
                    remaining = content[len(saved):]
                    with ledger.open('ab') as stream:
                        if os.fstat(stream.fileno()).st_nlink != 1:
                            raise ValueError('candidate history cannot be linked')
                        stream.write(remaining); stream.flush(); os.fsync(stream.fileno())
                    history += remaining
                cursor += len(content)
            for group in sorted(groups, key=lambda item:int(item['class_id'])):
                class_id, count = int(group['class_id']), int(group['n_train'])
                if not 20 <= count < 100:
                    continue
                source_rows = {r['relative_path']:r for r in source_by_class[class_id]}
                for slot in range(100-count):
                    name = source_for_slot(list(source_rows), class_id, slot, method='E1')
                    source = source_rows[name]
                    image = decode(_inside(dataset_root, name))
                    x = torch.from_numpy(np.array(image, copy=True)).permute(2,0,1).float().div(255)
                    for attempt in range(5):
                        seed = seed_for('E1', class_id, 5*slot+attempt, 'transforms')
                        transforms = sample_transforms(seed)
                        job = {'method':'E1','class_id':class_id,'slot':slot,'attempt':attempt,'source_id':source['image_id'],
                               'source_sha256':source['sha256'], 'source_pixel_sha256':source['decoded_pixel_sha256'],
                               'config_sha256':configuration['sha256']['methods'],'seed':seed}
                        result = apply_transforms(x, transforms, generator=torch.Generator().manual_seed(seed))
                        technical_failure = validate_candidate_tensor(result, image.size)
                        if technical_failure:
                            record({**job,'transforms':transforms,
                                    'status':'rejected','reason':technical_failure})
                            continue
                        pixels = quantize(result).permute(1,2,0).cpu().numpy()
                        output = io.BytesIO(); Image.fromarray(pixels).save(output, format='PNG')
                        output_id = hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
                        relative = 'images/'+output_id+'.png'
                        path = stage/relative; path.parent.mkdir(parents=True,exist_ok=True)
                        complete_temporary(path,output.getvalue())
                        file_hash, pixel_hash = sha256_file(path), decoded_pixel_sha256(path)
                        try:
                            reason = audit_candidate(file_hash,pixel_hash,train_files=train_files,train_pixels=train_pixels,holdout_files=holdout_files,holdout_pixels=holdout_pixels,accepted_files=accepted_files,accepted_pixels=accepted_pixels)
                        except HoldoutCollision:
                            record({**job,'output_id':output_id,'transforms':transforms,'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'status':'blocked','reason':'holdout_exact_duplicate'})
                            raise
                        record({**job,'output_id':output_id,'class_name':group['class_name'],'group':'medium',
                                'source_cycle':slot // count,'source_position':slot % count,'output_size':list(image.size),
                                'transforms':transforms,'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'status':'accepted' if reason is None else 'rejected','reason':reason})
                        if reason is None:
                            accepted_files.add(file_hash); accepted_pixels.add(pixel_hash)
                            rows.append({'image_id':output_id,'class_id':class_id,'source_id':source['image_id'],
                                         'source_sha256':source['sha256'], 'source_pixel_sha256':source['decoded_pixel_sha256'],
                                         'method':'E1', 'job_id':output_id, 'seed':seed, 'method_config_sha256':configuration['sha256']['methods'],
                                         'relative_path':'augmentation/E1/'+relative,'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'sample_type':'traditional'})
                            break
                    else:
                        record({'method':'E1','class_id':class_id,'slot':slot,'status':'shortage',
                                'reason':'five_attempts_rejected'})
                        raise ValueError(f'E1 quota shortage for class {class_id} slot {slot}')
            manifest = io.StringIO(newline='')
            fields=['image_id','class_id','source_id','source_sha256','source_pixel_sha256','method','job_id','seed','method_config_sha256','relative_path','sha256','decoded_pixel_sha256','sample_type']
            writer=csv.DictWriter(manifest,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
            complete_temporary(stage/'manifest.csv',manifest.getvalue().encode())
            if cursor != len(history):
                raise ValueError('candidate history has unexpected records')
            if not ledger.exists():
                complete_temporary(ledger, b'')
            files=['manifest.csv','candidates.jsonl','configuration.json','policy.json']+[row['relative_path'].removeprefix('augmentation/E1/') for row in rows]
            generation_lock = publish_lock(target,{'identity':identity,'files':files,'dependencies':{'splits/lock.json':sha256_file(data_lock),'pools/E0/pool_lock.json':sha256_file(e0_lock)},'artifact_root':root,'claim':token})
        finally:
            token.close()
        pool = root/'pools/E1'
        pool_token_value = hashlib.sha256(('e1-pool\0'+owner_token).encode()).hexdigest()
        pool_token = claim_writer(pool, 'e1', token=pool_token_value, resume=(pool/'.writer').exists())
        try:
            pool_stage = staging_path(pool, pool_token)
            complete_temporary(pool_stage/'manifest.csv', manifest.getvalue().encode())
            counts = {str(int(group['class_id'])): 100-int(group['n_train'])
                      for group in groups if 20 <= int(group['n_train']) < 100}
            pool_metadata = {'sample_type':'traditional', 'total_count':len(rows), 'per_class_counts':counts,
                             'required_count':sum(counts.values()), 'accepted_count':len(rows),
                             'rejected_count':sum(event['status']=='rejected' for event in events),
                             'incomplete_count':0, 'shortage':0, 'qc_policy':'technical_and_exact_duplicates',
                             'manifest_sha256':sha256_file(pool_stage/'manifest.csv'),
                             'generation_lock_sha256':sha256_file(generation_lock)}
            complete_temporary(pool_stage/'pool.json',(json.dumps(pool_metadata,sort_keys=True,separators=(',', ':'))+'\n').encode())
            return publish_lock(pool,{'identity':'e1','files':['manifest.csv','pool.json'],
                                'dependencies':{'augmentation/E1/lock.json':sha256_file(generation_lock),
                                                'splits/lock.json':sha256_file(data_lock),
                                                'pools/E0/pool_lock.json':sha256_file(e0_lock)},
                                'artifact_root':root,'claim':pool_token,'lock_name':'pool_lock.json'})
        finally:
            pool_token.close()
