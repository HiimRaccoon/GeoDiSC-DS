"""Manifest-backed classifier inputs and deterministic loader settings."""
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .artifacts import _inside, sha256_file
from .images import classifier_geometry, decode, decoded_pixel_sha256, normalize_classifier


class ImageDataset(Dataset):
    def __init__(self, rows, dataset_root, artifact_root):
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.artifact_root = Path(artifact_root).resolve(strict=True)
        self.rows = []
        seen = set()
        for row in rows:
            item = dict(row)
            image_id = item.get('image_id')
            if not isinstance(image_id, str) or not image_id or image_id in seen:
                raise ValueError('missing or duplicate image ID')
            seen.add(image_id)
            kind = item.get('sample_type') or ('real' if item.get('is_synthetic') == 'false' else None)
            if kind not in {'real', 'traditional', 'final_synthetic'}:
                raise ValueError('unknown sample type')
            try:
                label = int(item['class_id'])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('invalid class ID') from exc
            if not 0 <= label < 124:
                raise ValueError('invalid class ID')
            root = self.dataset_root if kind == 'real' else self.artifact_root
            path = _inside(root, item.get('relative_path'))
            for key in ('sha256', 'decoded_pixel_sha256'):
                if not isinstance(item.get(key), str) or len(item[key]) != 64 or any(c not in '0123456789abcdef' for c in item[key]):
                    raise ValueError('invalid image hash')
            self.rows.append((path, label, item['sha256'], item['decoded_pixel_sha256']))
        if not self.rows:
            raise ValueError('empty dataset')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label, file_hash, pixel_hash = self.rows[index]
        if sha256_file(path) != file_hash or decoded_pixel_sha256(path) != pixel_hash:
            raise ValueError('image hash mismatch')
        return normalize_classifier(classifier_geometry(decode(path))), label


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(train_dataset, validation_dataset, *, seed):
    if type(seed) is not int or seed not in (42, 123, 2026):
        raise ValueError('invalid training seed')
    generator = torch.Generator().manual_seed(seed)
    settings = dict(batch_size=128, num_workers=4, pin_memory=True,
                    persistent_workers=False, prefetch_factor=2, drop_last=False,
                    worker_init_fn=_seed_worker)
    train = DataLoader(train_dataset, shuffle=True, generator=generator, **settings)
    validation = DataLoader(validation_dataset, shuffle=False, **settings)
    return train, validation, generator


def load_locked_datasets(root, dataset_root, data_lock, pool_lock):
    """Load train and validation membership only from verified pool/data locks."""
    import csv
    import json
    from .artifacts import verify_lock

    root = Path(root).resolve(strict=True)
    data_lock, pool_lock = Path(data_lock).resolve(strict=True), Path(pool_lock).resolve(strict=True)
    if not data_lock.is_relative_to(root) or not pool_lock.is_relative_to(root):
        raise ValueError('lock outside artifact root')
    data = verify_lock(data_lock, boundary=root)
    pool = verify_lock(pool_lock, boundary=root)
    if not data['identity'].startswith('common-data:'):
        raise ValueError('invalid common data identity')
    if pool['dependencies'].get(data_lock.relative_to(root).as_posix()) != sha256_file(data_lock):
        raise ValueError('pool data dependency mismatch')
    for name in ('train.csv', 'val.csv'):
        if name not in data['files']:
            raise ValueError('data manifest not locked')
    def read(path):
        with path.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    real = read(data_lock.parent / 'train.csv')
    val = read(data_lock.parent / 'val.csv')
    for rows, split in ((real, 'train'), (val, 'val')):
        if any(row.get('split') != split or row.get('is_synthetic') != 'false' for row in rows):
            raise ValueError('invalid real split membership')
    if pool['identity'].startswith('e0:'):
        if 'pool.json' not in pool['files']:
            raise ValueError('pool metadata not locked')
        metadata = json.loads((pool_lock.parent / 'pool.json').read_text(encoding='utf-8'))
        if metadata.get('manifest_sha256') != sha256_file(data_lock.parent / 'train.csv'):
            raise ValueError('E0 manifest hash mismatch')
        train = real
    else:
        if 'manifest.csv' not in pool['files']:
            raise ValueError('training manifest not locked')
        train = read(pool_lock.parent / 'manifest.csv')
        original = {row['image_id']: row for row in real}
        pool_real = [row for row in train if row.get('sample_type') == 'real']
        if len(pool_real) != len(real) or {row['image_id'] for row in pool_real} != set(original):
            raise ValueError('training pool real membership mismatch')
        for row in pool_real:
            source = original[row['image_id']]
            if any(row.get(key) != source.get(key) for key in ('relative_path', 'class_id', 'sha256', 'decoded_pixel_sha256')):
                raise ValueError('training pool real content mismatch')
    return ImageDataset(train, dataset_root, root), ImageDataset(val, dataset_root, root)
