import pytest
import torch
from PIL import Image

from geobench.artifacts import sha256_file
from geobench.images import decoded_pixel_sha256


def row(path, sample_type):
    return {'image_id': path.stem, 'relative_path': path.name, 'class_id': '7',
            'sha256': sha256_file(path), 'decoded_pixel_sha256': decoded_pixel_sha256(path),
            'sample_type': sample_type}


def test_dataset_resolves_real_and_augmented_roots_and_checks_content(tmp_path):
    from geobench.training_data import ImageDataset
    real = tmp_path / 'real'; real.mkdir()
    artifacts = tmp_path / 'artifacts'; artifacts.mkdir()
    a = real / 'a.png'; b = artifacts / 'b.png'
    Image.new('RGB', (10, 6), 'red').save(a)
    Image.new('RGB', (10, 6), 'blue').save(b)
    dataset = ImageDataset([row(a, 'real'), row(b, 'traditional')], real, artifacts)
    image, label = dataset[0]
    assert image.shape == (3, 384, 896) and label == 7
    assert torch.isfinite(image).all() and not torch.equal(image, dataset[1][0])
    Image.new('RGB', (10, 6), 'green').save(a)
    with pytest.raises(ValueError, match='hash'):
        dataset[0]


def test_dataset_rejects_duplicate_ids_and_path_escape(tmp_path):
    from geobench.training_data import ImageDataset
    path = tmp_path / 'a.png'; Image.new('RGB', (8, 8)).save(path)
    entry = row(path, 'real')
    with pytest.raises(ValueError, match='duplicate'):
        ImageDataset([entry, entry], tmp_path, tmp_path)
    with pytest.raises(ValueError, match='path'):
        ImageDataset([{**entry, 'relative_path': '../a.png'}], tmp_path, tmp_path)


def test_loader_uses_confirmed_traversal_and_settings(tmp_path):
    from geobench.training_data import ImageDataset, make_loaders
    path = tmp_path / 'a.png'; Image.new('RGB', (8, 8)).save(path)
    dataset = ImageDataset([row(path, 'real')], tmp_path, tmp_path)
    train, validation, generator = make_loaders(dataset, dataset, seed=42)
    from torch.utils.data import RandomSampler, SequentialSampler
    assert isinstance(train.sampler, RandomSampler) and train.sampler.generator is generator
    assert isinstance(validation.sampler, SequentialSampler)
    assert all(loader.batch_size == 128 and loader.num_workers == 4 and loader.pin_memory
               and loader.prefetch_factor == 2 and not loader.persistent_workers and not loader.drop_last
               for loader in (train, validation))
    with pytest.raises(ValueError, match='seed'):
        make_loaders(dataset, dataset, seed=99)


def test_locked_inputs_enforce_real_train_membership_and_validation(tmp_path):
    import csv
    import json
    from geobench.artifacts import claim_writer, staging_path, publish_lock
    from geobench.training_data import load_locked_datasets

    dataset = tmp_path / 'dataset'; dataset.mkdir()
    root = tmp_path / 'artifacts'; root.mkdir()
    image = dataset / 'a.png'; Image.new('RGB', (8, 8)).save(image)
    entry = {**row(image, 'real'), 'split': 'train', 'is_synthetic': 'false'}
    fields = list(entry)
    token = claim_writer(root / 'splits', 'common-data:fixture')
    stage = staging_path(root / 'splits', token)
    for name in ('train', 'val'):
        with (stage / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader()
            writer.writerow({**entry, 'split': name})
    data = publish_lock(root / 'splits', {'identity': 'common-data:fixture',
        'files': ['train.csv', 'val.csv'], 'dependencies': {}, 'claim': token})
    token = claim_writer(root / 'pools/E0', 'e0:fixture')
    stage = staging_path(root / 'pools/E0', token)
    (stage / 'pool.json').write_text(json.dumps({'manifest_sha256': sha256_file(data.parent / 'train.csv')}))
    pool = publish_lock(root / 'pools/E0', {'identity': 'e0:fixture', 'files': ['pool.json'],
        'dependencies': {'splits/lock.json': sha256_file(data)}, 'artifact_root': root,
        'claim': token, 'lock_name': 'pool_lock.json'})
    train, val = load_locked_datasets(root, dataset, data, pool)
    assert len(train) == len(val) == 1 and train[0][1] == val[0][1] == 7
    (data.parent / 'val.csv').write_text('tampered')
    with pytest.raises(ValueError):
        load_locked_datasets(root, dataset, data, pool)


def test_hybrid_pool_cannot_change_real_membership(tmp_path):
    import csv
    from geobench.artifacts import claim_writer, staging_path, publish_lock
    from geobench.training_data import load_locked_datasets

    dataset = tmp_path / 'dataset'; dataset.mkdir()
    root = tmp_path / 'artifacts'; root.mkdir()
    image = dataset / 'a.png'; Image.new('RGB', (8, 8)).save(image)
    entry = {**row(image, 'real'), 'split': 'train', 'is_synthetic': 'false'}
    token = claim_writer(root / 'splits', 'common-data:fixture'); stage = staging_path(root / 'splits', token)
    for name in ('train', 'val'):
        with (stage / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(entry)); writer.writeheader(); writer.writerow({**entry, 'split': name})
    data = publish_lock(root / 'splits', {'identity': 'common-data:fixture', 'files': ['train.csv', 'val.csv'],
                                        'dependencies': {}, 'claim': token})
    token = claim_writer(root / 'pools/hybrid', 'pool:fixture'); stage = staging_path(root / 'pools/hybrid', token)
    with (stage / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(entry)); writer.writeheader(); writer.writerow({**entry, 'class_id': '8'})
    pool = publish_lock(root / 'pools/hybrid', {'identity': 'pool:fixture', 'files': ['manifest.csv'],
                        'dependencies': {'splits/lock.json': sha256_file(data)}, 'artifact_root': root,
                        'claim': token, 'lock_name': 'pool_lock.json'})
    with pytest.raises(ValueError, match='real content'):
        load_locked_datasets(root, dataset, data, pool)
