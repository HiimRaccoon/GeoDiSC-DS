from pathlib import Path
import pytest
import torch
from PIL import Image, PngImagePlugin
from geobench.data import quotas, split_inventory, inventory
from geobench.images import classifier_geometry, normalize_classifier, pad_native, crop_native, decoded_pixel_sha256


@pytest.mark.parametrize('n,expected', [(1,(1,0,0)), (2,(1,0,1)), (3,(1,1,1)), (9,(7,1,1)), (10,(8,1,1)), (15,(11,2,2))])
def test_split_quotas(n, expected):
    assert quotas(n) == expected


def make_images(root, counts):
    for cls, values in counts.items():
        (root / cls).mkdir(parents=True)
        for i, value in enumerate(values):
            Image.new('RGB', (16, 16), (value, value, value)).save(root / cls / f'{i}.png')


def test_duplicate_components_and_conflicts(tmp_path):
    make_images(tmp_path, {'a': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    rows, mapping = inventory(tmp_path, extensions={'.png'})
    result = split_inventory(rows)
    duplicate = [r for r in result if r['relative_path'] in ('a/0.png', 'a/1.png')]
    assert duplicate[0]['split'] == duplicate[1]['split']
    assert [sum(r['split'] == s for r in result) for s in ('train', 'val', 'test')] == [8,1,1]
    make_images(tmp_path, {'b': [1]})
    rows, _ = inventory(tmp_path, extensions={'.png'})
    with pytest.raises(ValueError, match='label'):
        split_inventory(rows)


def test_infeasible_duplicate_quota(tmp_path):
    make_images(tmp_path, {'a': [1, 1, 1]})
    rows, _ = inventory(tmp_path, extensions={'.png'})
    with pytest.raises(ValueError, match='quota'):
        split_inventory(rows)


def test_pixel_identity_ignores_metadata(tmp_path):
    image = Image.new('RGB', (2, 3), 'red')
    image.save(tmp_path / 'a.png')
    meta = PngImagePlugin.PngInfo(); meta.add_text('note', 'different')
    image.save(tmp_path / 'b.png', pnginfo=meta)
    assert decoded_pixel_sha256(tmp_path / 'a.png') == decoded_pixel_sha256(tmp_path / 'b.png')
    assert (tmp_path / 'a.png').read_bytes() != (tmp_path / 'b.png').read_bytes()


def test_geometry_and_padding():
    image = Image.new('RGB', (1030, 703), 'white')
    tensor = classifier_geometry(image)
    assert tensor.shape == (3, 384, 896)
    assert torch.allclose(normalize_classifier(tensor)[:, 0, 0], torch.zeros(3))
    padded, meta = pad_native(image)
    assert padded.shape == (3, 704, 1032)
    assert crop_native(padded, meta).shape == (3, 703, 1030)
    assert torch.equal(crop_native(padded, meta), torch.ones((3,703,1030)))


def test_prepare_data_rejects_unverified_inventory(tmp_path):
    from geobench.data import prepare_data
    dataset = tmp_path / 'dataset'
    make_images(dataset, {'a': [1]})
    with pytest.raises(ValueError):
        prepare_data(dataset, tmp_path / 'artifacts')
    assert not (tmp_path / 'artifacts' / 'splits' / 'lock.json').exists()


def test_fixture_data_publication_has_locked_manifests(tmp_path, monkeypatch):
    import geobench.data as data
    import geobench.artifacts as artifacts
    import json
    from geobench.artifacts import claim_writer, staging_path, publish_lock, verify_lock, sha256_file
    dataset = tmp_path / 'dataset'
    make_images(dataset, {'a': [1, 2, 3], 'b': [4]})
    monkeypatch.setattr(data, 'EXPECTED_CLASSES', 2)
    monkeypatch.setattr(data, 'EXPECTED_IMAGES', 4)
    monkeypatch.setattr(data, 'IMAGE_EXTENSIONS', {'.png'})
    root = tmp_path / 'artifacts'
    rows, mapping = data.inventory(dataset, {'.png'})
    evidence = root / 'deployment/data_readiness'
    token = claim_writer(evidence, 'data-readiness')
    stage = staging_path(evidence, token)
    records = {'V01': {'status': 'verified', 'dataset_root': str(dataset.resolve())},
               'V02': {'status': 'verified', 'artifact_root': str(root.resolve())},
               'V04': {'status': 'verified', 'fingerprint': data.inventory_fingerprint(rows, mapping),
                       'images': 4, 'classes': 2, 'decoder': data.pillow_version}}
    for name, record in records.items():
        record['evidence_files'] = [name + '-evidence.txt']
        (stage / (name + '-evidence.txt')).write_text('fixture evidence')
        if name == 'V01':
            record.update(ssh_target='user@fixture.invalid:22', login_host='login.fixture.invalid', workspace='/fixture/workspace/')
        if name == 'V02':
            record.update(free_bytes=1000, free_inodes=100, required_bytes=500, required_inodes=50, model_cache_paths=['/fixture/cache'], quota_ok=True, compute_visible=True, atomic_publication=True, single_writer=True)
        if name == 'V04':
            record['checks'] = dict.fromkeys(['readable', 'file_hashes', 'pixel_hashes', 'duplicate_labels', 'quotas', 'leakage'], True)
        (stage / (name + '.json')).write_text(json.dumps(record))
    from site_evidence import publish_site
    site_lock = publish_site(root)
    publish_lock(evidence, {'identity': 'data-readiness', 'files': [name + suffix for name in records for suffix in ('.json', '-evidence.txt')], 'dependencies': {'deployment/private_site/lock.json': sha256_file(site_lock)}, 'artifact_root':root, 'claim': token})
    original_publish = artifacts.publish_lock
    def interrupt_e0(path, descriptor):
        if descriptor['identity'].startswith('e0:'):
            raise ValueError('injected interruption')
        return original_publish(path, descriptor)
    monkeypatch.setattr(artifacts, 'publish_lock', interrupt_e0)
    with pytest.raises(ValueError, match='injected interruption'):
        data.prepare_data(dataset, root, owner_token='fixture-owner')
    monkeypatch.setattr(artifacts, 'publish_lock', original_publish)
    frozen_data_lock = (root/'splits/lock.json').read_bytes()
    for partial in (root/'splits/.staging/train.csv', root/'pools/E0/.staging/pool.json'):
        complete = partial.read_bytes()
        partial.write_bytes(complete[:10])

    with pytest.raises(ValueError):
        data.prepare_data(dataset, root, owner_token='wrong-owner')
    lock = data.prepare_data(dataset, root, owner_token='fixture-owner')
    import hashlib
    snapshot = json.loads((root/'splits/configuration.json').read_text())
    for name, value in snapshot['values'].items():
        assert snapshot['sha256'][name] == hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    pool = json.loads((root/'pools/E0/pool.json').read_text())
    assert pool['condition'] == 'E0' and pool['non_real_count'] == 0
    import csv
    with (root/'splits/train.csv').open() as stream:
        assert all(row['is_synthetic'] == 'false' for row in csv.DictReader(stream))
    second_root = tmp_path/'second-artifacts'
    records['V02']['artifact_root'] = str(second_root.resolve())
    evidence_files = {name+'-evidence.txt': b'fixture evidence' for name in records}
    publish_site(second_root)
    produced = data.publish_readiness(dataset, second_root, records, evidence_files, owner_token='readiness-owner')
    assert verify_lock(produced, boundary=second_root)['identity'] == 'data-readiness'
    assert data.publish_readiness(dataset, second_root, records, evidence_files, owner_token='readiness-owner') == produced
    invalid_root = tmp_path/'invalid-readiness'
    records['V02']['artifact_root'] = str(invalid_root.resolve())
    records['V01']['status'] = 'pending'
    with pytest.raises(ValueError):
        data.publish_readiness(dataset, invalid_root, records, evidence_files, owner_token='bad-evidence')
    assert not (invalid_root/'deployment/data_readiness/lock.json').exists()
    records['V01']['status'] = 'verified'

    assert data._verify_publication_readiness(second_root, data.inventory_fingerprint(rows,mapping), dataset)
    assert (root/'splits/lock.json').read_bytes() == frozen_data_lock
    info = verify_lock(lock, boundary=root)
    assert {'train.csv', 'val.csv', 'test.csv', 'class_groups.csv', 'split_metadata.json'} <= set(info['files'])
    metadata = json.loads((root / 'splits/split_metadata.json').read_text(encoding='utf-8'))
    assert metadata['duplicate_audit_sha256'] == info['files']['duplicate_audit.json']
    assert 'source_identity.json' in info['files']
    assert verify_lock(tmp_path / 'artifacts/pools/E0/pool_lock.json', boundary=root)


def test_empty_class_rejected(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError):
        inventory(tmp_path, extensions={'.png'})


def test_pixel_duplicate_links_distinct_files_and_isolates_splits(tmp_path):
    make_images(tmp_path, {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    img = Image.open(tmp_path / 'a/0.png')
    meta = PngImagePlugin.PngInfo(); meta.add_text('note', 'other metadata')
    img.save(tmp_path / 'a/1.png', pnginfo=meta)
    rows, _ = inventory(tmp_path, extensions={'.png'})
    assert rows[0]['sha256'] != rows[1]['sha256']
    result = split_inventory(rows)
    a, b = result[0], result[1]
    assert a['duplicate_group_id'] == b['duplicate_group_id']
    assert a['split'] == b['split']
    for key in ('image_id', 'sha256', 'decoded_pixel_sha256'):
        sets = [{r[key] for r in result if r['split'] == split} for split in ('train', 'val', 'test')]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_other_native_size_roundtrip():
    image = Image.new('RGB', (1536, 662), 'blue')
    padded, metadata = pad_native(image)
    assert padded.shape == (3, 664, 1536)
    assert crop_native(padded, metadata).shape == (3, 662, 1536)
    assert metadata['offsets'] == [0, 1, 0, 1]


def test_data_freeze_blocks_without_verification_evidence(tmp_path, monkeypatch):
    import geobench.data as data
    dataset = tmp_path / 'dataset'
    make_images(dataset, {'a': [1, 2, 3]})
    monkeypatch.setattr(data, 'EXPECTED_CLASSES', 1)
    monkeypatch.setattr(data, 'EXPECTED_IMAGES', 3)
    monkeypatch.setattr(data, 'IMAGE_EXTENSIONS', {'.png'})
    with pytest.raises(ValueError):
        data.prepare_data(dataset, tmp_path / 'artifacts')
    assert not (tmp_path / 'artifacts/splits/lock.json').exists()


def test_pixel_duplicate_across_labels_is_rejected(tmp_path):
    make_images(tmp_path, {'a': [1], 'b': [2]})
    image = Image.open(tmp_path / 'a/0.png')
    meta = PngImagePlugin.PngInfo(); meta.add_text('label', 'different metadata')
    image.save(tmp_path / 'b/0.png', pnginfo=meta)
    rows, _ = inventory(tmp_path, extensions={'.png'})
    assert rows[0]['sha256'] != rows[1]['sha256']
    with pytest.raises(ValueError, match='label'):
        split_inventory(rows)


@pytest.mark.parametrize('n,expected', [(1,(1,0,0)), (2,(1,0,1)), (3,(1,1,1)), (9,(7,1,1)), (10,(8,1,1)), (15,(11,2,2))])
def test_boundary_memberships(tmp_path, n, expected):
    make_images(tmp_path, {'a': list(range(1, n + 1))})
    rows, _ = inventory(tmp_path, extensions={'.png'})
    split = split_inventory(rows)
    assert tuple(sum(r['split'] == name for r in split) for name in ('train', 'val', 'test')) == expected
    assert split == split_inventory(rows)


def test_letterbox_odd_padding_goes_on_right():
    x = classifier_geometry(Image.new('RGB', (1030, 703), 'white'))
    assert x[0, 0, 165].item() == pytest.approx(.485)
    assert x[0, 0, 166].item() == pytest.approx(1)
    assert x[0, 0, 728].item() == pytest.approx(1)
    assert x[0, 0, 729].item() == pytest.approx(.485)


def test_pixel_hash_canonical_header_and_dimensions(tmp_path):
    import hashlib, struct
    path = tmp_path / 'rgb.png'
    Image.new('RGB', (2, 1), (1, 2, 3)).save(path)
    expected = hashlib.sha256(b'RGB8\0' + struct.pack('>QQ', 2, 1) + bytes([1, 2, 3, 1, 2, 3])).hexdigest()
    assert decoded_pixel_sha256(path) == expected


def test_fingerprint_is_root_independent_and_detects_content_change(tmp_path):
    from geobench.data import inventory_fingerprint
    first = tmp_path / 'first'; second = tmp_path / 'second'
    make_images(first, {'a': [1, 2, 3]})
    make_images(second, {'a': [1, 2, 3]})
    a, am = inventory(first, {'.png'}); b, bm = inventory(second, {'.png'})
    assert inventory_fingerprint(a, am) == inventory_fingerprint(b, bm)
    Image.new('RGB', (16, 16), (4, 4, 4)).save(second / 'a/0.png')
    c, cm = inventory(second, {'.png'})
    assert inventory_fingerprint(a, am) != inventory_fingerprint(c, cm)
