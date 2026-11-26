import json
import pytest
from geobench.artifacts import claim_writer, staging_path, publish_lock, verify_lock, sha256_file
from geobench.config import load_config


def test_research_override_rejected(tmp_path):
    config = json.loads(open('configs/benchmark.json', encoding='utf-8').read())
    config['classifier']['batch_size'] = 64
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_partial_publish_not_frozen(tmp_path):
    token = claim_writer(tmp_path, 'run-1')
    with pytest.raises(ValueError):
        publish_lock(tmp_path, {'identity': 'run-1', 'files': ['missing'], 'dependencies': {}, 'claim': token})
    assert not (tmp_path / 'lock.json').exists()


def test_checksum_mismatch_blocks_read(tmp_path):
    token = claim_writer(tmp_path, 'run-1')
    (staging_path(tmp_path, token) / 'output').write_text('original')
    lock = publish_lock(tmp_path, {'identity': 'run-1', 'files': ['output'], 'dependencies': {}, 'claim': token})
    assert verify_lock(lock)['identity'] == 'run-1'
    (tmp_path / 'output').write_text('changed')
    with pytest.raises(ValueError):
        verify_lock(lock)


def test_single_writer_claim(tmp_path):
    claim_writer(tmp_path, 'run-1')
    with pytest.raises(FileExistsError):
        claim_writer(tmp_path, 'run-2')


def test_root_escape_and_changed_dependency_rejected(tmp_path):
    dep = tmp_path / 'dep'; dep.mkdir()
    dep_token = claim_writer(dep, 'dep')
    (staging_path(dep, dep_token) / 'out').write_text('a')
    lock = publish_lock(dep, {'identity': 'dep', 'files': ['out'], 'dependencies': {}, 'claim': dep_token})
    consumer = tmp_path / 'consumer'; consumer.mkdir()
    token = claim_writer(consumer, 'consumer')
    (staging_path(consumer, token) / 'out').write_text('b')
    with pytest.raises(ValueError):
        publish_lock(consumer, {'identity': 'consumer', 'files': ['../dep/out'], 'dependencies': {}, 'claim': token})
    child = publish_lock(consumer, {'identity': 'consumer', 'files': ['out'], 'dependencies': {'dep/lock.json': sha256_file(lock)}, 'artifact_root': tmp_path, 'claim': token})
    (dep / 'out').write_text('tampered')
    with pytest.raises(ValueError):
        verify_lock(child)


def test_cli_verifies_existing_lock(tmp_path):
    import subprocess, sys
    token = claim_writer(tmp_path, 'fixture')
    (staging_path(tmp_path, token) / 'out').write_text('content')
    lock = publish_lock(tmp_path, {'identity': 'fixture', 'files': ['out'], 'dependencies': {}, 'claim': token})
    result = subprocess.run([sys.executable, '-S', '-m', 'geobench', 'verify', str(lock)], capture_output=True, text=True)
    assert result.returncode == 0


def test_boolean_cannot_replace_numeric_research_value(tmp_path):
    config = json.loads(open('configs/benchmark.json', encoding='utf-8').read())
    config['classifier']['amp'] = 0
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_claim_token_and_staged_publication(tmp_path):
    from geobench.artifacts import staging_path
    token = claim_writer(tmp_path, 'job-1')
    stage = staging_path(tmp_path, token)
    (stage / 'out').write_text('complete')
    with pytest.raises(ValueError):
        publish_lock(tmp_path, {'identity': 'job-1', 'files': ['out'], 'dependencies': {}, 'claim': 'wrong'})
    lock = publish_lock(tmp_path, {'identity': 'job-1', 'files': ['out'], 'dependencies': {}, 'claim': token})
    assert (tmp_path / 'out').read_text() == 'complete'
    assert verify_lock(lock)['identity'] == 'job-1'
    assert publish_lock(tmp_path, {'identity': 'job-1', 'files': ['out'], 'dependencies': {}, 'claim': token}) == lock


def test_readiness_requires_verified_evidence_for_same_dataset(tmp_path, monkeypatch):
    import geobench.data as data
    from geobench.artifacts import staging_path
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    root = tmp_path / 'artifacts'
    evidence = root / 'deployment' / 'data_readiness'
    token = claim_writer(evidence, 'data-readiness')
    stage = staging_path(evidence, token)
    records = {
        'V01': {'status': 'verified', 'dataset_root': str(dataset.resolve())},
        'V02': {'status': 'verified', 'artifact_root': str(root.resolve())},
        'V04': {'status': 'verified', 'fingerprint': 'f'*64, 'images': data.EXPECTED_IMAGES, 'classes': data.EXPECTED_CLASSES, 'decoder': data.pillow_version},
    }
    for name, record in records.items():
        record['evidence_files'] = [name + '-evidence.txt']
        (stage / (name + '-evidence.txt')).write_text('fixture evidence')
        if name == 'V01':
            record.update(ssh_target='binhvp@10.204.2.109:22', login_host='login01.cm.cluster', workspace='/datastore/binhvp/BuiTrongTan/')
        if name == 'V02':
            record.update(free_bytes=1000, free_inodes=100, required_bytes=500, required_inodes=50, model_cache_paths=['/fixture/cache'], quota_ok=True, compute_visible=True, atomic_publication=True, single_writer=True)
        if name == 'V04':
            record['checks'] = dict.fromkeys(['readable', 'file_hashes', 'pixel_hashes', 'duplicate_labels', 'quotas', 'leakage'], True)
        (stage / (name + '.json')).write_text(json.dumps(record))
    publish_lock(evidence, {'identity': 'data-readiness', 'files': [name + suffix for name in records for suffix in ('.json', '-evidence.txt')], 'dependencies': {}, 'claim': token})
    assert data._verify_publication_readiness(root, 'f'*64, dataset)
    with pytest.raises(ValueError):
        data._verify_publication_readiness(root, 'e'*64, dataset)


def test_method_config_rejects_research_overrides(tmp_path):
    method = json.loads(open('configs/methods.json', encoding='utf-8').read())
    method['diffusion']['guidance_scale'] = 7.5
    path = tmp_path / 'methods.json'
    path.write_text(json.dumps(method))
    with pytest.raises(ValueError):
        load_config(path)


def test_deployment_cannot_override_research(tmp_path):
    config = json.loads(open('configs/uit.json', encoding='utf-8').read())
    config['batch_size'] = 64
    path = tmp_path / 'uit.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_shared_runtime_requires_confirmed_fields():
    research = load_config(__import__('pathlib').Path('configs/benchmark.json'))
    assert research['classifier']['runtime']['tf32'] is False
    assert research['classifier']['runtime']['num_workers'] == 4
    assert research['classifier']['optimizer']['betas'] == [0.9, 0.999]


def test_confirmed_sampler_and_prompt_configuration():
    from pathlib import Path
    config = load_config(Path('configs/methods.json'))
    assert config['diffusion']['scheduler']['solver_order'] == 2
    assert config['diffusion']['scheduler']['solver_type'] == 'midpoint'
    assert config['diffusion']['prompt'] == 'A realistic street-level road photograph in {country_name}, look like picture in Google Street View.'
    assert config['traditional']['operator_order'] == ['brightness', 'contrast', 'blur', 'noise', 'rotation']


def test_readiness_does_not_accept_unsupported_verified_status(tmp_path):
    from geobench.data import _verify_publication_readiness, EXPECTED_IMAGES, EXPECTED_CLASSES, pillow_version
    root = tmp_path / 'root'; dataset = tmp_path / 'dataset'; dataset.mkdir()
    evidence = root / 'deployment/data_readiness'
    token = claim_writer(evidence, 'data-readiness'); stage = staging_path(evidence, token)
    records = {'V01': {'status': 'verified', 'dataset_root': str(dataset.resolve())},
               'V02': {'status': 'verified', 'artifact_root': str(root.resolve())},
               'V04': {'status': 'verified', 'fingerprint': 'f' * 64, 'images': EXPECTED_IMAGES,
                       'classes': EXPECTED_CLASSES, 'decoder': pillow_version}}
    for name, record in records.items():
        (stage / (name + '.json')).write_text(json.dumps(record))
    publish_lock(evidence, {'identity': 'data-readiness', 'files': ['V01.json', 'V02.json', 'V04.json'], 'dependencies': {}, 'claim': token})
    with pytest.raises(ValueError):
        _verify_publication_readiness(root, 'f' * 64, dataset)


def test_dependency_outside_declared_root_is_rejected(tmp_path):
    root = tmp_path / 'root'; token = claim_writer(root, 'job')
    (staging_path(root, token) / 'out').write_text('test')
    with pytest.raises(ValueError):
        publish_lock(root, {'identity': 'job', 'files': ['out'], 'dependencies': {'../outside/lock.json': 'a' * 64}, 'claim': token})
    assert not (root / 'lock.json').exists()


def test_failed_prepublication_verification_never_exposes_lock(tmp_path, monkeypatch):
    import geobench.artifacts as artifacts
    token = claim_writer(tmp_path, 'job')
    (staging_path(tmp_path, token) / 'out').write_text('ok')
    def reject(path, *args):
        raise ValueError('injected verification failure')
    monkeypatch.setattr(artifacts, 'verify_lock', reject)
    with pytest.raises(ValueError):
        publish_lock(tmp_path, {'identity': 'job', 'files': ['out'], 'dependencies': {}, 'claim': token})
    assert not (tmp_path / 'lock.json').exists()


def test_cli_prepare_data_calls_real_readiness_gate(tmp_path):
    import subprocess, sys
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    result = subprocess.run([sys.executable, '-S', '-m', 'geobench', 'prepare-data', '--dataset-root', str(dataset), '--root', str(tmp_path / 'artifacts')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'invalid choice' not in result.stderr
    assert not (tmp_path / 'artifacts/splits/lock.json').exists()


def test_published_outputs_are_independent_of_staging(tmp_path):
    token = claim_writer(tmp_path, 'job')
    stage_file = staging_path(tmp_path, token) / 'out'
    stage_file.write_text('frozen')
    lock = publish_lock(tmp_path, {'identity': 'job', 'files': ['out'], 'dependencies': {}, 'claim': token})
    stage_file.write_text('changed')
    assert (tmp_path / 'out').read_text() == 'frozen'
    verify_lock(lock)


def test_nested_dependency_cannot_escape_trusted_root(tmp_path):
    outside = tmp_path / 'outside'
    outside_token = claim_writer(outside, 'outside')
    (staging_path(outside, outside_token) / 'out').write_text('a')
    outside_lock = publish_lock(outside, {'identity': 'outside', 'files': ['out'], 'dependencies': {}, 'claim': outside_token})
    root = tmp_path / 'benchmark'; child = root / 'child'
    child_token = claim_writer(child, 'child')
    (staging_path(child, child_token) / 'out').write_text('b')
    child_lock = publish_lock(child, {'identity': 'child', 'files': ['out'], 'dependencies': {'outside/lock.json': sha256_file(outside_lock)}, 'artifact_root': tmp_path, 'claim': child_token})
    consumer = root / 'consumer'; token = claim_writer(consumer, 'consumer')
    (staging_path(consumer, token) / 'out').write_text('c')
    with pytest.raises(ValueError):
        publish_lock(consumer, {'identity': 'consumer', 'files': ['out'], 'dependencies': {'child/lock.json': sha256_file(child_lock)}, 'artifact_root': root, 'claim': token})
    assert not (consumer / 'lock.json').exists()


def test_readiness_requires_actual_site_and_storage_facts(tmp_path):
    from geobench.data import _verify_publication_readiness, EXPECTED_IMAGES, EXPECTED_CLASSES, pillow_version
    root = tmp_path / 'root'; dataset = tmp_path / 'dataset'; dataset.mkdir()
    evidence = root / 'deployment/data_readiness'
    token = claim_writer(evidence, 'data-readiness'); stage = staging_path(evidence, token)
    records = {'V01': {'status': 'verified', 'dataset_root': str(dataset.resolve())},
               'V02': {'status': 'verified', 'artifact_root': str(root.resolve())},
               'V04': {'status': 'verified', 'fingerprint': 'f' * 64, 'images': EXPECTED_IMAGES,
                       'classes': EXPECTED_CLASSES, 'decoder': pillow_version,
                       'checks': dict.fromkeys(['readable', 'file_hashes', 'pixel_hashes', 'duplicate_labels', 'quotas', 'leakage'], True)}}
    for name, record in records.items():
        record['evidence_files'] = [name + '.txt']
        (stage / (name + '.txt')).write_text('fixture')
        (stage / (name + '.json')).write_text(json.dumps(record))
    files = [name + suffix for name in records for suffix in ('.json', '.txt')]
    publish_lock(evidence, {'identity': 'data-readiness', 'files': files, 'dependencies': {}, 'claim': token})
    with pytest.raises(ValueError):
        _verify_publication_readiness(root, 'f' * 64, dataset)


def test_interrupted_owner_can_resume_without_overwriting_mismatches(tmp_path):
    owner = 'persisted-job-secret'
    token = claim_writer(tmp_path, 'job', token=owner)
    stage = staging_path(tmp_path, token)
    (stage / 'a').write_text('first')
    (stage / 'b').write_text('second')
    (tmp_path / 'a').write_text('first')
    assert claim_writer(tmp_path, 'job', token=owner, resume=True) == owner
    with pytest.raises((FileExistsError, ValueError)):
        claim_writer(tmp_path, 'job', token='other', resume=True)
    lock = publish_lock(tmp_path, {'identity': 'job', 'files': ['a', 'b'], 'dependencies': {}, 'claim': token})
    assert verify_lock(lock)['files']['a'] == sha256_file(tmp_path / 'a')


def test_interrupted_output_mismatch_is_never_repaired(tmp_path):
    token = claim_writer(tmp_path, 'job')
    (staging_path(tmp_path, token) / 'out').write_text('expected')
    (tmp_path / 'out').write_text('different')
    with pytest.raises(ValueError):
        publish_lock(tmp_path, {'identity': 'job', 'files': ['out'], 'dependencies': {}, 'claim': token})
    assert (tmp_path / 'out').read_text() == 'different'
    assert not (tmp_path / 'lock.json').exists()


def test_resolve_run_verifies_frozen_dependencies(tmp_path):
    from geobench.config import resolve_run
    dep = tmp_path / 'deps'; token = claim_writer(dep, 'dep')
    (staging_path(dep, token) / 'out').write_text('data')
    lock = publish_lock(dep, {'identity': 'dep', 'files': ['out'], 'dependencies': {}, 'claim': token})
    run = tmp_path / 'runs' / 'E0-42'; run.mkdir(parents=True)
    (run / 'descriptor.json').write_text(json.dumps({'run_id': 'E0-42', 'dependencies': {'deps/lock.json': sha256_file(lock)}}))
    assert resolve_run(tmp_path, 'E0-42')['run_id'] == 'E0-42'
    (dep / 'out').write_text('tampered')
    with pytest.raises(ValueError):
        resolve_run(tmp_path, 'E0-42')
