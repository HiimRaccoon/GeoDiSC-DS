import pytest
from geobench.preflight import validate_profile, readiness_by_role


def valid():
    return {'artifact_root':'/tmp/benchmark','cpu_count':8,'memory':'64G','walltime':'3-00:00:00',
            'account':'uit','partition':'normal','qos':'mig3-71','gres':'gpu:3g.71gb:1'}


@pytest.mark.parametrize('field', ['artifact_root','cpu_count','memory','walltime'])
def test_unknown_resources_are_rejected(field):
    profile=valid(); profile[field]=None
    with pytest.raises(ValueError): validate_profile(profile)


def test_invalid_gpu_pair_and_research_override_rejected():
    profile=valid(); profile['gres']='gpu:7g.141gb:1'
    with pytest.raises(ValueError): validate_profile(profile)
    profile=valid(); profile['batch_size']=32
    with pytest.raises(ValueError): validate_profile(profile)


def test_missing_dino_does_not_block_classifier_and_diffusion():
    roles=readiness_by_role({'site':True,'storage':True,'base_models':True,'dino':False})
    assert roles['classifier'] and roles['diffusion'] and not roles['dino']
import hashlib
import json
import pytest
from geobench.preflight import inspect_deployment, verify_snapshot, probe_gpu
from geobench.artifacts import verify_lock


def test_inspection_publishes_pending_evidence_without_private_profile(tmp_path):
    report = inspect_deployment(tmp_path, {'profile': None}, owner_token='owner')
    lock = verify_lock(report, boundary=tmp_path)
    value = json.loads((report.parent/'preflight.json').read_text())
    assert value['status']['site'] == 'pending'
    assert value['status']['dino'] == 'pending'
    assert not value['roles']['dino']
    assert lock['identity'] == 'deployment-preflight'


def test_snapshot_requires_pinned_revision_and_matching_files(tmp_path):
    (tmp_path/'config.json').write_text('model')
    digest = hashlib.sha256(b'model').hexdigest()
    desc={'repository':'example/model','revision':'a'*40,'files':{'config.json':digest}}
    assert verify_snapshot(tmp_path,desc)['files']['config.json'] == digest
    (tmp_path/'config.json').write_text('different')
    with pytest.raises(ValueError): verify_snapshot(tmp_path,desc)
    desc['revision']='main'
    with pytest.raises(ValueError): verify_snapshot(tmp_path,desc)


def test_probe_refuses_to_run_outside_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='Slurm'): probe_gpu(tmp_path,'classifier')


def test_environment_and_storage_inspection_records_actual_observations(tmp_path):
    from geobench.preflight import inspect_environment, inspect_storage
    environment = inspect_environment(['json', 'geobench_missing_fixture_package'])
    assert environment['python']['version']
    assert environment['packages']['geobench_missing_fixture_package']['status'] == 'unavailable'
    storage = inspect_storage(tmp_path)
    assert storage['free_bytes'] > 0 and storage['status'] == 'observed'


def test_probe_delegates_only_to_shared_provider(tmp_path, monkeypatch):
    import types
    import geobench.preflight as preflight
    calls=[]
    fake=types.SimpleNamespace(probe_gpu=lambda root: calls.append(root) or 'shared-result')
    monkeypatch.setenv('SLURM_JOB_ID','1234')
    monkeypatch.setattr('importlib.import_module', lambda name: fake)
    assert preflight.probe_gpu(tmp_path,'classifier') == 'shared-result'
    assert calls == [tmp_path]


def test_dino_snapshot_rejects_wrong_checkpoint(tmp_path):
    from geobench.preflight import verify_model_snapshot
    (tmp_path/'weights').write_bytes(b'weights')
    import hashlib
    descriptor={'repository':'wrong/model','revision':'a'*40,'files':{'weights':hashlib.sha256(b'weights').hexdigest()}}
    with pytest.raises(ValueError): verify_model_snapshot(tmp_path,'dino',descriptor)


def test_preflight_cli_is_available_and_requires_owner(tmp_path, monkeypatch):
    from geobench.cli import main
    monkeypatch.delenv('GEOBENCH_OWNER_TOKEN', raising=False)
    with pytest.raises(ValueError):
        main(['preflight','--root',str(tmp_path)])


def test_role_gates_require_verified_lock_evidence(tmp_path):
    from geobench.preflight import assess_readiness
    from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file
    evidence={}
    for key in ('V01','V02','V03','V04','V05','V07_base','V08','V09','V13'):
        path=tmp_path/'deployment'/key
        token=claim_writer(path,key); stage=staging_path(path,token)
        (stage/'evidence.json').write_text(json.dumps({'status':'verified','verification':key,'evidence_files':['observation.txt']}))
        (stage/'observation.txt').write_text('fixture observation')
        lock=publish_lock(path,{'identity':key,'files':['evidence.json','observation.txt'],'dependencies':{},'claim':token})
        evidence[key]={'path':lock.relative_to(tmp_path).as_posix(),'sha256':sha256_file(lock)}
    result=assess_readiness(tmp_path,evidence)
    assert not result['roles']['classifier'] and not result['roles']['diffusion']
    assert result['checks']['V05'] == 'evidence_present'
    assert not result['roles']['dino'] and result['checks']['V06']=='pending'
    (tmp_path/'deployment/V05/evidence.json').write_text('changed')
    with pytest.raises(ValueError): assess_readiness(tmp_path,evidence)


def test_snapshot_rejects_unlisted_files(tmp_path):
    from geobench.preflight import verify_snapshot
    import hashlib
    (tmp_path/'a').write_text('a'); (tmp_path/'b').write_text('b')
    descriptor={'repository':'example/model','revision':'a'*40,'files':{'a':hashlib.sha256(b'a').hexdigest()}}
    with pytest.raises(ValueError): verify_snapshot(tmp_path,descriptor)


def test_preflight_cli_produces_a_locked_pending_report(tmp_path, monkeypatch, capsys):
    from geobench.cli import main
    monkeypatch.setenv('GEOBENCH_OWNER_TOKEN','fixture-owner')
    assert main(['preflight','--root',str(tmp_path)]) == 0
    lock = tmp_path/'deployment/preflight/lock.json'
    assert str(lock) in capsys.readouterr().out
    assert verify_lock(lock, boundary=tmp_path)['identity'] == 'deployment-preflight'


def test_prompt_audit_detects_overflow_without_changing_text():
    from geobench.preflight import audit_prompts
    class Tokenizer:
        def __call__(self, text, truncation=False):
            assert truncation is False
            return {'input_ids': list(range(78 if 'long' in text else 3))}
    result=audit_prompts(Tokenizer(), {'base':['normal'], 'geo':['long prompt']})
    assert result['overflow_roles']==['geo']
    assert result['records'][1]['text']=='long prompt'
