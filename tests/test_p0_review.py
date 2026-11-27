import json, os, subprocess, sys
from pathlib import Path
import pytest
from geobench import artifacts as a
from geobench.cli import main
from geobench.config import resolve_run


def test_staging_lock_is_not_authoritative(tmp_path):
    token = a.claim_writer(tmp_path, 'job')
    (a.staging_path(tmp_path, token) / 'out').write_text('x')
    a.publish_lock(tmp_path, {'identity':'job','files':['out'],'dependencies':{},'claim':token})
    with pytest.raises(ValueError):
        a.verify_lock(tmp_path / '.lock.staging')


def test_same_token_process_cannot_enter_active_writer(tmp_path):
    token = a.claim_writer(tmp_path, 'job')
    with a.active_writer(tmp_path):
        code = 'from geobench.artifacts import claim_writer; import sys; claim_writer(sys.argv[1], "job", token=sys.argv[2], resume=True)'
        result = subprocess.run([sys.executable, '-S', '-c', code, str(tmp_path), token], capture_output=True)
        assert result.returncode != 0
    token.close()
    resumed = a.claim_writer(tmp_path, 'job', token=token, resume=True)
    assert resumed == token
    resumed.close()


def test_cli_rejects_modified_config_before_dataset_read(tmp_path):
    from geobench.config import RESEARCH
    config = json.loads(json.dumps(RESEARCH)); config['classifier']['batch_size'] = 1
    path = tmp_path/'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='research config'):
        main(['prepare-data','--root',str(tmp_path/'root'),'--dataset-root',str(tmp_path/'absent'),'--benchmark-config',str(path)])


@pytest.mark.parametrize('condition,seed', [('E2',42),('E0',1),('A0',42)])
def test_invalid_official_descriptor(tmp_path, condition, seed):
    path = tmp_path/'runs'/condition/f'seed_{seed}'; path.mkdir(parents=True)
    (path/'descriptor.json').write_text(json.dumps({'run_id':f'{condition}/{seed}','condition':condition,'seed':seed,'dependencies':{}}))
    with pytest.raises(ValueError):
        resolve_run(tmp_path, f'{condition}/{seed}')


def test_readiness_producer_interface_exists():
    from geobench.data import publish_readiness
    assert callable(publish_readiness)


def test_no_alias_paths_in_artifact_manifest(tmp_path):
    token = a.claim_writer(tmp_path, 'job')
    stage = a.staging_path(tmp_path, token); (stage/'out').write_text('x'); (stage/'sub').mkdir()
    with pytest.raises(ValueError):
        a.publish_lock(tmp_path, {'identity':'job','files':['sub/../out'],'dependencies':{},'claim':token})


def test_descriptor_rejects_wrong_condition_and_missing_contract(tmp_path):
    path = tmp_path/'runs/E0/seed_42'; path.mkdir(parents=True)
    for record in [ {'run_id':'E0/42','condition':'E1E2','seed':42,'dependencies':{}},
                    {'run_id':'E0/42','condition':'E0','seed':42,'dependencies':{'data':{}}} ]:
        (path/'descriptor.json').write_text(json.dumps(record))
        with pytest.raises(ValueError): resolve_run(tmp_path, 'E0/42')


def test_claim_holds_lease_until_explicit_close(tmp_path):
    token = a.claim_writer(tmp_path, 'job')
    code = 'from geobench.artifacts import claim_writer; import sys; claim_writer(sys.argv[1], "job", token=sys.argv[2], resume=True)'
    args = [sys.executable, '-S', '-c', code, str(tmp_path), str(token)]
    assert subprocess.run(args, capture_output=True).returncode != 0
    token.close()
    assert subprocess.run(args, capture_output=True).returncode == 0


def test_process_exit_releases_lease_without_reclaiming_claim(tmp_path):
    code = 'from geobench.artifacts import claim_writer; import sys,os; t=claim_writer(sys.argv[1], "job", token="known"); os._exit(0)'
    assert subprocess.run([sys.executable,'-S','-c',code,str(tmp_path)],capture_output=True).returncode == 0
    token = a.claim_writer(tmp_path,'job',token='known',resume=True)
    token.close()
    with pytest.raises((ValueError,FileExistsError)):
        a.claim_writer(tmp_path,'other')


def test_public_docs_preserve_specifications_without_private_identifiers():
    import re
    docs = Path('docs')
    assert (docs/'tasks.md').is_file() and (docs/'plan.md').is_file()
    assert len(list((docs/'specs').glob('*.md'))) == 5
    for path in docs.rglob('*.md'):
        text = path.read_text(encoding='utf-8-sig')
        assert not re.search(r'10\.204\.\d+\.\d+|/datastore/bin\w+/|BEGIN (?:OPENSSH |RSA )?PRIVATE KEY', text), path



def test_same_thread_cannot_resume_an_independent_live_claim(tmp_path):
    token = a.claim_writer(tmp_path, 'job')
    with pytest.raises(ValueError, match='already active'):
        a.claim_writer(tmp_path, 'job', token=token, resume=True)
    token.close()


def test_prepare_requires_recoverable_owner_before_mutation(tmp_path):
    from geobench.data import prepare_data
    target = tmp_path/'uncreated'
    with pytest.raises(ValueError, match='owner_token'):
        prepare_data(tmp_path/'dataset', target)
    assert not target.exists()


def test_symlink_cannot_promote_staging_lock(tmp_path):
    target = tmp_path/'.lock.staging'; target.write_text('{}')
    link = tmp_path/'lock.json'
    try: link.symlink_to(target)
    except OSError: pytest.skip('symlink privilege unavailable on this host')
    with pytest.raises(ValueError, match='authoritative'):
        a.verify_lock(link)
