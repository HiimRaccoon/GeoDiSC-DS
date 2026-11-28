import json
import pytest
from geobench.artifacts import sha256_file,verify_lock
from geobench.deployment import stage_model,prepare_probe


def test_model_materialization_and_locked_probe_inputs(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'weights.pth').write_bytes(b'verified fixture weights')
    descriptor={'repository':'fixture/efficientnet','revision':'a'*40,'files':{'weights.pth':sha256_file(source/'weights.pth')}}
    root=tmp_path/'artifacts'
    lock=stage_model(root,'classifier',source,descriptor,owner_token='fixture')
    assert verify_lock(lock,boundary=root)['identity']=='classifier-weights'
    assert (lock.parent/'weights.pth').read_bytes()==(source/'weights.pth').read_bytes()
    assert stage_model(root,'classifier',source,descriptor,owner_token='fixture')==lock
    probe=prepare_probe(root,'classifier',lock.relative_to(root).as_posix(),owner_token='fixture')
    assert verify_lock(probe,boundary=root)['dependencies'][lock.relative_to(root).as_posix()]==sha256_file(lock)
    with pytest.raises(ValueError):prepare_probe(root,'dino',lock.relative_to(root).as_posix(),owner_token='fixture')
    (source/'weights.pth').write_bytes(b'changed')
    with pytest.raises(ValueError):stage_model(root,'classifier',source,descriptor,owner_token='fixture')


def test_site_inspection_uses_private_inputs_and_read_only_commands(tmp_path,monkeypatch):
    from site_evidence import publish_site
    from geobench.preflight import inspect_site
    publish_site(tmp_path);calls=[]
    def fake(argv,**kwargs):
        calls.append(argv);return type('Result',(),{'returncode':0,'stdout':'fixture observation','stderr':''})()
    monkeypatch.setattr('subprocess.run',fake)
    path=inspect_site(tmp_path,owner_token='fixture')
    record=json.loads((path.parent/'inspection.json').read_text())
    assert record['status']=='observed' and len(calls)==4
    assert all(argv[:5]==['ssh','-o','BatchMode=yes','-p','22'] for argv in calls)
    assert all('user@fixture.invalid' in argv for argv in calls)
