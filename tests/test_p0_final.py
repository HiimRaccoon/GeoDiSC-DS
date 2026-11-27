import copy, hashlib, json
from pathlib import Path
import pytest
from geobench import artifacts as a
from geobench.config import verified_configuration
from geobench.data import prepare_data


def descriptor(token, lock_name='lock.json'):
    return {'identity':'job','files':['out'],'dependencies':{},'claim':token,'lock_name':lock_name}


@pytest.mark.parametrize('temporary', ['copy', 'lock'])
def test_partial_temporary_publication_can_resume(tmp_path, temporary):
    token = a.claim_writer(tmp_path, 'job', token='owner')
    stage = a.staging_path(tmp_path, token); (stage/'out').write_bytes(b'complete')
    if temporary == 'copy':
        copies=stage/'.publishing'; copies.mkdir()
        (copies/hashlib.sha256(b'out').hexdigest()).write_bytes(b'com')
    else:
        (tmp_path/'.lock.staging').write_bytes(b'{"art')
    lock=a.publish_lock(tmp_path,descriptor(token))
    assert a.verify_lock(lock)['files']['out'] == a.sha256_file(tmp_path/'out')


def test_recovery_does_not_replace_published_mismatch(tmp_path):
    token=a.claim_writer(tmp_path,'job'); stage=a.staging_path(tmp_path,token)
    (stage/'out').write_bytes(b'expected'); (tmp_path/'out').write_bytes(b'wrong')
    with pytest.raises(ValueError): a.publish_lock(tmp_path,descriptor(token))
    assert (tmp_path/'out').read_bytes() == b'wrong'


@pytest.mark.parametrize('change', ['boolean_type', 'deployment_extra', 'deployment_type', 'extra_role'])
def test_programmatic_configuration_is_as_strict_as_cli(tmp_path, change):
    snapshot=copy.deepcopy(verified_configuration())
    if change=='boolean_type': snapshot['values']['benchmark']['classifier']['amp']=0
    if change=='deployment_extra': snapshot['values']['deployment']['batch_size']=64
    if change=='deployment_type': snapshot['values']['deployment']['cpu_count']='four'
    if change=='extra_role': snapshot['values']['extra']={}
    snapshot['sha256']={name:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest() for name,value in snapshot['values'].items()}
    if change != 'extra_role':
        role = 'benchmark' if change == 'boolean_type' else 'deployment'
        path = tmp_path/'invalid.json'; path.write_text(json.dumps(snapshot['values'][role]))
        with pytest.raises(ValueError):
            verified_configuration(**{role:path})
    with pytest.raises(ValueError):
        prepare_data(tmp_path/'absent',tmp_path/'root',owner_token='owner',configuration=snapshot)
    assert not (tmp_path/'root').exists()


@pytest.mark.parametrize('name',['checkpoint_lock.json','selection_lock.json','bank_lock.json','feature_lock.json','mining_lock.json','evaluation_lock.json','report_lock.json'])
def test_authoritative_specification_lock_names(tmp_path,name):
    token=a.claim_writer(tmp_path,'job'); (a.staging_path(tmp_path,token)/'out').write_bytes(b'complete')
    lock=a.publish_lock(tmp_path,descriptor(token,name))
    assert lock.name==name and a.verify_lock(lock)['identity']=='job'
    consumer=tmp_path/'consumer'; owner=a.claim_writer(consumer,'consumer')
    (a.staging_path(consumer,owner)/'out').write_bytes(b'consumer')
    child=a.publish_lock(consumer,{'identity':'consumer','files':['out'],'dependencies':{name:a.sha256_file(lock)},'artifact_root':tmp_path,'claim':owner})
    assert a.verify_lock(child,boundary=tmp_path)['dependencies'][name] == a.sha256_file(lock)


def test_source_contains_no_private_site_constants():
    for path in Path('src').rglob('*.py'):
        text=path.read_text(encoding='utf-8-sig')
        import re
        assert not re.search(r'10\.204\.\d+\.\d+|/datastore/bin\w+/|login01\.cm\.cluster',text),path



def test_private_site_is_required_and_tampering_is_rejected(tmp_path):
    from geobench.data import _private_site
    from site_evidence import publish_site
    with pytest.raises(ValueError): _private_site(tmp_path)
    path=publish_site(tmp_path)
    site, lock=_private_site(tmp_path)
    assert lock==path and site['workspace']=='/fixture/workspace/'
    (path.parent/'site.json').write_text('{}')
    with pytest.raises(ValueError): _private_site(tmp_path)


def test_unrelated_temporary_content_is_preserved(tmp_path):
    token=a.claim_writer(tmp_path,'job'); stage=a.staging_path(tmp_path,token)
    (stage/'out').write_bytes(b'expected')
    temporary=tmp_path/'.lock.staging'; temporary.write_bytes(b'foreign')
    with pytest.raises(ValueError): a.publish_lock(tmp_path,descriptor(token))
    assert temporary.read_bytes()==b'foreign' and not (tmp_path/'lock.json').exists()
