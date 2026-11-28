import json
import pytest
from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock


def test_tuning_protocol_freezes_nine_shared_run_identities(tmp_path):
    from geobench.tuning import prepare_tuning, resolve_tuning
    def make(path, identity, deps=None, name='lock.json'):
        dest=tmp_path/path;token=claim_writer(dest,identity);stage=staging_path(dest,token)
        (stage/'record.json').write_text('{}')
        lock=publish_lock(dest,{'identity':identity,'files':['record.json'],'dependencies':deps or {},
                                'artifact_root':tmp_path,'claim':token,'lock_name':name})
        return lock.relative_to(tmp_path).as_posix(),sha256_file(lock)
    data=make('splits','common-data:'+'a'*64)
    runtime=make('runtime','shared-runtime')
    e0=make('pools/E0','e0:'+'a'*64,{data[0]:data[1]},'pool_lock.json')
    e1=make('pools/E1','e1',{data[0]:data[1],e0[0]:e0[1]},'pool_lock.json')
    make('specs/E1E2','spec:E1E2')
    for seed in (42,123,2026):
        make(f'initializations/seed_{seed}',f'initialization:{seed}',{runtime[0]:runtime[1]})
    for strength in (.25,.35,.5):
        make(f'pools/E1E2/candidates/strength_{strength}',f'candidate-pool:{strength}',
             {data[0]:data[1],e0[0]:e0[1],e1[0]:e1[1]},'pool_lock.json')
    lock=prepare_tuning(tmp_path,owner_token='fixture')
    assert verify_lock(lock,boundary=tmp_path)['identity']=='tuning-protocol:E2'
    protocol=json.loads((lock.parent/'protocol.json').read_text())
    assert len(protocol['runs'])==9
    for strength in (.25,.35,.5):
        for seed in (42,123,2026):
            descriptor=resolve_tuning(tmp_path,strength,seed)
            assert descriptor['strength']==strength and descriptor['seed']==seed
            assert descriptor['dependencies']['pool']['identity']==f'candidate-pool:{strength}'
    with pytest.raises(ValueError): resolve_tuning(tmp_path,.4,42)
    assert prepare_tuning(tmp_path,owner_token='fixture')==lock
