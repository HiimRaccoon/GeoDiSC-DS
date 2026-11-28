import json
import pytest
from geobench.artifacts import publish_records,sha256_file,verify_lock
from geobench.bindings import CONDITIONS,SEEDS,prepare_run,encoded
from geobench.config import resolve_run


def test_eighteen_descriptors_and_a4_share_exact_a3_pool(tmp_path):
    def lock(path,identity,deps=None,files=None,name='lock.json'):
        result=publish_records(tmp_path,path,identity,files or {'record.json':b'{}'},deps or {},owner_token='fixture',lock_name=name)
        return {result.relative_to(tmp_path).as_posix():sha256_file(result)}
    data=lock('splits','common-data:'+'a'*64)
    runtime=lock('runtime','shared-runtime')
    e1=lock('pools/E1','e1',data,name='pool_lock.json')
    strength=lock('tuning/E2/selection','selected-strength',name='selection_lock.json')
    bank=lock('augmentation/E3/A2/references','reference-bank',{**data,**strength},name='bank_lock.json')
    mining=lock('augmentation/E3/A2','mining',bank,name='mining_lock.json')
    for condition in CONDITIONS:
        lock(f'specs/{condition}',f'spec:{condition}')
        if condition=='E3-A4':continue
        deps=dict(data)
        if condition!='E0':deps.update({**e1,**strength})
        if condition.startswith('E3-'):deps.update(bank)
        if condition in ('E3-A2','E3-A3'):deps.update(mining)
        lock(f'pools/{condition}','e0:'+'a'*64 if condition=='E0' else f'pool:{condition}',deps,
             {'manifest.csv':b'image_id,class_id,sample_type\na,0,real\n'},'pool_lock.json')
    for seed in SEEDS:lock(f'initializations/seed_{seed}',f'initialization:{seed}',runtime)
    for condition in CONDITIONS:
        for seed in SEEDS:
            result=prepare_run(tmp_path,f'{condition}/{seed}',owner_token='fixture')
            descriptor=resolve_run(tmp_path,f'{condition}/{seed}')
            assert verify_lock(result,boundary=tmp_path)['identity']==f'descriptor:{condition}/{seed}'
            if condition=='E3-A4':
                binding=json.loads((result.parent/'pool_binding.json').read_text())
                assert binding['new_images']==0 and binding['total_steps']==20
                assert descriptor['dependencies']['pool']['path']=='pools/E3-A3/pool_lock.json'
    from geobench.bindings import plan_fresh_e1e2
    fresh=plan_fresh_e1e2(tmp_path,'reuse not possible for all three seeds',owner_token='fixture')
    record=json.loads((fresh.parent/'plan.json').read_text())
    assert record['mode']=='fresh' and record['seeds']==[42,123,2026] and record['executions']==27
