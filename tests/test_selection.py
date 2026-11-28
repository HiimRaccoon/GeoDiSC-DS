import pytest
from geobench.selection import select_strength


def records():
    return [{'strength':strength, 'seed':seed, 'tail_f1':score, 'complete':True}
            for strength,score in ((.25,.4),(.35,.5),(.5,.5)) for seed in (42,123,2026)]


def test_selects_three_seed_mean_with_smaller_strength_tie():
    result=select_strength(records())
    assert result['strength']==.35
    assert result['means']=={.25:.4,.35:.5,.5:.5}


def test_missing_duplicate_failed_and_nonfinite_records_block_selection():
    samples=records()
    with pytest.raises(ValueError): select_strength(samples[:-1])
    with pytest.raises(ValueError): select_strength(samples+[samples[0]])
    failed=records();failed[0]['complete']=False
    with pytest.raises(ValueError): select_strength(failed)
    invalid=records();invalid[0]['tail_f1']=float('nan')
    with pytest.raises(ValueError): select_strength(invalid)


def test_selection_requires_locked_best_overall_checkpoint_evidence(tmp_path):
    import json
    from geobench.artifacts import claim_writer,staging_path,publish_lock,sha256_file,verify_lock
    from geobench.selection import select_strength_from_locks, publish_strength_selection
    refs=[]
    def lock_fixture(path, identity, name='lock.json'):
        folder=tmp_path/path; token=claim_writer(folder,identity); stage=staging_path(folder,token)
        (stage/'record.json').write_text('{}')
        lock=publish_lock(folder,{'identity':identity,'files':['record.json'],'dependencies':{},'claim':token,'lock_name':name})
        return {lock.relative_to(tmp_path).as_posix():sha256_file(lock)}
    runtime=lock_fixture('runtime','shared-runtime')
    initial={seed:lock_fixture(f'initializations/seed_{seed}',f'initialization:{seed}') for seed in (42,123,2026)}
    for strength in (.25,.35,.5):
        pool=lock_fixture(f'pools/E1E2/candidates/strength_{strength}',f'candidate-pool:{strength}','pool_lock.json')
        for seed in (42,123,2026):
            folder=tmp_path/'tuning'/str(strength)/str(seed)
            token=claim_writer(folder,f'tuning:E2:{strength}:{seed}');stage=staging_path(folder,token)
            history=[{'epoch':i,'macro_f1':.8 if i==2 else .5,'tail_f1':.2 if strength==.25 else .4} for i in range(1,21)]
            (stage/'history.json').write_text(json.dumps(history))
            (stage/'best_checkpoint.pt').write_bytes(b'checkpoint')
            (stage/'validation_at_best.json').write_text(json.dumps({'epoch':2,'macro_f1':.8,'tail_f1':.2 if strength==.25 else .4,'checkpoint_sha256':sha256_file(stage/'best_checkpoint.pt')}))
            lock=publish_lock(folder,{'identity':f'tuning:E2:{strength}:{seed}','files':['history.json','validation_at_best.json','best_checkpoint.pt'],'dependencies':{**runtime,**initial[seed],**pool},'artifact_root':tmp_path,'claim':token})
            refs.append({'strength':strength,'seed':seed,'path':lock.relative_to(tmp_path).as_posix(),'sha256':sha256_file(lock)})
    result=select_strength_from_locks(tmp_path,refs)
    assert result['strength']==.35
    first=tmp_path/refs[0]['path']
    bad=tmp_path/'bad'; token=claim_writer(bad,'tuning:E2:0.25:42');stage=staging_path(bad,token)
    for name in ('history.json','best_checkpoint.pt'):
        (stage/name).write_bytes((first.parent/name).read_bytes())
    selected=json.loads((first.parent/'validation_at_best.json').read_text());selected['tail_f1']=.9
    (stage/'validation_at_best.json').write_text(json.dumps(selected))
    original=verify_lock(first,boundary=tmp_path)
    bad_lock=publish_lock(bad,{'identity':'tuning:E2:0.25:42','files':['history.json','validation_at_best.json','best_checkpoint.pt'],
                          'dependencies':original['dependencies'],'artifact_root':tmp_path,'claim':token})
    invalid=[{**refs[0],'path':bad_lock.relative_to(tmp_path).as_posix(),'sha256':sha256_file(bad_lock)},*refs[1:]]
    with pytest.raises(ValueError,match='tail'): select_strength_from_locks(tmp_path,invalid)
    selection_lock=publish_strength_selection(tmp_path,refs,owner_token='fixture')
    assert verify_lock(selection_lock,boundary=tmp_path)['identity']=='selected-strength'
    assert len(verify_lock(selection_lock,boundary=tmp_path)['dependencies'])==9
    first=tmp_path/refs[0]['path']; (first.parent/'history.json').write_text('[]')
    with pytest.raises(ValueError): select_strength_from_locks(tmp_path,refs)
