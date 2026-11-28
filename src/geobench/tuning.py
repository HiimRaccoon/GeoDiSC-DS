"""Immutable shared protocol for the nine E2 strength-search runs."""
import hashlib
import json
from pathlib import Path

from .artifacts import active_writer, claim_writer, complete_temporary, publish_lock, sha256_file, staging_path, verify_lock

STRENGTHS = (.25, .35, .5)
SEEDS = (42, 123, 2026)


def prepare_tuning(root, *, owner_token):
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    root=Path(root).resolve(strict=True)
    def reference(path, expected=None):
        lock=verify_lock(root/path,boundary=root)
        if expected is not None and lock['identity'] != expected:
            raise ValueError('tuning dependency identity mismatch')
        return {'path':path,'sha256':sha256_file(root/path),'identity':lock['identity']}
    data=reference('splits/lock.json')
    if not data['identity'].startswith('common-data:') or len(data['identity'].removeprefix('common-data:')) != 64:
        raise ValueError('invalid tuning data identity')
    fingerprint=data['identity'].removeprefix('common-data:')
    runtime=reference('runtime/lock.json','shared-runtime')
    e0=reference('pools/E0/pool_lock.json','e0:'+fingerprint)
    e1=reference('pools/E1/pool_lock.json','e1')
    spec=reference('specs/E1E2/lock.json','spec:E1E2')
    runs=[]; all_refs=[data,runtime,e0,e1,spec]
    for strength in STRENGTHS:
        pool=reference(f'pools/E1E2/candidates/strength_{strength}/pool_lock.json',f'candidate-pool:{strength}')
        lock=verify_lock(root/pool['path'],boundary=root)
        if any(lock['dependencies'].get(ref['path']) != ref['sha256'] for ref in (data,e0,e1)):
            raise ValueError('tuning pool lineage mismatch')
        all_refs.append(pool)
        for seed in SEEDS:
            initialization=reference(f'initializations/seed_{seed}/lock.json',f'initialization:{seed}')
            init_lock=verify_lock(root/initialization['path'],boundary=root)
            if init_lock['dependencies'].get(runtime['path']) != runtime['sha256']:
                raise ValueError('initialization runtime mismatch')
            all_refs.append(initialization)
            runs.append({'strength':strength,'seed':seed,'dependencies':{'data':data,'pool':pool,'runtime':runtime,
                         'initialization':initialization,'spec':spec,'e1':e1}})
    protocol={'strengths':list(STRENGTHS),'seeds':list(SEEDS),'selection':'mean_tail_f1_at_overall_best',
              'tie_break':'smallest_strength','runs':runs}
    dependencies={ref['path']:ref['sha256'] for ref in all_refs}
    target=root/'tuning/E2'
    token_value=hashlib.sha256(('tuning-protocol\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,'tuning-protocol:E2',token=token_value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token)
            complete_temporary(stage/'protocol.json',(json.dumps(protocol,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
            return publish_lock(target,{'identity':'tuning-protocol:E2','files':['protocol.json'],'dependencies':dependencies,
                                        'artifact_root':root,'claim':token})
        finally:
            token.close()


def resolve_tuning(root, strength, seed):
    if type(strength) is not float or strength not in STRENGTHS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('invalid tuning identity')
    root=Path(root).resolve(strict=True); path=root/'tuning/E2/lock.json'
    lock=verify_lock(path,boundary=root)
    if lock['identity'] != 'tuning-protocol:E2' or 'protocol.json' not in lock['files']:
        raise ValueError('invalid tuning protocol')
    protocol=json.loads((path.parent/'protocol.json').read_text(encoding='utf-8'))
    runs=protocol.get('runs',[])
    if (protocol.get('strengths') != list(STRENGTHS) or protocol.get('seeds') != list(SEEDS)
            or protocol.get('selection') != 'mean_tail_f1_at_overall_best' or protocol.get('tie_break') != 'smallest_strength'
            or len(runs) != 9 or {(run['strength'],run['seed']) for run in runs} != {(s,n) for s in STRENGTHS for n in SEEDS}):
        raise ValueError('invalid tuning protocol contents')
    run=next(item for item in runs if item['strength']==strength and item['seed']==seed)
    refs=run.get('dependencies')
    if not isinstance(refs,dict) or set(refs) != {'data','pool','runtime','initialization','spec','e1'}:
        raise ValueError('invalid tuning dependency roles')
    expected={'pool':f'candidate-pool:{strength}','runtime':'shared-runtime',
              'initialization':f'initialization:{seed}','spec':'spec:E1E2','e1':'e1'}
    for role,reference in refs.items():
        if not isinstance(reference,dict) or set(reference) != {'path','sha256','identity'}:
            raise ValueError('invalid tuning dependency record')
        if lock['dependencies'].get(reference['path']) != reference['sha256']:
            raise ValueError('tuning protocol dependency mismatch')
        if role == 'data':
            if not reference['identity'].startswith('common-data:') or len(reference['identity'].removeprefix('common-data:')) != 64:
                raise ValueError('invalid tuning data identity')
        elif reference['identity'] != expected[role]:
            raise ValueError('invalid tuning dependency identity')
        actual=verify_lock(root/reference['path'],boundary=root)
        if actual['identity'] != reference['identity']:
            raise ValueError('tuning lock identity mismatch')
    pool=verify_lock(root/refs['pool']['path'],boundary=root)
    init=verify_lock(root/refs['initialization']['path'],boundary=root)
    if (any(pool['dependencies'].get(refs[role]['path']) != refs[role]['sha256'] for role in ('data','e1'))
            or init['dependencies'].get(refs['runtime']['path']) != refs['runtime']['sha256']):
        raise ValueError('tuning run lineage mismatch')
    return run
