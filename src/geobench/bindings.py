"""Official descriptors and immutable whole-triplet promotions."""
import json
from pathlib import Path
from .artifacts import _inside, active_writer, complete_temporary, publish_records, sha256_file, verify_lock

CONDITIONS=('E0','E1E2','E3-A1','E3-A2','E3-A3','E3-A4')
SEEDS=(42,123,2026)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def reference(root,path):
    lock=verify_lock(_inside(root,path),boundary=root)
    return {'path':path,'sha256':sha256_file(Path(root)/path),'identity':lock['identity']}


def prepare_run(root, run_id, *, owner_token):
    from .config import resolve_run
    root=Path(root).resolve()
    parts=run_id.split('/')
    if len(parts)!=2 or parts[0] not in CONDITIONS or parts[1] not in tuple(map(str,SEEDS)):
        raise ValueError('invalid official identity')
    condition=parts[0]; seed=int(parts[1])
    paths={'data':'splits/lock.json','runtime':'runtime/lock.json','initialization':f'initializations/seed_{seed}/lock.json',
           'spec':f'specs/{condition}/lock.json','pool':f'pools/{"E3-A3" if condition=="E3-A4" else condition}/pool_lock.json'}
    if condition!='E0': paths.update(strength='tuning/E2/selection/selection_lock.json',e1='pools/E1/pool_lock.json')
    if condition.startswith('E3-'): paths['references']='augmentation/E3/A2/references/bank_lock.json'
    if condition in ('E3-A2','E3-A3','E3-A4'): paths['mining']='augmentation/E3/A2/mining_lock.json'
    refs={role:reference(root,path) for role,path in paths.items()}
    descriptor={'run_id':run_id,'condition':condition,'seed':seed,'dependencies':refs}
    target=root/f'runs/{condition}/seed_{seed}'
    # resolve_run is the authority; a mismatched existing descriptor is never replaced.
    with active_writer(target):
        complete_temporary(target/'descriptor.json',encoded(descriptor))
        resolve_run(root,run_id)
    contents={'descriptor.json':encoded(descriptor)}
    if condition=='E3-A4':
        from .training import steps_per_epoch
        import csv
        manifest=(root/refs['pool']['path']).parent/'manifest.csv'
        with manifest.open(encoding='utf-8',newline='') as stream: anchors=list(csv.DictReader(stream))
        if any(row.get('sample_type') not in ('real','traditional','final_synthetic') for row in anchors):
            raise ValueError('invalid A3 anchor membership')
        contents['pool_binding.json']=encoded({'condition':'E3-A4','canonical_condition':'E3-A3','pool':refs['pool'],
            'manifest_sha256':sha256_file(manifest),'anchors':len(anchors),'steps_per_epoch':steps_per_epoch(len(anchors)),
            'total_steps':20*steps_per_epoch(len(anchors)),'new_images':0})
    return publish_records(root,f'descriptors/{condition}/seed_{seed}',f'descriptor:{run_id}',
        contents,{r['path']:r['sha256'] for r in refs.values()},owner_token=owner_token)


def freeze_specs(root,*,owner_token):
    from .config import verified_configuration
    source=Path(__file__).resolve().parents[2]/'docs'
    mapping={'E0':'01_E0_original.md','E1E2':'02_E1E2_hybrid_baseline.md',
             **{f'E3-A{i}':'03_E3_geodisc_ds.md' for i in range(1,5)}}
    results=[]
    for condition in CONDITIONS:
        contents={name:(source/'specs'/name).read_bytes() for name in ('00_common.md','04_evaluation.md',mapping[condition])}
        contents['configuration.json']=encoded(verified_configuration())
        results.append(publish_records(root,f'specs/{condition}',f'spec:{condition}',contents,{},owner_token=owner_token))
    return results


def freeze_selected_pool(root, *, owner_token):
    root=Path(root).resolve()
    selection=reference(root,'tuning/E2/selection/selection_lock.json')
    selected=json.loads((root/selection['path']).with_name('selected_strength.json').read_text())
    from .selection import select_strength_from_locks
    result=select_strength_from_locks(root,selected['runs'])
    if result['strength']!=selected['selected_strength']: raise ValueError('selection evidence mismatch')
    s=result['strength']; source=f'pools/E1E2/candidates/strength_{s}/pool_lock.json'
    candidate=verify_lock(root/source,boundary=root)
    if candidate['identity']!=f'candidate-pool:{s}': raise ValueError('candidate pool mismatch')
    files={name:(root/source).parent.joinpath(name).read_bytes() for name in ('manifest.csv','pool.json')}
    deps={**candidate['dependencies'],selection['path']:selection['sha256'],source:sha256_file(root/source)}
    return publish_records(root,'pools/E1E2','pool:E1E2',files,deps,owner_token=owner_token,lock_name='pool_lock.json')


def bind_selected_runs(root, *, owner_token):
    """Verify and promote all three selected executions without copying weights."""
    import torch
    from .config import resolve_run
    from .selection import select_strength_from_locks
    root=Path(root).resolve(); selection=reference(root,'tuning/E2/selection/selection_lock.json')
    if (root/'runs/E1E2/plan/lock.json').exists():raise ValueError('fresh-three-seed plan already frozen')
    selected=json.loads((root/selection['path']).with_name('selected_strength.json').read_text())
    winner=select_strength_from_locks(root,selected['runs'])['strength']
    if winner!=selected['selected_strength']: raise ValueError('selection mismatch')
    pool=reference(root,'pools/E1E2/pool_lock.json')
    contents={}; deps={selection['path']:selection['sha256'],pool['path']:pool['sha256']}
    for seed in SEEDS:
        descriptor=resolve_run(root,f'E1E2/{seed}')
        source=next(r for r in selected['runs'] if r['strength']==winner and r['seed']==seed)
        path=_inside(root,source['path']); lock=verify_lock(path,boundary=root)
        if any('test' in name for name in lock['files']): raise ValueError('tuning run contains test evidence')
        if not {'last_checkpoint.pt','run_descriptor.json','history.json','lr_steps.csv'} <= set(lock['files']):
            raise ValueError('incomplete reuse evidence')
        saved=torch.load(path.parent/'last_checkpoint.pt',map_location='cpu',weights_only=True)
        history=json.loads((path.parent/'history.json').read_text())
        if (saved.get('schema_version')!=3 or saved.get('identity')!=lock['identity'] or saved.get('completed_epoch')!=20 or saved.get('history')!=history
                or saved.get('global_step')!=20*saved.get('steps_per_epoch',0) or saved.get('mixup_seed') is not None):
            raise ValueError('reuse checkpoint progress mismatch')
        for role in ('data','runtime','initialization','spec','e1'):
            ref=descriptor['dependencies'][role]
            if lock['dependencies'].get(ref['path'])!=ref['sha256']: raise ValueError('reuse dependency mismatch')
        candidate=f'pools/E1E2/candidates/strength_{winner}/pool_lock.json'
        if sha256_file(root/'pools/E1E2/manifest.csv')!=sha256_file((root/candidate).parent/'manifest.csv'):
            raise ValueError('official candidate membership mismatch')
        import csv
        from .training import steps_per_epoch,learning_rate
        with (root/'pools/E1E2/manifest.csv').open(encoding='utf-8',newline='') as stream: anchors=list(csv.DictReader(stream))
        steps=steps_per_epoch(len(anchors))
        if (saved['steps_per_epoch']!=steps or any(row.get('samples')!=len(anchors) or row.get('optimizer_steps')!=steps for row in history)
                or len(saved['lr_history'])!=20*steps or any(row['lr']!=learning_rate(i,steps) for i,row in enumerate(saved['lr_history']))):
            raise ValueError('reuse traversal or scheduler mismatch')
        best=json.loads((path.parent/'validation_at_best.json').read_text())
        if best['epoch']!=saved['best_epoch'] or best['macro_f1']!=saved['best_score']:
            raise ValueError('best checkpoint mismatch')
        deps[source['path']]=source['sha256']
        contents[f'seed_{seed}/official_binding.json']=encoded({'run_id':f'E1E2/{seed}','mode':'reused',
            'source':source,'checkpoint_sha256':lock['files']['best_checkpoint.pt'],'dependencies':descriptor['dependencies']})
    return publish_records(root,'runs/E1E2','official-bindings:E1E2',contents,deps,owner_token=owner_token)


def plan_fresh_e1e2(root,reason,*,owner_token):
    from .config import resolve_run
    root=Path(root).resolve()
    if not isinstance(reason,str) or not reason.strip():raise ValueError('fresh-triplet reason required')
    if (root/'runs/E1E2/lock.json').exists():raise ValueError('reused triplet already frozen')
    descriptors=[resolve_run(root,f'E1E2/{seed}') for seed in SEEDS]
    deps={ref['path']:ref['sha256'] for descriptor in descriptors for ref in descriptor['dependencies'].values()}
    return publish_records(root,'runs/E1E2/plan','fresh-triplet:E1E2',
        {'plan.json':encoded({'mode':'fresh','seeds':list(SEEDS),'reason':reason,'executions':27})},deps,owner_token=owner_token)
