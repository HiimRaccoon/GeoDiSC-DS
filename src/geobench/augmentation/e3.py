"""CADS reference and final ablation generation through the shared sampler."""
import csv
import json
from pathlib import Path
from ..artifacts import _inside, sha256_file, verify_lock
from .diffusion import generate_pool


def generate_e3(root, descriptor, *, variant, reference=False, owner_token):
    if variant not in ('A1','A2','A3') or type(reference) is not bool or (reference and variant != 'A1'):
        raise ValueError('invalid E3 generation role')
    root=Path(root).resolve()
    selection=root/'tuning/E2/selection/selection_lock.json'
    selected=verify_lock(selection,boundary=root)
    if selected['identity'] != 'selected-strength' or 'selected_strength.json' not in selected['files']:
        raise ValueError('selected strength required')
    strength=json.loads((selection.parent/'selected_strength.json').read_text())['selected_strength']
    if descriptor.get('strength') != strength:
        raise ValueError('E3 must use the selected strength')
    dependencies={selection.relative_to(root).as_posix():sha256_file(selection)}
    def manifest(lock_path, identity):
        lock=verify_lock(lock_path,boundary=root)
        if lock['identity'] != identity or 'manifest.csv' not in lock['files']:
            raise ValueError('invalid cross-role generation dependency')
        dependencies[lock_path.relative_to(root).as_posix()]=sha256_file(lock_path)
        with (lock_path.parent/'manifest.csv').open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            from ..images import decoded_pixel_sha256
            path=_inside(root,row['relative_path'])
            if sha256_file(path) != row['sha256'] or decoded_pixel_sha256(path) != row['decoded_pixel_sha256']:
                raise ValueError('cross-role image content mismatch')
        return rows
    if reference:
        excluded=manifest(root/f'pools/E2/strength_{strength}/pool_lock.json',f'e2:{strength}')
        prefix='augmentation/E3/A2/references';identity='reference-bank'
    else:
        excluded=manifest(root/'augmentation/E3/A2/references/bank_lock.json','reference-bank')
        prefix=f'augmentation/E3/{variant}';identity='mining' if variant=='A2' else f'e3-generation:{variant}'
    confusers={}
    if variant in ('A2','A3'):
        path=root/'augmentation/E3/A2/mining_lock.json'
        lock=verify_lock(path,boundary=root)
        if lock['identity'] != 'mining' or 'confusers.json' not in lock['files']:
            raise ValueError('cached confusers required')
        bank='augmentation/E3/A2/references/bank_lock.json'
        if lock['dependencies'].get(bank) != dependencies[bank]:
            raise ValueError('confuser bank mismatch')
        dependencies[path.relative_to(root).as_posix()]=sha256_file(path)
        mapping=json.loads((path.parent/'confusers.json').read_text())
        with (root/'splits/class_groups.csv').open(newline='',encoding='utf-8') as stream:
            names={int(row['class_id']):row['class_name'] for row in csv.DictReader(stream)}
        for target, other in mapping.items():
            if int(target)==int(other) or int(target) not in names or int(other) not in names:
                raise ValueError('invalid confuser class')
            confusers[str(target)]=names[int(other)]
        if set(confusers) != {str(key) for key in names}:
            raise ValueError('incomplete confuser mapping')
    from . import guidance
    plan={'variant':variant,'role':'reference' if reference else 'final','prefix':prefix,'identity':identity,
          'code_sha256':sha256_file(__file__),'guidance_sha256':sha256_file(guidance.__file__),
          'dependencies':dependencies,'confusers':confusers,
          'excluded_files':sorted({r['sha256'] for r in excluded}),
          'excluded_pixels':sorted({r['decoded_pixel_sha256'] for r in excluded})}
    return generate_pool(root,descriptor,owner_token=owner_token,_e3=plan)


def build_e3_pool(root, variant, *, owner_token):
    import hashlib, io
    from ..artifacts import active_writer, claim_writer, staging_path, complete_temporary, publish_lock
    if variant not in ('A1','A2','A3') or not isinstance(owner_token,str) or not owner_token:
        raise ValueError('variant and owner token required')
    root=Path(root).resolve()
    paths={'data':'splits/lock.json','e0':'pools/E0/pool_lock.json','e1':'pools/E1/pool_lock.json',
           'strength':'tuning/E2/selection/selection_lock.json','references':'augmentation/E3/A2/references/bank_lock.json',
           'generation':f'augmentation/E3/{variant}/lock.json'}
    if variant in ('A2','A3'): paths['mining']='augmentation/E3/A2/mining_lock.json'
    locks={role:verify_lock(root/path,boundary=root) for role,path in paths.items()}
    dependencies={path:sha256_file(root/path) for path in paths.values()}
    if locks['generation']['identity'] != ('mining' if variant=='A2' else f'e3-generation:{variant}'):
        raise ValueError('generation identity mismatch')
    for role in ('data','e0','e1','strength','references') + (('mining',) if variant in ('A2','A3') else ()):
        if locks['generation']['dependencies'].get(paths[role]) != dependencies[paths[role]]:
            raise ValueError('generation lineage mismatch')
    def read(path):
        with path.open(newline='',encoding='utf-8') as stream: return list(csv.DictReader(stream))
    real=read(root/'splits/train.csv');traditional=read(root/'pools/E1/manifest.csv')
    synthetic=read(root/f'augmentation/E3/{variant}/manifest.csv')
    groups=read(root/'splits/class_groups.csv');sources={r['image_id']:r for r in real}
    from .jobs import validate_training_groups
    validate_training_groups(real,groups)
    bank=read(root/'augmentation/E3/A2/references/manifest.csv')
    excluded_files={r['sha256'] for r in bank};excluded_pixels={r['decoded_pixel_sha256'] for r in bank}
    for records, kind in ((traditional,'traditional'),(synthetic,'final_synthetic')):
        for row in records:
            source=sources.get(row.get('source_id'))
            if source is None or row['sample_type']!=kind or source['class_id']!=row['class_id']:
                raise ValueError('augmented source membership mismatch')
            from ..images import decoded_pixel_sha256
            path=_inside(root,row['relative_path'])
            if sha256_file(path)!=row['sha256'] or decoded_pixel_sha256(path)!=row['decoded_pixel_sha256']:
                raise ValueError('augmented content mismatch')
            if kind=='final_synthetic' and (row['sha256'] in excluded_files or row['decoded_pixel_sha256'] in excluded_pixels):
                raise ValueError('reference leaked into final pool')
    for group in groups:
        c=group['class_id']; n=int(group['n_train']); original=int(group['n_original'])
        if sum(r['class_id']==c for r in traditional)!=(100-n if 20<=n<100 else 0):
            raise ValueError('E1 quota mismatch')
        if sum(r['class_id']==c for r in synthetic)!=(min(100,5*n)-n if n<20 and original>1 else 0):
            raise ValueError('E3 quota mismatch')
    rows=[{**r,'sample_type':'real'} for r in real]+traditional+synthetic
    rows.sort(key=lambda r:(int(r['class_id']),r['image_id']))
    if len({r['image_id'] for r in rows})!=len(rows): raise ValueError('duplicate pool IDs')
    fields=sorted(set().union(*(r.keys() for r in rows)))
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    target=root/f'pools/E3-{variant}';identity=f'pool:E3-{variant}'
    value=hashlib.sha256((identity+'\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,identity,token=value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token);complete_temporary(stage/'manifest.csv',out.getvalue().encode())
            info={'variant':variant,'real_count':len(real),'traditional_count':len(traditional),'synthetic_count':len(synthetic),
                  'reference_count':0,'manifest_sha256':sha256_file(stage/'manifest.csv')}
            complete_temporary(stage/'pool.json',(json.dumps(info,sort_keys=True)+'\n').encode())
            return publish_lock(target,{'identity':identity,'files':['manifest.csv','pool.json'],'dependencies':dependencies,
                                        'artifact_root':root,'claim':token,'lock_name':'pool_lock.json'})
        finally:
            token.close()
