"""Official fixed-checkpoint inference and reproducible prediction exports."""
import csv
import io
import json
from pathlib import Path
import numpy as np
import torch
from .artifacts import _inside, active_writer, publish_records, sha256_file, verify_lock
from .bindings import encoded
from .metrics import compute_metrics, rank_logits


def csv_bytes(rows, fields):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def prediction_exports(rows, logits, groups, class_rows):
    """Validate exact manifest order and export full-precision raw evidence."""
    if not rows or len({r['image_id'] for r in rows})!=len(rows): raise ValueError('duplicate or empty manifest IDs')
    if rows != sorted(rows,key=lambda r:(int(r['class_id']),r['relative_path'])):
        raise ValueError('evaluation manifest order mismatch')
    scores=torch.as_tensor(logits)
    if scores.dtype!=torch.float32 or scores.shape!=(len(rows),124) or not torch.isfinite(scores).all():
        raise ValueError('expected finite FP32 [N,124] logits')
    ranking=rank_logits(scores)
    labels=[int(r['class_id']) for r in rows]
    metrics=compute_metrics(labels,scores,groups)
    if sum(sum(row) for row in metrics['confusion_matrix'])!=len(rows): raise ValueError('confusion coverage mismatch')
    predictions=[{'image_id':r['image_id'],'relative_path':r['relative_path'],'true_class_id':labels[i],
        'predicted_class_id':int(ranking[i,0]),'top5_class_ids':json.dumps(ranking[i,:5].tolist()),'logits_row':i} for i,r in enumerate(rows)]
    mapping={int(r['class_id']):r for r in class_rows}
    if set(mapping)!=set(range(124)): raise ValueError('full class row mapping required')
    per_class=[]
    for c, values in enumerate(metrics['per_class']):
        info=mapping[c]
        per_class.append({'class_id':c,'class_name':info['class_name'],'group':info['group'],
            'n_original':info['n_original'],'n_train':info['n_train'],'singleton':info['singleton'],**values})
    array=io.BytesIO();np.save(array,scores.cpu().numpy(),allow_pickle=False)
    matrix=io.StringIO(newline='');csv.writer(matrix,lineterminator='\n').writerows(metrics['confusion_matrix'])
    return {'predictions':csv_bytes(predictions,list(predictions[0])), 'logits':array.getvalue(),
            'confusion_matrix':matrix.getvalue().encode(),'per_class_metrics':csv_bytes(per_class,list(per_class[0]))},metrics


def official_checkpoint(root,run_id):
    from .config import resolve_run
    root=Path(root).resolve(); descriptor=resolve_run(root,run_id)
    condition,seed=descriptor['condition'],descriptor['seed']; target=root/f'runs/{condition}/seed_{seed}'
    own=target/'checkpoint_lock.json'
    if own.exists():
        lock=verify_lock(own,boundary=root)
        snapshot=target/'run_descriptor.json'
        if snapshot.read_bytes()!=(target/'descriptor.json').read_bytes(): raise ValueError('checkpoint descriptor mismatch')
        deps={r['path']:r['sha256'] for r in descriptor['dependencies'].values()}
        if any(lock['dependencies'].get(k)!=v for k,v in deps.items()): raise ValueError('checkpoint lineage mismatch')
        source=own; authority=own
    elif condition=='E1E2':
        authority=root/'runs/E1E2/lock.json';bindings=verify_lock(authority,boundary=root)
        if bindings['identity']!='official-bindings:E1E2' or any(f'seed_{s}/official_binding.json' not in bindings['files'] for s in (42,123,2026)):
            raise ValueError('whole-triplet binding required')
        binding=json.loads((target/'official_binding.json').read_text())
        if binding['run_id']!=run_id or binding['dependencies']!=descriptor['dependencies']: raise ValueError('binding identity mismatch')
        source=_inside(root,binding['source']['path']);lock=verify_lock(source,boundary=root)
        if sha256_file(source)!=binding['source']['sha256'] or lock['files']['best_checkpoint.pt']!=binding['checkpoint_sha256']:
            raise ValueError('binding source mismatch')
    else:
        raise ValueError('official training incomplete')
    required={'best_checkpoint.pt','last_checkpoint.pt','history.json','validation_at_best.json'}
    if not required<=set(lock['files']): raise ValueError('incomplete checkpoint evidence')
    best=json.loads((source.parent/'validation_at_best.json').read_text())
    history=json.loads((source.parent/'history.json').read_text())
    if len(history)!=20 or best['checkpoint_sha256']!=lock['files']['best_checkpoint.pt']:
        raise ValueError('checkpoint completeness mismatch')
    chosen=max(history,key=lambda r:(r['macro_f1'],-r['epoch']))
    if chosen['epoch']!=best['epoch'] or chosen['macro_f1']!=best['macro_f1']: raise ValueError('checkpoint selection mismatch')
    return descriptor,source,authority,best,lock['identity']


def _predict(model,loader):
    outputs=[]
    model=model.to(device='cuda',dtype=torch.float32).eval()
    with torch.inference_mode():
        for images,labels in loader:
            outputs.append(model(images.to(device='cuda',dtype=torch.float32)).cpu())
    return torch.cat(outputs)


def evaluate_run(root,run_id,*,owner_token):
    from .classifier import build_classifier
    from .training import _require_classifier_runtime, _rng_state, _restore_rng
    from .training_data import ImageDataset, make_loaders
    root=Path(root).resolve()
    descriptor,source,authority,best,identity=official_checkpoint(root,run_id)
    target=root/f'runs/{descriptor["condition"]}/seed_{descriptor["seed"]}'
    existing=target/'evaluation_lock.json'
    if existing.exists():
        frozen=verify_lock(existing,boundary=root)
        metadata=json.loads((target/'evaluation_metadata.json').read_text())
        if (metadata['run_id']!=run_id or metadata['checkpoint_sha256']!=sha256_file(source.parent/'best_checkpoint.pt')
                or frozen['dependencies'].get(authority.relative_to(root).as_posix())!=sha256_file(authority)):
            raise ValueError('published evaluation identity mismatch')
        return existing
    if not isinstance(owner_token,str) or not owner_token: raise ValueError('owner token required')
    _require_classifier_runtime(descriptor['seed'])
    deps={authority.relative_to(root).as_posix():sha256_file(authority)}
    refs=descriptor['dependencies'];data_path=_inside(root,refs['data']['path']);data=verify_lock(data_path,boundary=root)
    for name in ('val.csv','test.csv','class_groups.csv'):
        if name not in data['files']: raise ValueError('evaluation data not locked')
    deps[refs['data']['path']]=refs['data']['sha256']
    runtime_path=_inside(root,refs['runtime']['path']);runtime=json.loads((runtime_path.parent/'runtime.json').read_text())
    deps[refs['runtime']['path']]=refs['runtime']['sha256']
    readiness=root/'deployment/data_readiness/lock.json';verify_lock(readiness,boundary=root)
    dataset_root=json.loads((readiness.parent/'V01.json').read_text())['dataset_root']
    def read(name):
        with (data_path.parent/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
    classes=read('class_groups.csv');groups={name:[int(r['class_id']) for r in classes if r['group']==name] for name in ('head','medium','extreme-tail')}
    model=build_classifier(_inside(root,runtime['weights']),runtime['weights_sha256'],seed=descriptor['seed'])
    state=torch.load(source.parent/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    if state['identity']!=identity:raise ValueError('best checkpoint identity mismatch')
    model.load_state_dict(state['model'],strict=True)
    contents={};results={};coverage={}
    for split in ('val','test'):
        rows=sorted(read(split+'.csv'),key=lambda r:(int(r['class_id']),r['relative_path']))
        if any(r['split']!=split or r['is_synthetic']!='false' for r in rows): raise ValueError('evaluation must contain only real split members')
        dataset=ImageDataset(rows,dataset_root,root);_,loader,generator=make_loaders(dataset,dataset,seed=descriptor['seed'])
        rng=_rng_state(generator)
        try: logits=_predict(model,loader)
        finally: _restore_rng(rng,generator)
        exports,metrics=prediction_exports(rows,logits,groups,classes)
        if split=='val' and metrics['macro_f1']!=best['macro_f1']: raise ValueError('validation replay differs from frozen best score')
        if split=='val' and (source.parent/'training_metrics.json').exists():
            expected=json.loads((source.parent/'training_metrics.json').read_text())['validation']
            if expected!=metrics:raise ValueError('validation replay differs from frozen metric evidence')
        for kind,content in exports.items(): contents[f'{kind}_{split}.{ "npy" if kind=="logits" else "csv"}']=content
        results[split]=metrics;coverage[split]={'samples':len(rows),'manifest_sha256':sha256_file(data_path.parent/(split+'.csv'))}
    contents['metrics.json']=encoded({'validation':results['val'],'test':results['test']})
    contents['group_metrics.json']=encoded({split:{name:{**values,
        'evaluated_class_ids':[c for c in groups[name] if metrics['per_class'][c]['evaluable']]} for name,values in metrics['groups'].items()}
        for split,metrics in results.items()})
    contents['evaluation_metadata.json']=encoded({'run_id':run_id,'checkpoint_sha256':sha256_file(source.parent/'best_checkpoint.pt'),
        'selected_epoch':best['epoch'],'coverage':coverage,'status':'evaluation_completed','source_lock':source.relative_to(root).as_posix(),
        'class_groups_sha256':sha256_file(data_path.parent/'class_groups.csv'),'dependencies':deps})
    # Training metrics.json is mutable until evaluation; training_metrics.json is its frozen authority.
    with active_writer(target):
        mutable=target/'metrics.json'
        if mutable.exists() and mutable.read_bytes()!=contents['metrics.json']:
            frozen=source.parent/'training_metrics.json'
            if not frozen.exists() or mutable.read_bytes()!=frozen.read_bytes(): raise ValueError('unexpected existing metrics')
            temporary=target/'.metrics.evaluation'
            from .artifacts import complete_temporary
            import os
            complete_temporary(temporary,contents['metrics.json']);os.replace(temporary,mutable)
        return publish_records(root,target.relative_to(root).as_posix(),identity if authority==source else f'evaluation:{run_id}',
            contents,deps,owner_token=owner_token,lock_name='evaluation_lock.json')
