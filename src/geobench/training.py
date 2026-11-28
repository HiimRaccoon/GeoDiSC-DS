"""Confirmed shared classifier learning-rate schedule."""
import math


def steps_per_epoch(anchors):
    if type(anchors) is not int or anchors < 1:
        raise ValueError('anchor count must be positive')
    return (anchors + 127) // 128


def learning_rate(step, steps):
    if type(step) is not int or type(steps) is not int or steps < 1 or not 0 <= step <= 20 * steps:
        raise ValueError('invalid optimizer step')
    epoch = step / steps
    if epoch < 1:
        return 3e-4 * (0.1 + 0.9 * epoch)
    return 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + math.cos(math.pi * (epoch - 1) / 19))


def _rng_state(generator):
    import random
    import numpy as np
    import torch
    numpy_state=np.random.get_state()
    safe_numpy=(numpy_state[0],numpy_state[1].tolist(),numpy_state[2],numpy_state[3],numpy_state[4])
    return {'python':random.getstate(), 'numpy':safe_numpy, 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader':generator.get_state()}


def _restore_rng(state, generator):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    numpy_state=state['numpy']
    np.random.set_state((numpy_state[0],np.asarray(numpy_state[1],dtype=np.uint32),numpy_state[2],numpy_state[3],numpy_state[4]))
    torch.set_rng_state(state['torch']); generator.set_state(state['loader'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available(): raise ValueError('checkpoint requires CUDA RNG state')
        torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_checkpoint(path, data):
    import os
    import torch
    temp=path.with_name(path.name+'.staging')
    with temp.open('wb') as stream:
        torch.save(data,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def train_epochs(model, train_loader, validation_loader, groups, loader_generator, checkpoint_dir, *, identity, device='cpu', on_epoch=None, mixup_seed=None):
    """Require the confirmed loader contract before entering the training core."""
    from torch.utils.data import RandomSampler, SequentialSampler
    for loader in (train_loader, validation_loader):
        if (loader.batch_size != 128 or loader.drop_last or loader.num_workers != 4 or not loader.pin_memory
                or loader.persistent_workers or loader.prefetch_factor != 2):
            raise ValueError('loader differs from confirmed runtime')
    if (not isinstance(train_loader.sampler, RandomSampler) or train_loader.sampler.replacement
            or train_loader.sampler.num_samples != len(train_loader.dataset)
            or train_loader.sampler.generator is not loader_generator
            or not isinstance(validation_loader.sampler, SequentialSampler)
            or len(train_loader) != steps_per_epoch(len(train_loader.dataset))):
        raise ValueError('loader traversal differs from confirmed runtime')
    return _train_epochs(model,train_loader,validation_loader,groups,loader_generator,checkpoint_dir,
                         identity=identity,device=device,on_epoch=on_epoch,mixup_seed=mixup_seed)


def _train_epochs(model, train_loader, validation_loader, groups, loader_generator, checkpoint_dir, *, identity, device='cpu', on_epoch=None, mixup_seed=None):
    """Train the common 20 epochs and commit recoverable epoch-boundary state."""
    import copy
    from pathlib import Path
    import torch
    from .metrics import compute_metrics
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise ValueError('training requires autocast OFF')
    if not isinstance(identity,str) or not identity or not isinstance(loader_generator,torch.Generator):
        raise ValueError('run identity and dedicated loader generator are required')
    count=len(train_loader)
    if count < 1:
        raise ValueError('training loader must be nonempty')
    root=Path(checkpoint_dir); root.mkdir(parents=True,exist_ok=True)
    path=root/'last_checkpoint.pt'
    model=model.to(device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,
                                amsgrad=False,maximize=False,foreach=False,fused=False,capturable=False,differentiable=False)
    if mixup_seed is not None and (type(mixup_seed) is not int or mixup_seed not in (42,123,2026) or not hasattr(train_loader.dataset, 'real_pool')):
        raise ValueError('invalid Mixup seed or dataset')
    state={'schema_version':1,'identity':identity,'completed_epoch':0,'global_step':0,'steps_per_epoch':count,'best_score':None,
           'best_epoch':None,'best_state':None,'history':[],'lr_history':[], 'mixup_seed':mixup_seed, 'mixup_history':[]}
    if path.exists():
        if path.is_symlink():
            raise ValueError('checkpoint path cannot be a symlink')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('schema_version') != 1 or type(saved.get('schema_version')) is not int:
            raise ValueError('unsupported checkpoint schema')
        if saved['identity'] != identity or saved['steps_per_epoch'] != count:
            raise ValueError('checkpoint identity or schedule mismatch')
        if saved.get('mixup_seed') != mixup_seed:
            raise ValueError('checkpoint Mixup identity mismatch')
        if (type(saved['completed_epoch']) is not int or not 0 <= saved['completed_epoch'] <= 20
                or saved['global_step'] != saved['completed_epoch'] * count
                or len(saved['history']) != saved['completed_epoch']
                or len(saved['lr_history']) != saved['global_step']
                or len(saved['mixup_history']) != (saved['completed_epoch'] if mixup_seed is not None else 0)):
            raise ValueError('checkpoint progress mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state={key:saved[key] for key in state}
        _restore_rng(saved['rng'],loader_generator)
        if state['best_state'] is not None:
            _atomic_checkpoint(root/'best_checkpoint.pt', {'identity':identity,'epoch':state['best_epoch'],
                                                            'score':state['best_score'],'model':state['best_state']})
    for epoch in range(state['completed_epoch']+1,21):
        model.train(); examples=0; loss_total=0.0; steps=0
        if mixup_seed is not None:
            from .augmentation.mixup import select_batches, mix_batch
            selected = set(select_batches(count, training_seed=mixup_seed, epoch=epoch))
            mixup_records = []
        for batch_index, batch in enumerate(train_loader):
            if mixup_seed is None:
                images, labels = batch
            else:
                from .images import MEAN, STD
                eligible = [int(label) for label, kind, original, ntrain in zip(batch['labels'], batch['sample_type'], batch['n_original'], batch['n_train'])
                            if kind == 'final_synthetic' and int(original) != 1 and int(ntrain) < 20]
                context = {'selected':batch_index in selected, 'batch_index':batch_index,
                           'training_seed':mixup_seed, 'epoch':epoch,
                           'real_pool':train_loader.dataset.real_pool(eligible) if batch_index in selected else {}}
                mixed, evidence = mix_batch(batch, context)
                images, labels = mixed['images'], mixed['labels']
                images = (images - images.new_tensor(MEAN)[None,:,None,None]) / images.new_tensor(STD)[None,:,None,None]
                mixup_records.append({'batch_index':batch_index, **evidence})
            images=images.to(device=device,dtype=torch.float32); labels=labels.to(device=device,dtype=torch.long)
            if not len(labels): raise ValueError('empty batch')
            rate=learning_rate(state['global_step'],count)
            for group in optimizer.param_groups: group['lr']=rate
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=torch.nn.functional.cross_entropy(logits,labels,reduction='mean',label_smoothing=0.0)
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            loss.backward(); optimizer.step()
            state['lr_history'].append({'step':state['global_step'],'lr':rate})
            state['global_step']+=1; steps+=1; examples+=len(labels); loss_total+=float(loss.detach())*len(labels)
        if steps != count or examples == 0:
            raise ValueError('incomplete training traversal')
        rng=_rng_state(loader_generator)
        try:
            model.eval(); ys=[]; predictions=[]
            with torch.inference_mode():
                for images,labels in validation_loader:
                    ys.append(labels.cpu()); predictions.append(model(images.to(device=device,dtype=torch.float32)).cpu())
            if not ys: raise ValueError('empty validation')
            metrics=compute_metrics(torch.cat(ys),torch.cat(predictions),groups)
        finally:
            _restore_rng(rng,loader_generator)
        score=metrics['macro_f1']
        if state['best_score'] is None or score > state['best_score']:
            state['best_score']=score; state['best_epoch']=epoch
            state['best_state']=copy.deepcopy(model.state_dict())
        tail = metrics['groups'].get('extreme-tail', {}).get('macro_f1')
        state['history'].append({'epoch':epoch,'loss':loss_total/examples,'macro_f1':score,'tail_f1':tail})
        if mixup_seed is not None:
            state['mixup_history'].append({'epoch':epoch,'selected_batches':sorted(selected),'batches':mixup_records})
        state['completed_epoch']=epoch
        checkpoint={**state,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':_rng_state(loader_generator)}
        _atomic_checkpoint(path,checkpoint)
        _atomic_checkpoint(root/'best_checkpoint.pt', {'identity':identity,'epoch':state['best_epoch'],
                                                       'score':state['best_score'],'model':state['best_state']})
        if on_epoch is not None: on_epoch(epoch)
    return state


def _require_classifier_runtime(seed):
    """Enforce the confirmed deterministic single-GPU execution context."""
    import os
    import random
    import numpy as np
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise ValueError('classifier training requires a Slurm CUDA allocation')
    if os.environ.get('PYTHONHASHSEED') != str(seed) or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('deterministic interpreter environment is required')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def train_run(root, run_id, *, owner_token):
    """Train one frozen official run and publish its completed checkpoints."""
    import csv
    import hashlib
    import json
    from pathlib import Path
    import torch
    from .artifacts import _inside, active_writer, claim_writer, complete_temporary, publish_lock, sha256_file, staging_path, verify_lock
    from .classifier import build_classifier
    from .config import resolve_run
    from .training_data import load_locked_datasets, make_loaders
    from .augmentation.jobs import validate_shared_research

    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError('owner token required')
    root = Path(root).resolve(strict=True)
    descriptor = resolve_run(root, run_id)
    condition, seed = descriptor['condition'], descriptor['seed']
    target = root/'runs'/condition/f'seed_{seed}'
    descriptor_path = target/'descriptor.json'
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor_hash = hashlib.sha256(descriptor_bytes).hexdigest()
    identity = f'run:{run_id}:{descriptor_hash}'
    readiness_path = root/'deployment/data_readiness/lock.json'
    readiness = verify_lock(readiness_path, boundary=root)
    if readiness['identity'] != 'data-readiness' or 'V01.json' not in readiness['files']:
        raise ValueError('data readiness evidence required')
    dataset_root = Path(json.loads((readiness_path.parent/'V01.json').read_text(encoding='utf-8'))['dataset_root']).resolve(strict=True)
    if root == dataset_root or root.is_relative_to(dataset_root) or dataset_root.is_relative_to(root):
        raise ValueError('dataset and artifact roots must be separate')
    dependencies = {item['path']:item['sha256'] for item in descriptor['dependencies'].values()}
    dependencies[readiness_path.relative_to(root).as_posix()] = sha256_file(readiness_path)
    existing = target/'checkpoint_lock.json'
    if existing.exists():
        locked = verify_lock(existing,boundary=root)
        if locked['identity'] != identity or locked['dependencies'] != dependencies:
            raise ValueError('published checkpoint identity mismatch')
        return existing
    _require_classifier_runtime(seed)
    records = descriptor['dependencies']
    data_path = _inside(root, records['data']['path']); pool_path = _inside(root, records['pool']['path'])
    data_lock = verify_lock(data_path,boundary=root)
    if data_lock['dependencies'].get(readiness_path.relative_to(root).as_posix()) != dependencies[readiness_path.relative_to(root).as_posix()]:
        raise ValueError('data readiness lineage mismatch')
    validate_shared_research(root,data_path)
    if 'class_groups.csv' not in data_lock['files']:
        raise ValueError('class groups not locked')
    with (data_path.parent/'class_groups.csv').open(encoding='utf-8',newline='') as stream:
        group_rows=list(csv.DictReader(stream))
    groups = {}
    for row in group_rows:
        groups.setdefault(row['group'],[]).append(int(row['class_id']))
    train_data, val_data = load_locked_datasets(root,dataset_root,data_path,pool_path,mixup=condition=='E3-A4')
    train_loader, val_loader, generator = make_loaders(train_data,val_data,seed=seed)
    runtime_path = _inside(root,records['runtime']['path'])
    runtime_lock = verify_lock(runtime_path,boundary=root)
    if 'runtime.json' not in runtime_lock['files']:
        raise ValueError('runtime metadata not locked')
    runtime = json.loads((runtime_path.parent/'runtime.json').read_text(encoding='utf-8'))
    weights = _inside(root,runtime['weights'])
    model_lock_path = root/'models/efficientnet/lock.json'
    model_lock = verify_lock(model_lock_path,boundary=root)
    if (model_lock['identity'] != 'classifier-weights' or
        runtime_lock['dependencies'].get(model_lock_path.relative_to(root).as_posix()) != sha256_file(model_lock_path) or
        not weights.is_relative_to(model_lock_path.parent) or
        model_lock['files'].get(weights.relative_to(model_lock_path.parent).as_posix()) != runtime['weights_sha256']):
        raise ValueError('runtime classifier weights mismatch')
    initialization = _inside(root,records['initialization']['path'])
    initial_lock = verify_lock(initialization,boundary=root)
    if 'initial_state.pt' not in initial_lock['files']:
        raise ValueError('initial state not locked')
    model = build_classifier(weights,runtime['weights_sha256'],seed=seed)
    model.load_state_dict(torch.load(initialization.parent/'initial_state.pt',map_location='cpu',weights_only=True),strict=True)
    token_value = hashlib.sha256((identity+'\0'+owner_token).encode()).hexdigest()
    with active_writer(root):
        token = claim_writer(target,identity,token=token_value,resume=(target/'.writer').exists())
        try:
            stage = staging_path(target,token)
            complete_temporary(stage/'run_descriptor.json',descriptor_bytes)
            state = train_epochs(model,train_loader,val_loader,groups,generator,stage,identity=identity,device='cuda',
                                 mixup_seed=seed if condition=='E3-A4' else None)
            if state['completed_epoch'] != 20 or state['global_step'] != 20*state['steps_per_epoch'] or len(state['history']) != 20:
                raise ValueError('training run incomplete')
            if sha256_file(descriptor_path) != descriptor_hash:
                raise ValueError('run descriptor changed during training')
            def write_json(name, value):
                complete_temporary(stage/name,(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
            write_json('history.json',state['history'])
            best = state['history'][state['best_epoch']-1]
            write_json('validation_at_best.json',{'epoch':state['best_epoch'],'macro_f1':state['best_score'],
                'tail_f1':best['tail_f1'],'checkpoint_sha256':sha256_file(stage/'best_checkpoint.pt')})
            files=['run_descriptor.json','best_checkpoint.pt','last_checkpoint.pt','history.json','validation_at_best.json']
            if condition == 'E3-A4':
                write_json('mixup_history.json',state['mixup_history']); files.append('mixup_history.json')
            return publish_lock(target,{'identity':identity,'files':files,'dependencies':dependencies,'artifact_root':root,
                                        'claim':token,'lock_name':'checkpoint_lock.json'})
        finally:
            token.close()
