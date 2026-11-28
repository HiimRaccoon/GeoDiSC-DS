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


def train_epochs(model, train_loader, validation_loader, groups, loader_generator, checkpoint_dir, *, identity, device='cpu', on_epoch=None):
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
                         identity=identity,device=device,on_epoch=on_epoch)


def _train_epochs(model, train_loader, validation_loader, groups, loader_generator, checkpoint_dir, *, identity, device='cpu', on_epoch=None):
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
    state={'schema_version':1,'identity':identity,'completed_epoch':0,'global_step':0,'steps_per_epoch':count,'best_score':None,
           'best_epoch':None,'best_state':None,'history':[],'lr_history':[]}
    if path.exists():
        if path.is_symlink():
            raise ValueError('checkpoint path cannot be a symlink')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('schema_version') != 1 or type(saved.get('schema_version')) is not int:
            raise ValueError('unsupported checkpoint schema')
        if saved['identity'] != identity or saved['steps_per_epoch'] != count:
            raise ValueError('checkpoint identity or schedule mismatch')
        if (type(saved['completed_epoch']) is not int or not 0 <= saved['completed_epoch'] <= 20
                or saved['global_step'] != saved['completed_epoch'] * count
                or len(saved['history']) != saved['completed_epoch']
                or len(saved['lr_history']) != saved['global_step']):
            raise ValueError('checkpoint progress mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state={key:saved[key] for key in state}
        _restore_rng(saved['rng'],loader_generator)
        if state['best_state'] is not None:
            _atomic_checkpoint(root/'best_checkpoint.pt', {'identity':identity,'epoch':state['best_epoch'],
                                                            'score':state['best_score'],'model':state['best_state']})
    for epoch in range(state['completed_epoch']+1,21):
        model.train(); examples=0; loss_total=0.0; steps=0
        for images,labels in train_loader:
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
        state['completed_epoch']=epoch
        checkpoint={**state,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':_rng_state(loader_generator)}
        _atomic_checkpoint(path,checkpoint)
        _atomic_checkpoint(root/'best_checkpoint.pt', {'identity':identity,'epoch':state['best_epoch'],
                                                       'score':state['best_score'],'model':state['best_state']})
        if on_epoch is not None: on_epoch(epoch)
    return state
