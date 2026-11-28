"""Small Slurm-only probes that call the production model providers."""
import csv
import json
import os
from pathlib import Path
import time
import torch
from .artifacts import _inside,publish_records,sha256_file,verify_lock
from .bindings import encoded
from .config import METHODS,verified_configuration
from .images import decode


def run_probe(root,role,*,owner_token):
    if role not in ('classifier','diffusion','dino'):raise ValueError('unknown probe role')
    if not isinstance(owner_token,str) or not owner_token:raise ValueError('owner token required')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('probe requires Slurm CUDA')
    root=Path(root).resolve()
    input_path=root/f'deployment/probe_inputs/{role}/lock.json'
    inputs=verify_lock(input_path,boundary=root)
    if inputs['identity']!=f'probe-input:{role}' or 'input.json' not in inputs['files']:raise ValueError('locked probe input required')
    settings=json.loads((input_path.parent/'input.json').read_text())
    if set(settings)!={'model_lock'}:raise ValueError('invalid probe input schema')
    model_path=_inside(root,settings['model_lock']);model_lock=verify_lock(model_path,boundary=root)
    data_path=root/'splits/lock.json';data=verify_lock(data_path,boundary=root)
    if not {'train.csv','class_groups.csv'}<=set(data['files']):raise ValueError('locked probe source membership required')
    readiness=root/'deployment/data_readiness/lock.json';verify_lock(readiness,boundary=root)
    dataset_root=Path(json.loads((readiness.parent/'V01.json').read_text())['dataset_root'])
    with (data_path.parent/'train.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    with (data_path.parent/'class_groups.csv').open(encoding='utf-8',newline='') as stream:groups=list(csv.DictReader(stream))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    observations=[];model=None
    if role=='classifier':
        from .classifier import build_classifier
        from .training import _require_classifier_runtime,build_optimizer,classifier_update
        from .training_data import ImageDataset,make_loaders
        from .augmentation.mixup import mix_batch
        from .images import MEAN,STD
        _require_classifier_runtime(42)
        if model_lock['identity']!='classifier-weights' or 'model.json' not in model_lock['files']:
            raise ValueError('classifier model receipt required')
        metadata=json.loads((model_path.parent/'model.json').read_text())
        weights=_inside(model_path.parent,metadata['weights'])
        if model_lock['files'].get(metadata['weights'])!=metadata['weights_sha256']:raise ValueError('weights not locked')
        if len(rows)<129:raise ValueError('probe requires 129 real training records')
        model=build_classifier(weights,metadata['weights_sha256'],seed=42).to('cuda').train()
        optimizer=build_optimizer(model)
        dataset=ImageDataset(rows[:129],dataset_root,root)
        loader,_,_=make_loaders(dataset,dataset,seed=42)
        for images,labels in loader:
            # Exercise the same A4 hook with disposable probe pixels and stable IDs.
            raw=(images*images.new_tensor(STD)[None,:,None,None]+images.new_tensor(MEAN)[None,:,None,None]).clamp(0,1)
            batch={'images':raw,'labels':labels,'image_id':[f'probe-{i}' for i in range(len(labels))],
                   'sample_type':['final_synthetic']*len(labels),'n_original':[2]*len(labels),'n_train':[1]*len(labels)}
            pools={int(c):{'images':raw[labels==c],'ids':[f'partner-{i}' for i in range(int((labels==c).sum()))]} for c in labels.unique()}
            mixed,_=mix_batch(batch,{'selected':True,'batch_index':0,'training_seed':42,'epoch':1,'real_pool':pools})
            normalized=(mixed['images']-raw.new_tensor(MEAN)[None,:,None,None])/raw.new_tensor(STD)[None,:,None,None]
            loss=classifier_update(model,optimizer,normalized.to('cuda'),labels.to('cuda'))
            observations.append({'batch':len(labels),'loss':float(loss)})
        model.eval()
        with torch.inference_mode():model(images.to('cuda'))
    else:
        from .preflight import verify_model_snapshot
        if 'model.json' not in model_lock['files']:raise ValueError('model metadata required')
        metadata=json.loads((model_path.parent/'model.json').read_text())
        if role=='diffusion' and (model_lock['identity']!='sd15-snapshot' or metadata.get('family')!='stable-diffusion-v1-5'):
            raise ValueError('SD1.5 snapshot required')
        if role=='dino' and model_lock['identity']!='dinov3-snapshot':raise ValueError('DINO snapshot required')
        snapshot=_inside(model_path.parent,metadata['snapshot'])
        inventory={key.removeprefix(metadata['snapshot']+'/'):digest for key,digest in model_lock['files'].items() if key.startswith(metadata['snapshot']+'/')}
        verify_model_snapshot(snapshot,role,{'repository':metadata['repository'],'revision':metadata['revision'],'files':inventory})
        images=[]
        from .images import decoded_pixel_sha256
        for row in rows[:33]:
            path=_inside(dataset_root,row['relative_path'])
            if sha256_file(path)!=row['sha256'] or decoded_pixel_sha256(path)!=row['decoded_pixel_sha256']:raise ValueError('probe source mismatch')
            images.append(decode(path))
        if role=='dino':
            from .augmentation.mining import _load_dino,extract_cls
            model,processor,device=_load_dino(snapshot)
            if len(images)!=33:raise ValueError('DINO probe requires 33 sources')
            with torch.inference_mode():
                extract_cls(model,processor,images[:32],device) # warmup is discarded
                for batch in (images[:32],images[32:]):
                    value=extract_cls(model,processor,batch,device)
                    observations.append({'shape':list(value.shape)})
        else:
            from .augmentation.diffusion import _load_runtime,generate_image
            runtime=_load_runtime(snapshot)
            if len(groups)<2 or not images:raise ValueError('diffusion probe requires two classes')
            for strength in (.25,.35,.5):
                value,info=generate_image(images[0],groups[0]['class_name'],strength,**runtime,vae_seed=42,latent_seed=123,
                    variant='A2',confuser_country=groups[1]['class_name'],role='final',class_id=int(groups[0]['class_id']),slot=0,attempt=0)
                observations.append({'strength':strength,'shape':list(value.shape),'active_steps':info['active_steps']})
    torch.cuda.synchronize()
    result={'role':role,'status':'observed','job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-started,
            'max_memory_bytes':torch.cuda.max_memory_allocated(),'device':torch.cuda.get_device_name(),
            'torch_version':torch.__version__,'configuration_sha256':verified_configuration()['sha256'],'observations':observations,
            'state_discarded':True}
    del model
    torch.cuda.empty_cache()
    dependencies={path.relative_to(root).as_posix():sha256_file(path) for path in (input_path,model_path,data_path,readiness)}
    return publish_records(root,f'deployment/probes/{role}',f'probe:{role}',{'probe.json':encoded(result)},dependencies,owner_token=owner_token)
