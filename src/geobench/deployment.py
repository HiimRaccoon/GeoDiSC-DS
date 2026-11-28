"""Materialize verified local models and freeze the shared classifier runtime."""
import hashlib
import json
from pathlib import Path
from .artifacts import _inside,_complete_stream,active_writer,claim_writer,complete_temporary,publish_lock,publish_records,sha256_file,staging_path,verify_lock
from .bindings import encoded
from .preflight import verify_model_snapshot,inspect_environment


def stage_model(root,role,snapshot,descriptor,*,owner_token,cache_root=None):
    if role not in ('classifier','diffusion','dino') or not isinstance(owner_token,str) or not owner_token:
        raise ValueError('model role and owner token required')
    descriptor=verify_model_snapshot(snapshot,role,descriptor,cache_root=cache_root)
    snapshot=Path(snapshot).resolve();files=descriptor['files']
    if role=='diffusion':
        required={'model_index.json','scheduler/scheduler_config.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt',
                  'text_encoder/config.json','unet/config.json','vae/config.json'}
        if not required<=set(files) or any(not any(k.startswith(component+'/') and k.endswith(('.bin','.safetensors')) for k in files) for component in ('unet','vae','text_encoder')):
            raise ValueError('incomplete SD1.5 components')
    elif role=='dino':
        if not {'config.json','preprocessor_config.json'}<=set(files) or not any(k.endswith(('.bin','.safetensors')) for k in files):
            raise ValueError('DINO weights and official processor required')
        if json.loads((snapshot/'config.json').read_text())['hidden_size']!=768:raise ValueError('DINO hidden size mismatch')
    elif len(files)!=1 or not next(iter(files)).endswith(('.pt','.pth')):
        raise ValueError('classifier receipt requires one verified ImageNet weight file')
    root=Path(root).resolve();folder={'classifier':'efficientnet','diffusion':'sd15','dino':'dino'}[role]
    identity={'classifier':'classifier-weights','diffusion':'sd15-snapshot','dino':'dinov3-snapshot'}[role]
    target=root/'models'/folder;value=hashlib.sha256((identity+'\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,identity,token=value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token);outputs=[]
            for name,digest in files.items():
                relative=('snapshot/' if role!='classifier' else '')+name
                dest=_inside(stage,relative);dest.parent.mkdir(parents=True,exist_ok=True)
                with (snapshot/name).resolve(strict=True).open('rb') as source:_complete_stream(dest,source)
                if sha256_file(dest)!=digest:raise ValueError('materialized model hash mismatch')
                outputs.append(relative)
            metadata={'repository':descriptor['repository'],'revision':descriptor['revision']}
            if role=='classifier':
                name=next(iter(files));metadata.update(weights=name,weights_sha256=files[name],family='efficientnet-b0',weights_name='IMAGENET1K_V1')
            else:metadata.update(snapshot='snapshot',family='stable-diffusion-v1-5' if role=='diffusion' else 'dinov3-vitb16')
            complete_temporary(stage/'model.json',encoded(metadata));outputs.append('model.json')
            return publish_lock(target,{'identity':identity,'files':outputs,'dependencies':{},'artifact_root':root,'claim':token})
        finally:token.close()


def freeze_runtime(root,*,owner_token):
    """Freeze actual identities only after the disposable classifier probe."""
    import os,subprocess,sys,torch
    root=Path(root).resolve();model_path=root/'models/efficientnet/lock.json';model=verify_lock(model_path,boundary=root)
    if model['identity']!='classifier-weights' or 'model.json' not in model['files']:raise ValueError('classifier weights receipt required')
    metadata=json.loads((model_path.parent/'model.json').read_text())
    probe_path=root/'deployment/probes/classifier/lock.json';probe=verify_lock(probe_path,boundary=root)
    if probe['identity']!='probe:classifier' or 'probe.json' not in probe['files']:raise ValueError('shared classifier probe required')
    observed=json.loads((probe_path.parent/'probe.json').read_text())
    if (observed.get('role')!='classifier' or observed.get('state_discarded') is not True or not observed.get('job_id')
            or observed.get('max_memory_bytes',0)<=0 or probe['dependencies'].get(model_path.relative_to(root).as_posix())!=sha256_file(model_path)):
        raise ValueError('classifier probe provenance mismatch')
    if not torch.cuda.is_available():raise ValueError('freeze actual runtime on the verified CUDA environment')
    from .config import verified_configuration
    config=verified_configuration();package_root=Path(__file__).parent
    environment=inspect_environment(['torch','torchvision','numpy','Pillow','diffusers','transformers'])
    if any(environment['packages'][key]['version'] is None for key in ('torch','torchvision','numpy','Pillow')):
        raise ValueError('classifier environment unresolved')
    pip=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    conda_exe=os.environ.get('CONDA_EXE')
    if not conda_exe:raise ValueError('active Conda executable required for environment export')
    conda=subprocess.run([conda_exe,'list','--explicit'],capture_output=True,text=True,check=True).stdout
    driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    if not driver:raise ValueError('GPU driver identity unresolved')
    runtime={'configuration':config,'environment':environment,'device':torch.cuda.get_device_name(),
             'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'probe_sha256':sha256_file(probe_path),
             'driver_version':driver,
             'weights':(model_path.parent/metadata['weights']).relative_to(root).as_posix(),'weights_sha256':metadata['weights_sha256'],
             'code_sha256':{path.relative_to(package_root).as_posix():sha256_file(path) for path in sorted(package_root.rglob('*.py'))}}
    dependencies={path.relative_to(root).as_posix():sha256_file(path) for path in (model_path,probe_path)}
    return publish_records(root,'runtime','shared-runtime',{'runtime.json':encoded(runtime),'pip_freeze.txt':pip.encode(),
        'conda_explicit.txt':conda.encode()},dependencies,owner_token=owner_token)


def prepare_probe(root,role,model_lock,*,owner_token):
    expected={'classifier':'classifier-weights','diffusion':'sd15-snapshot','dino':'dinov3-snapshot'}
    if role not in expected:raise ValueError('unknown probe role')
    root=Path(root).resolve();path=_inside(root,model_lock);lock=verify_lock(path,boundary=root)
    if lock['identity']!=expected[role]:raise ValueError('probe model identity mismatch')
    return publish_records(root,f'deployment/probe_inputs/{role}',f'probe-input:{role}',
        {'input.json':encoded({'model_lock':model_lock})},{model_lock:sha256_file(path)},owner_token=owner_token)
