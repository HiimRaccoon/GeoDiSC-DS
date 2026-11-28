"""Shared native-resolution SD1.5 img2img inference primitives."""
import torch


def probe_gpu(root):
    import os
    from ..probes import run_probe
    return run_probe(root,'diffusion',owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN'))
from ..images import pad_native, crop_native
from .quality import TechnicalRejection

BASE_PROMPT = 'A realistic street-level road photograph in {country_name}, look like picture in Google Street View.'
SCHEDULER_SETTINGS = {'algorithm_type':'dpmsolver++', 'solver_order':2, 'solver_type':'midpoint',
                      'timestep_spacing':'linspace', 'lower_order_final':True, 'thresholding':False,
                      'use_karras_sigmas':False}


def e2_policy():
    import importlib.metadata
    from ..artifacts import sha256_file
    from .traditional import e1_policy
    versions = {}
    for name in ('diffusers','transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'code_sha256':sha256_file(__file__), 'shared':e1_policy(), 'libraries':versions}


def build_scheduler(base_config):
    from diffusers import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler.from_config(
        base_config, algorithm_type='dpmsolver++', solver_order=2, solver_type='midpoint',
        timestep_spacing='linspace', lower_order_final=True, thresholding=False, use_karras_sigmas=False)


def _embedding(tokenizer, encoder, text, device):
    tokens = tokenizer(text, truncation=False, return_tensors='pt')
    ids = tokens['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > 77:
        raise ValueError('prompt exceeds tokenizer context')
    tokens = tokenizer(text, padding='max_length', max_length=77, truncation=False, return_tensors='pt')
    return encoder(tokens['input_ids'].to(device))[0]


def generate_image(image, country, strength, *, vae, unet, tokenizer, text_encoder, scheduler, device, vae_seed, latent_seed,
                   variant='E2', confuser_country=None, role='final', class_id=None, slot=None, attempt=None):
    """One deterministic source/attempt, returning native RGB and provenance."""
    if strength not in (0.25, 0.35, 0.5) or not isinstance(country, str) or not country:
        raise ValueError('invalid strength or country')
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise ValueError('autocast is disabled for generation')
    if variant not in {'E2', 'A1', 'A2', 'A3'} or role not in {'reference', 'final'}:
        raise ValueError('invalid generation variant or role')
    if role == 'reference' and variant != 'A1':
        raise ValueError('reference generation uses A1 guidance')
    if variant in {'A2', 'A3'} and (not isinstance(confuser_country, str) or not confuser_country or confuser_country == country):
        raise ValueError('a different confuser country is required')
    config = getattr(scheduler, 'config', None)
    if config is None or any(config.get(key) != value for key, value in SCHEDULER_SETTINGS.items()):
        raise ValueError('scheduler differs from confirmed configuration')
    for model in (vae, unet, text_encoder):
        if isinstance(model, torch.nn.Module):
            if model.training or any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError('generation models must use FP32 eval mode')
    padded, padding = pad_native(image)
    pixels = padded.unsqueeze(0).to(device=device, dtype=torch.float32)
    from ..config import METHODS
    positive_prompt = (METHODS['geo']['prompt'] if variant == 'A3' else BASE_PROMPT).format(country_name=country)
    baseline_prompt = METHODS['geo']['negative_prompt'] if variant == 'A3' else ''
    with torch.inference_mode():
        baseline = _embedding(tokenizer, text_encoder, baseline_prompt, device)
        positive = _embedding(tokenizer, text_encoder, positive_prompt, device)
        confuser = _embedding(tokenizer, text_encoder, BASE_PROMPT.format(country_name=confuser_country), device) if variant in {'A2','A3'} else None
        posterior = vae.encode(pixels * 2 - 1).latent_dist
        vae_rng = torch.Generator(device=device).manual_seed(vae_seed)
        latents = posterior.sample(generator=vae_rng) * vae.config.scaling_factor
        if latents.dtype != torch.float32 or not torch.isfinite(latents).all():
            raise TechnicalRejection('invalid_vae_latents')
        scheduler.set_timesteps(50, device=device)
        active_steps = min(int(50 * strength), 50)
        begin = 50 - active_steps
        timesteps = scheduler.timesteps[begin:]
        if hasattr(scheduler, 'set_begin_index'):
            scheduler.set_begin_index(begin)
        latent_rng = torch.Generator(device=device).manual_seed(latent_seed)
        noise = torch.randn(latents.shape, generator=latent_rng, device=device, dtype=torch.float32)
        latents = scheduler.add_noise(latents, noise, timesteps[:1])
        guidance = []
        for index, timestep in enumerate(timesteps):
            model_input = scheduler.scale_model_input(latents, timestep)
            if variant == 'E2':
                unconditioned = unet(model_input, timestep, encoder_hidden_states=baseline).sample
                conditioned = unet(model_input, timestep, encoder_hidden_states=positive).sample
                predicted = unconditioned + 2.0 * (conditioned - unconditioned)
            else:
                from .guidance import predict_step
                try:
                    predicted, diagnostic = predict_step({'unet':unet, 'latent':model_input, 'timestep':timestep,
                        'index':index, 'count':len(timesteps), 'positive':positive, 'confuser':confuser,
                        'baseline':baseline, 'role':role, 'class_id':class_id, 'slot':slot, 'attempt':attempt, 'variant':variant})
                except TechnicalRejection as exc:
                    exc.guidance=guidance+[{'step':index,'N':len(timesteps),'t_sched':int(timestep),'status':'rejected',
                        'reason':str(exc),'gamma':None,'d_pos':None,'d_neg':None,'w_pos':None,'w_neg':None}]
                    raise
                diagnostic.update({'step':index,'N':len(timesteps),'t_sched':int(timestep),'scheduler_update_index':index,
                                   'status':'valid','reason':None})
                guidance.append(diagnostic)
            if not torch.isfinite(predicted).all():
                raise TechnicalRejection('nonfinite_diffusion_prediction')
            latents = scheduler.step(predicted, timestep, latents).prev_sample
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        if decoded.dtype != torch.float32 or not torch.isfinite(decoded).all():
            raise TechnicalRejection('invalid_vae_output')
        output = crop_native((decoded[0] / 2 + .5).clamp(0, 1), padding)
    return output, {'padding': padding, 'configured_steps': 50, 'active_steps': len(timesteps),
                    'timesteps': [int(t) for t in timesteps], 'strength': strength,
                    'vae_seed': vae_seed, 'latent_seed': latent_seed, 'prompt': positive_prompt, 'guidance': guidance}


def _load_runtime(snapshot):
    """Load the locked local SD1.5 components, exclusively within Slurm."""
    import os
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise ValueError('actual diffusion generation requires a Slurm CUDA allocation')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    from diffusers import StableDiffusionImg2ImgPipeline
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=torch.float32, safety_checker=None)
    pipeline = pipeline.to('cuda')
    pipeline.vae.eval(); pipeline.unet.eval(); pipeline.text_encoder.eval()
    scheduler = build_scheduler(pipeline.scheduler.config)
    return {'vae':pipeline.vae, 'unet':pipeline.unet, 'tokenizer':pipeline.tokenizer,
            'text_encoder':pipeline.text_encoder, 'scheduler':scheduler, 'device':'cuda'}


def generate_pool(root, descriptor, *, owner_token, _e3=None):
    """Freeze one E2 strength pool from locked data, E0/E1, and model inputs."""
    import csv
    import hashlib
    import io
    import json
    import os
    import re
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from ..artifacts import _inside, active_writer, claim_writer, complete_temporary, publish_lock, sha256_file, staging_path, verify_lock
    from ..images import decode, decoded_pixel_sha256
    from ..preflight import audit_prompts
    from .jobs import seed_for, source_for_slot, validate_training_groups, validate_shared_research
    from .quality import HoldoutCollision, audit_candidate, validate_candidate_tensor
    from .traditional import quantize

    root=Path(root).resolve()
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    if not isinstance(descriptor,dict) or set(descriptor) != {'strength','model_lock','model_sha256','dataset_root'}:
        raise ValueError('invalid E2 descriptor')
    strength=descriptor['strength']
    variant = 'E2' if _e3 is None else _e3['variant']
    reference = _e3 is not None and _e3['role'] == 'reference'
    sample_type = 'reference' if reference else 'final_synthetic'
    if type(strength) is not float or strength not in (.25,.35,.5):
        raise ValueError('strength must be in the confirmed grid')
    dataset=Path(descriptor['dataset_root']).resolve(strict=True)
    if root == dataset or root.is_relative_to(dataset) or dataset.is_relative_to(root):
        raise ValueError('dataset and artifact roots must be separate')
    data_lock=root/'splits/lock.json'; e0_lock=root/'pools/E0/pool_lock.json'; e1_lock=root/'pools/E1/pool_lock.json'
    data=verify_lock(data_lock,boundary=root); e0=verify_lock(e0_lock,boundary=root); e1=verify_lock(e1_lock,boundary=root)
    if (not data['identity'].startswith('common-data:') or not e0['identity'].startswith('e0:') or e1['identity'] != 'e1'
            or e0['dependencies'].get('splits/lock.json') != sha256_file(data_lock)
            or e1['dependencies'].get('splits/lock.json') != sha256_file(data_lock)
            or e1['dependencies'].get('pools/E0/pool_lock.json') != sha256_file(e0_lock)):
        raise ValueError('incompatible common/E0/E1 dependencies')
    e0_metadata=json.loads((e0_lock.parent/'pool.json').read_text(encoding='utf-8'))
    if e0_metadata.get('manifest_sha256') != sha256_file(root/'splits/train.csv'):
        raise ValueError('E0 membership mismatch')
    model_lock=_inside(root,descriptor['model_lock'])
    if sha256_file(model_lock) != descriptor['model_sha256']:
        raise ValueError('model lock hash mismatch')
    model=verify_lock(model_lock,boundary=root)
    if model['identity'] != 'sd15-snapshot' or 'model.json' not in model['files']:
        raise ValueError('unverified SD1.5 model lock')
    metadata=json.loads((model_lock.parent/'model.json').read_text(encoding='utf-8'))
    if (metadata.get('family') != 'stable-diffusion-v1-5' or not isinstance(metadata.get('repository'),str)
            or not re.fullmatch(r'[0-9a-f]{40}',metadata.get('revision',''))):
        raise ValueError('SD1.5 immutable identity required')
    if not isinstance(metadata.get('snapshot'), str):
        raise ValueError('snapshot path missing')
    snapshot=_inside(model_lock.parent,metadata['snapshot'])
    if not snapshot.is_dir() or not any(name.startswith(metadata['snapshot']+'/') for name in model['files']):
        raise ValueError('snapshot contents are not locked')
    actual = {path.relative_to(model_lock.parent).as_posix() for path in snapshot.rglob('*') if path.is_file()}
    expected = {name for name in model['files'] if name.startswith(metadata['snapshot']+'/')}
    if actual != expected or any(path.is_symlink() for path in snapshot.rglob('*')):
        raise ValueError('snapshot files do not match the locked manifest')
    def read_csv(name):
        with (root/'splits'/name).open(encoding='utf-8',newline='') as stream:
            return list(csv.DictReader(stream))
    train,val,test,groups=(read_csv(name) for name in ('train.csv','val.csv','test.csv','class_groups.csv'))
    by_class=validate_training_groups(train,groups)
    for row in train:
        path=_inside(dataset,row['relative_path'])
        if sha256_file(path)!=row['sha256'] or decoded_pixel_sha256(path)!=row['decoded_pixel_sha256']:
            raise ValueError('training source hash mismatch')
    configuration=validate_shared_research(root, data_lock)
    runtime=_load_runtime(snapshot)
    prompts={'base':['',*[BASE_PROMPT.format(country_name=group['class_name']) for group in groups]]}
    if variant == 'A3':
        from ..config import METHODS
        prompts['geo']=[METHODS['geo']['negative_prompt'],*[METHODS['geo']['prompt'].format(country_name=g['class_name']) for g in groups]]
    audit=audit_prompts(runtime['tokenizer'],prompts)
    if audit['overflow_roles']:
        raise ValueError('base prompt exceeds tokenizer context')
    tag=strength.__str__()
    prefix = f'augmentation/E2/strength_{tag}' if _e3 is None else _e3['prefix']
    target=root/prefix
    generation_identity = 'e2-generation:'+tag if _e3 is None else _e3['identity']
    token_value=hashlib.sha256(('e2:'+tag+'\0'+owner_token).encode()).hexdigest()
    if _e3 is not None and variant=='A2':
        token_value=hashlib.sha256(('mining\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,generation_identity,token=token_value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token)
            policy={'configuration':configuration,'model_sha256':descriptor['model_sha256'],'strength':strength,
                    'generator_identity':e2_policy(),'model':metadata}
            if _e3 is not None:
                policy['e3'] = _e3
            complete_temporary(stage/'config.json',(json.dumps(policy,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
            complete_temporary(stage/'token_audit.json',(json.dumps(audit,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
            ledger=stage/'candidates.jsonl'
            if ledger.is_symlink(): raise ValueError('candidate ledger cannot be a symlink')
            history=ledger.read_bytes() if ledger.exists() else b''; cursor=0; events=[]
            def record(event):
                nonlocal history,cursor
                content=(json.dumps(event,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()
                saved=history[cursor:cursor+len(content)]
                if not content.startswith(saved): raise ValueError('candidate history mismatch')
                if len(saved)<len(content):
                    remainder=content[len(saved):]
                    with ledger.open('ab') as stream:
                        if os.fstat(stream.fileno()).st_nlink != 1: raise ValueError('candidate history cannot be linked')
                        stream.write(remainder);stream.flush();os.fsync(stream.fileno())
                    history+=remainder
                cursor+=len(content);events.append(event)
            train_files={r['sha256'] for r in train};train_pixels={r['decoded_pixel_sha256'] for r in train}
            hold_files={r['sha256'] for r in val+test};hold_pixels={r['decoded_pixel_sha256'] for r in val+test}
            accepted_files=set();accepted_pixels=set(); rows=[]; budgets={}
            for group in sorted(groups,key=lambda item:int(item['class_id'])):
                class_id=int(group['class_id']); count=int(group['n_train']); original=int(group['n_original'])
                if not reference and (count >= 20 or original == 1): continue
                quota=20 if reference else min(100,5*count)-count; budgets[str(class_id)]=quota
                sources={r['relative_path']:r for r in by_class[class_id]}
                for slot in range(quota):
                    source=sources[source_for_slot(list(sources),class_id,slot,method='E2',reference=reference)]
                    image=decode(_inside(dataset,source['relative_path']))
                    for attempt in range(5):
                        index=5*slot+attempt
                        vae_seed=seed_for('E2',class_id,index,'vae',reference=reference);latent_seed=seed_for('E2',class_id,index,'latent',reference=reference)
                        job={'method':variant,'strength':strength,'class_id':class_id,'class_name':group['class_name'],'slot':slot,'attempt':attempt,
                             'source_id':source['image_id'],'source_sha256':source['sha256'],'source_pixel_sha256':source['decoded_pixel_sha256'],
                             'vae_seed':vae_seed,'latent_seed':latent_seed,'config_sha256':configuration['sha256']['methods']}
                        # Complete, durable decisions are replayed from their original bytes.
                        end = history.find(b'\n', cursor)
                        if end != -1:
                            try:
                                prior=json.loads(history[cursor:end].decode('utf-8'))
                            except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                                raise ValueError('invalid candidate history') from exc
                            if not isinstance(prior,dict) or any(prior.get(key)!=value for key,value in job.items()):
                                raise ValueError('candidate history identity mismatch')
                            status, reason = prior.get('status'), prior.get('reason')
                            technical = {'invalid_vae_latents','invalid_vae_output','nonfinite_diffusion_prediction',
                                         'nonfinite_pixels','geometry_mismatch','pixel_range','nonfinite_condition','invalid_condition_std',
                                         'nonfinite_rescaled_condition','nonfinite_guidance_distance','nonfinite_guidance_output','nonfinite_guidance_result'}
                            if status == 'rejected' and reason in technical and 'output_id' not in prior:
                                record(prior)
                                continue
                            output_id=hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
                            if prior.get('output_id') != output_id:
                                raise ValueError('candidate output identity mismatch')
                            relative='images/'+output_id+'.png'
                            path=stage/relative
                            try:
                                file_hash=sha256_file(path);pixel_hash=decoded_pixel_sha256(path)
                                if decode(path).size != image.size:
                                    raise ValueError('candidate geometry mismatch')
                            except OSError as exc:
                                raise ValueError('completed candidate missing or unreadable') from exc
                            if file_hash != prior.get('sha256') or pixel_hash != prior.get('decoded_pixel_sha256'):
                                raise ValueError('completed candidate hash mismatch')
                            try:
                                actual_reason=audit_candidate(file_hash,pixel_hash,train_files=train_files,train_pixels=train_pixels,
                                    holdout_files=hold_files,holdout_pixels=hold_pixels,accepted_files=accepted_files,accepted_pixels=accepted_pixels)
                            except HoldoutCollision:
                                if status != 'blocked' or reason != 'holdout_exact_duplicate':
                                    raise ValueError('candidate holdout decision mismatch')
                                record(prior)
                                raise
                            if actual_reason is None and _e3 is not None and (file_hash in _e3['excluded_files'] or pixel_hash in _e3['excluded_pixels']):
                                actual_reason='reference_final_exact_duplicate'
                            if status == 'accepted' and actual_reason is None and reason is None:
                                record(prior);accepted_files.add(file_hash);accepted_pixels.add(pixel_hash)
                                rows.append({**job,'image_id':output_id,'relative_path':prefix+'/'+relative,
                                             'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'sample_type':sample_type})
                                break
                            if status == 'rejected' and reason == actual_reason and actual_reason is not None:
                                record(prior)
                                continue
                            raise ValueError('candidate decision mismatch')
                        try:
                            kwargs={} if _e3 is None else {'variant':variant,'role':_e3['role'],'class_id':class_id,'slot':slot,'attempt':attempt,
                                'confuser_country':_e3['confusers'].get(str(class_id))}
                            candidate,provenance=generate_image(image,group['class_name'],strength,**runtime,vae_seed=vae_seed,latent_seed=latent_seed,**kwargs)
                        except TechnicalRejection as exc:
                            record({**job,'status':'rejected','reason':str(exc),**({'provenance':{'guidance':exc.guidance}} if hasattr(exc,'guidance') else {})})
                            continue
                        failure=validate_candidate_tensor(candidate,image.size)
                        if failure:
                            record({**job,'status':'rejected','reason':failure});continue
                        pixels=quantize(candidate).permute(1,2,0).cpu().numpy()
                        content=io.BytesIO();Image.fromarray(pixels).save(content,format='PNG')
                        output_id=hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
                        relative='images/'+output_id+'.png';path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
                        complete_temporary(path,content.getvalue())
                        file_hash=sha256_file(path);pixel_hash=decoded_pixel_sha256(path)
                        try:
                            reason=audit_candidate(file_hash,pixel_hash,train_files=train_files,train_pixels=train_pixels,
                                                   holdout_files=hold_files,holdout_pixels=hold_pixels,
                                                   accepted_files=accepted_files,accepted_pixels=accepted_pixels)
                        except HoldoutCollision:
                            record({**job,'output_id':output_id,'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,
                                    'status':'blocked','reason':'holdout_exact_duplicate'});raise
                        if reason is None and _e3 is not None and (file_hash in _e3['excluded_files'] or pixel_hash in _e3['excluded_pixels']):
                            reason='reference_final_exact_duplicate'
                        record({**job,'output_id':output_id,'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,
                                'status':'accepted' if reason is None else 'rejected','reason':reason,'provenance':provenance})
                        if reason is None:
                            accepted_files.add(file_hash);accepted_pixels.add(pixel_hash)
                            rows.append({**job,'image_id':output_id,'relative_path':prefix+'/'+relative,
                                         'sha256':file_hash,'decoded_pixel_sha256':pixel_hash,'sample_type':sample_type})
                            break
                    else:
                        record({'method':'E2','class_id':class_id,'slot':slot,'status':'shortage','reason':'five_attempts_rejected'})
                        raise ValueError('E2 quota shortage')
            if cursor != len(history): raise ValueError('candidate history has unexpected records')
            if not ledger.exists(): complete_temporary(ledger,b'')
            fields=['image_id','relative_path','class_id','class_name','source_id','source_sha256','source_pixel_sha256',
                    'method','strength','slot','attempt','vae_seed','latent_seed','config_sha256','sha256','decoded_pixel_sha256','sample_type']
            output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
            complete_temporary(stage/'manifest.csv',output.getvalue().encode())
            outputs=['config.json','token_audit.json','candidates.jsonl','manifest.csv']+[r['relative_path'].removeprefix(prefix+'/') for r in rows]
            dependencies={'splits/lock.json':sha256_file(data_lock),'pools/E0/pool_lock.json':sha256_file(e0_lock),
                          'pools/E1/pool_lock.json':sha256_file(e1_lock),descriptor['model_lock']:descriptor['model_sha256']}
            if _e3 is not None:
                dependencies.update(_e3['dependencies'])
                diagnostics=[]
                for event in events:
                    for step, detail in enumerate(event.get('provenance',{}).get('guidance',[])):
                        diagnostics.append({**{key:event[key] for key in ('class_id','slot','attempt')},'variant':variant,'role':_e3['role'],'step':step,**detail})
                complete_temporary(stage/'guidance.jsonl',b''.join((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode() for row in diagnostics))
                outputs.append('guidance.jsonl')
            generated=publish_lock(target,{'identity':generation_identity,'files':outputs,'dependencies':dependencies,'artifact_root':root,'claim':token,
                                           'lock_name':'bank_lock.json' if reference else 'lock.json'})
        finally:
            token.close()
        if _e3 is not None:
            return generated
        pool=root/'pools/E2'/('strength_'+tag)
        pool_token_value=hashlib.sha256(('e2-pool:'+tag+'\0'+owner_token).encode()).hexdigest()
        pool_token=claim_writer(pool,'e2:'+tag,token=pool_token_value,resume=(pool/'.writer').exists())
        try:
            stage=staging_path(pool,pool_token)
            complete_temporary(stage/'manifest.csv',output.getvalue().encode())
            pool_info={'strength':strength,'required':sum(budgets.values()),'accepted':len(rows),'per_class_required':budgets,
                       'rejected':sum(e['status']=='rejected' for e in events),'shortage':0,'generation_sha256':sha256_file(generated)}
            complete_temporary(stage/'pool.json',(json.dumps(pool_info,sort_keys=True,separators=(',', ':'))+'\n').encode())
            return publish_lock(pool,{'identity':'e2:'+tag,'files':['manifest.csv','pool.json'],
                'dependencies':{generated.relative_to(root).as_posix():sha256_file(generated)},'artifact_root':root,'claim':pool_token,'lock_name':'pool_lock.json'})
        finally:
            pool_token.close()


def build_hybrid_pool(root, strength, *, owner_token):
    """Freeze the E0 + E1 + E2 candidate membership for one strength."""
    import csv
    import hashlib
    import io
    import json
    from pathlib import Path
    from ..artifacts import _inside, active_writer, claim_writer, complete_temporary, publish_lock, sha256_file, staging_path, verify_lock
    from ..images import decoded_pixel_sha256
    from .jobs import validate_training_groups
    root=Path(root).resolve()
    if type(strength) is not float or strength not in (.25,.35,.5) or not isinstance(owner_token,str) or not owner_token:
        raise ValueError('valid strength and owner token are required')
    tag=str(strength)
    paths={'data':'splits/lock.json','e0':'pools/E0/pool_lock.json','e1':'pools/E1/pool_lock.json',
           'e2':f'pools/E2/strength_{tag}/pool_lock.json'}
    locks={role:verify_lock(_inside(root,path),boundary=root) for role,path in paths.items()}
    if (not locks['data']['identity'].startswith('common-data:') or not locks['e0']['identity'].startswith('e0:')
            or locks['e1']['identity'] != 'e1' or locks['e2']['identity'] != 'e2:'+tag):
        raise ValueError('pool identity mismatch')
    data_hash=sha256_file(root/paths['data']); e0_hash=sha256_file(root/paths['e0'])
    if (locks['e0']['dependencies'].get(paths['data']) != data_hash
            or locks['e1']['dependencies'].get(paths['data']) != data_hash
            or locks['e1']['dependencies'].get(paths['e0']) != e0_hash):
        raise ValueError('shared pool dependency mismatch')
    # E2's generation lock must bind these exact shared inputs as well.
    generation_path=f'augmentation/E2/strength_{tag}/lock.json'
    if locks['e2']['dependencies'].get(generation_path) != sha256_file(root/generation_path):
        raise ValueError('E2 generation dependency mismatch')
    generation=verify_lock(root/generation_path,boundary=root)
    if (generation['dependencies'].get(paths['data']) != data_hash or generation['dependencies'].get(paths['e0']) != e0_hash
            or generation['dependencies'].get(paths['e1']) != sha256_file(root/paths['e1'])):
        raise ValueError('E2 was generated from different shared inputs')
    def read(path):
        with path.open(encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))
    real=read(root/'splits/train.csv'); groups=read(root/'splits/class_groups.csv')
    validate_training_groups(real,groups)
    source={row['image_id']:row for row in real}
    if json.loads((root/'pools/E0/pool.json').read_text(encoding='utf-8')).get('manifest_sha256') != sha256_file(root/'splits/train.csv'):
        raise ValueError('E0 membership mismatch')
    group_by_class={int(group['class_id']):group for group in groups}
    traditional=read(root/'pools/E1/manifest.csv'); synthetic=read(root/f'pools/E2/strength_{tag}/manifest.csv')
    counts={'traditional':{},'final_synthetic':{}}
    for kind, records, prefix in (('traditional',traditional,'augmentation/E1/images/'),
                                   ('final_synthetic',synthetic,f'augmentation/E2/strength_{tag}/images/')):
        for row in records:
            class_id=int(row['class_id']); original=source.get(row.get('source_id')); relative=row.get('relative_path','')
            if (row.get('sample_type') != kind or original is None or int(original['class_id']) != class_id
                    or not relative.startswith(prefix) or row.get('source_sha256') != original['sha256']
                    or row.get('source_pixel_sha256') != original['decoded_pixel_sha256']):
                raise ValueError('augmentation source membership mismatch')
            image=_inside(root,relative)
            if sha256_file(image)!=row['sha256'] or decoded_pixel_sha256(image)!=row['decoded_pixel_sha256']:
                raise ValueError('augmentation content mismatch')
            counts[kind][class_id]=counts[kind].get(class_id,0)+1
        if set(counts[kind])-set(group_by_class): raise ValueError('unknown augmented class')
    for class_id, group in group_by_class.items():
        n=int(group['n_train']); original=int(group['n_original'])
        if counts['traditional'].get(class_id,0) != (100-n if 20<=n<100 else 0):
            raise ValueError('E1 class quota mismatch')
        if counts['final_synthetic'].get(class_id,0) != (min(100,5*n)-n if n<20 and original>=2 else 0):
            raise ValueError('E2 class quota mismatch')
    fields=['image_id','relative_path','class_name','class_id','sample_type','source_id','source_sha256','source_pixel_sha256',
            'sha256','decoded_pixel_sha256','method','job_id','seed','vae_seed','latent_seed','config_sha256','strength']
    combined=[]
    for row in real:
        combined.append({**{k:row.get(k,'') for k in fields},'sample_type':'real'})
    for row in traditional+synthetic:
        combined.append({**{k:row.get(k,'') for k in fields},'class_name':group_by_class[int(row['class_id'])]['class_name'],
                         'job_id':row.get('job_id') or row['image_id'], 'config_sha256':row.get('config_sha256') or row.get('method_config_sha256','')})
    if len({row['image_id'] for row in combined}) != len(combined):
        raise ValueError('duplicate hybrid image ID')
    combined.sort(key=lambda row:(int(row['class_id']),row['image_id']))
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(combined)
    dependencies={path:sha256_file(root/path) for path in paths.values()}
    target=root/'pools/E1E2/candidates'/('strength_'+tag)
    token_value=hashlib.sha256(('hybrid:'+tag+'\0'+owner_token).encode()).hexdigest()
    with active_writer(root):
        token=claim_writer(target,'candidate-pool:'+tag,token=token_value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token)
            complete_temporary(stage/'manifest.csv',output.getvalue().encode())
            info={'strength':strength,'real_count':len(real),'traditional_count':len(traditional),'synthetic_count':len(synthetic),
                  'manifest_sha256':sha256_file(stage/'manifest.csv')}
            complete_temporary(stage/'pool.json',(json.dumps(info,sort_keys=True,separators=(',', ':'))+'\n').encode())
            return publish_lock(target,{'identity':'candidate-pool:'+tag,'files':['manifest.csv','pool.json'],
                'dependencies':dependencies,'artifact_root':root,'claim':token,'lock_name':'pool_lock.json'})
        finally:
            token.close()
