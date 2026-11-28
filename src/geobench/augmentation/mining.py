"""FP32 reference embeddings, centroids and exact-tie confuser selection."""
import torch


def mine_confusers(features, labels):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    if (features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels)
            or len(features) == 0 or labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool):
        raise ValueError('invalid reference features or labels')
    if not torch.isfinite(features).all() or (labels < 0).any():
        raise ValueError('invalid reference values')
    labels = labels.to(features.device)
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('zero or invalid embedding norm')
    embeddings = features / norms
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError('at least two countries are required')
    centroids = torch.stack([embeddings[labels == class_id].mean(dim=0) for class_id in classes])
    centroid_norms = torch.linalg.vector_norm(centroids, dim=1, keepdim=True)
    if not torch.isfinite(centroid_norms).all() or (centroid_norms == 0).any():
        raise ValueError('zero or invalid centroid norm')
    normalized = centroids / centroid_norms
    similarities = normalized @ normalized.T
    if not torch.isfinite(similarities).all():
        raise ValueError('invalid cosine similarity')
    confusers = {}
    for i, class_id in enumerate(classes):
        candidates = [(float(similarities[i,j]), other) for j,other in enumerate(classes) if j != i]
        confusers[class_id] = min(candidates, key=lambda candidate: (-candidate[0],candidate[1]))[1]
    return {'embeddings': embeddings, 'centroids': centroids, 'similarities': similarities,
            'class_ids': classes, 'confusers': confusers}


def _load_dino(snapshot):
    import os
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise ValueError('DINO extraction requires a Slurm CUDA allocation')
    from transformers import AutoImageProcessor, AutoModel
    processor=AutoImageProcessor.from_pretrained(str(snapshot),local_files_only=True)
    model=AutoModel.from_pretrained(str(snapshot),local_files_only=True,torch_dtype=torch.float32).to('cuda').eval()
    model.requires_grad_(False)
    return model,processor,'cuda'


def extract_and_mine(root, *, model_lock, owner_token):
    """Extract the frozen reference bank once and publish reusable mining arrays."""
    import csv, hashlib, io, json
    from pathlib import Path
    import numpy as np
    from ..artifacts import _inside, active_writer, claim_writer, staging_path, complete_temporary, publish_lock, sha256_file, verify_lock
    from ..images import decode, decoded_pixel_sha256
    from ..config import METHODS
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    root=Path(root).resolve(); bank_path=root/'augmentation/E3/A2/references/bank_lock.json'
    bank=verify_lock(bank_path,boundary=root)
    if bank['identity'] != 'reference-bank' or 'manifest.csv' not in bank['files']:
        raise ValueError('frozen reference bank required')
    model_path=_inside(root,model_lock); locked_model=verify_lock(model_path,boundary=root)
    if locked_model['identity'] != 'dinov3-snapshot' or 'model.json' not in locked_model['files']:
        raise ValueError('official DINO snapshot lock required')
    metadata=json.loads((model_path.parent/'model.json').read_text())
    if metadata.get('repository') != METHODS['mining']['model'] or metadata.get('revision') != METHODS['mining']['revision']:
        raise ValueError('DINO checkpoint identity mismatch')
    snapshot=_inside(model_path.parent,metadata['snapshot'])
    actual={p.relative_to(model_path.parent).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    expected={key for key in locked_model['files'] if key.startswith(metadata['snapshot']+'/')}
    if not actual or actual != expected or any(p.is_symlink() for p in snapshot.rglob('*')):
        raise ValueError('DINO snapshot inventory mismatch')
    with (bank_path.parent/'manifest.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    rows.sort(key=lambda row:(int(row['class_id']),row['image_id']))
    if (len(rows)!=2480 or len({r['image_id'] for r in rows})!=2480
            or any(sum(int(r['class_id'])==c for r in rows)!=20 for c in range(124))
            or any(r['sample_type']!='reference' for r in rows)):
        raise ValueError('reference bank must contain 124 classes with 20 references each')
    target=root/'augmentation/E3/A2/features'
    dependencies={bank_path.relative_to(root).as_posix():sha256_file(bank_path),model_lock:sha256_file(model_path)}
    feature_path=target/'feature_lock.json'
    if feature_path.exists():
        frozen=verify_lock(feature_path,boundary=root)
        if frozen['identity']!='reference-features' or frozen['dependencies'] != dependencies:
            raise ValueError('feature dependencies mismatch')
        index=json.loads((target/'feature_index.json').read_text())
        result={'similarities':torch.from_numpy(np.load(target/'similarities.npy',allow_pickle=False)),
                'confusers':index['confusers']}
    else:
        model,processor,device=_load_dino(snapshot)
        if model.training or any(p.dtype!=torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('DINO must use frozen FP32 eval mode')
        features=[]
        with torch.inference_mode():
            for start in range(0,len(rows),32):
                images=[]
                for row in rows[start:start+32]:
                    path=_inside(root,row['relative_path'])
                    if sha256_file(path)!=row['sha256'] or decoded_pixel_sha256(path)!=row['decoded_pixel_sha256']:
                        raise ValueError('reference image content mismatch')
                    images.append(decode(path))
                inputs=processor(images=images,return_tensors='pt')
                pixels=inputs['pixel_values'].to(device=device,dtype=torch.float32)
                output=model(pixel_values=pixels).last_hidden_state[:,0,:]
                if output.shape != (len(images),768) or output.dtype!=torch.float32 or not torch.isfinite(output).all():
                    raise ValueError('invalid DINO CLS output')
                features.append(output.cpu())
        result=mine_confusers(torch.cat(features),[int(r['class_id']) for r in rows])
        value=hashlib.sha256(('features\0'+owner_token).encode()).hexdigest()
        with active_writer(target):
            token=claim_writer(target,'reference-features',token=value,resume=(target/'.writer').exists())
            try:
                stage=staging_path(target,token)
                for name,key in (('embeddings.npy','embeddings'),('centroids.npy','centroids'),('similarities.npy','similarities')):
                    out=io.BytesIO();np.save(out,result[key].numpy().astype(np.float32),allow_pickle=False)
                    complete_temporary(stage/name,out.getvalue())
                index={'class_ids':result['class_ids'],'confusers':result['confusers'],'model':metadata,'rows':[
                    {'row':i,'image_id':r['image_id'],'class_id':int(r['class_id']),'sha256':r['sha256']} for i,r in enumerate(rows)]}
                complete_temporary(stage/'feature_index.json',(json.dumps(index,sort_keys=True)+'\n').encode())
                feature_path=publish_lock(target,{'identity':'reference-features','files':['embeddings.npy','centroids.npy','similarities.npy','feature_index.json'],
                    'dependencies':dependencies,'artifact_root':root,'claim':token,'lock_name':'feature_lock.json'})
            finally:
                token.close()
    target=root/'augmentation/E3/A2'
    value=hashlib.sha256(('mining\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,'mining',token=value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token)
            out=io.BytesIO();np.save(out,result['similarities'].numpy().astype(np.float32),allow_pickle=False)
            complete_temporary(stage/'similarities.npy',out.getvalue())
            complete_temporary(stage/'confusers.json',(json.dumps({str(k):v for k,v in result['confusers'].items()},sort_keys=True)+'\n').encode())
            return publish_lock(target,{'identity':'mining','files':['similarities.npy','confusers.json'],
                'dependencies':{bank_path.relative_to(root).as_posix():sha256_file(bank_path),feature_path.relative_to(root).as_posix():sha256_file(feature_path)},
                'artifact_root':root,'claim':token,'lock_name':'mining_lock.json'})
        finally:
            token.close()
