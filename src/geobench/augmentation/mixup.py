"""Same-class real–synthetic image mixing before normalization."""
import hashlib
import torch


def _seed(training_seed, epoch, stream, batch_index=None, anchor_id=None):
    if type(training_seed) is not int or training_seed not in (42,123,2026) or type(epoch) is not int or epoch < 0:
        raise ValueError('invalid Mixup training seed or epoch')
    key = f'geo-e3-mixup-v1|{training_seed}|{epoch}|{stream}'
    if batch_index is not None:
        key += f'|{batch_index}|{anchor_id}'
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')


def select_batches(number_of_batches, *, training_seed, epoch):
    if type(number_of_batches) is not int or number_of_batches < 0:
        raise ValueError('invalid batch count')
    generator=torch.Generator().manual_seed(_seed(training_seed,epoch,'batch_selection'))
    return sorted(torch.randperm(number_of_batches, generator=generator)[:number_of_batches // 2].tolist())


def mix_batch(batch, context):
    images = batch['images']; labels = batch['labels']; count = len(images)
    if images.ndim != 4 or images.shape[1] != 3 or len(labels) != count or any(len(batch[key]) != count for key in ('sample_type','n_original','n_train','image_id')):
        raise ValueError('invalid batch shapes')
    if len(set(batch['image_id'])) != count or any(not isinstance(x,str) or not x for x in batch['image_id']):
        raise ValueError('anchor IDs must be unique strings')
    if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError('mixup requires pre-normalized RGB values')
    if type(context['selected']) is not bool or type(context['batch_index']) is not int or context['batch_index'] < 0:
        raise ValueError('batch selection and index must be explicit')
    _seed(context['training_seed'], context['epoch'], 'validate')
    output = images.clone(); pairs = []
    if context['selected']:
        for index in range(count):
            if batch['sample_type'][index] != 'final_synthetic' or batch['n_original'][index] == 1 or batch['n_train'][index] >= 20:
                continue
            class_id = int(labels[index]); pool = context['real_pool'].get(class_id)
            if not isinstance(pool,dict) or set(pool) != {'images','ids'}:
                raise ValueError('eligible synthetic anchor requires same-class real pool')
            real, ids = pool['images'], pool['ids']
            if (real.ndim != 4 or len(real) == 0 or real.shape[1:] != images.shape[1:]
                    or len(ids) != len(real) or len(set(ids)) != len(ids) or any(not isinstance(x,str) or not x for x in ids)):
                raise ValueError('invalid real partner pool')
            if not torch.isfinite(real).all() or (real < 0).any() or (real > 1).any():
                raise ValueError('invalid real partner pixels')
            anchor = batch['image_id'][index]
            pseed = _seed(context['training_seed'],context['epoch'],'partner',context['batch_index'],anchor)
            lseed = _seed(context['training_seed'],context['epoch'],'lambda',context['batch_index'],anchor)
            partner_index = int(torch.randint(len(real),(1,),generator=torch.Generator().manual_seed(pseed)).item())
            weight_index = int(torch.randint(21,(1,),generator=torch.Generator().manual_seed(lseed)).item())
            weight = weight_index / 20
            partner = real[partner_index].to(device=images.device,dtype=images.dtype)
            output[index] = weight * partner + (1-weight) * images[index]
            pairs.append({'anchor_index':index,'anchor_id':anchor,'class_id':class_id,'partner_index':partner_index,
                          'partner_id':ids[partner_index], 'lambda':weight, 'partner_seed':pseed,'lambda_seed':lseed})
    return {**batch,'images':output}, {'selected':context['selected'],'mixed_count':len(pairs),'pairs':pairs}
