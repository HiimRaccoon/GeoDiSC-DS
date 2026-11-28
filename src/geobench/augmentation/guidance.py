"""Numerical CADS and CCFG primitives in the confirmed Geo convention."""
import math
import torch
from .quality import TechnicalRejection


def active_time(index, count):
    if type(index) is not int or type(count) is not int or count < 1 or not 0 <= index < count:
        raise ValueError('invalid active timestep')
    return 0.0 if count == 1 else 1 - index / (count - 1)


def gamma_schedule(t):
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('invalid normalized time')
    if t <= .5: return 1.0
    if t >= .9: return 0.0
    return (.9-t)/.4


def corrupt_condition(condition, noise, gamma):
    condition = torch.as_tensor(condition, dtype=torch.float32)
    noise = torch.as_tensor(noise, dtype=torch.float32, device=condition.device)
    if condition.ndim != 3 or condition.shape != noise.shape or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('invalid condition or gamma')
    if not torch.isfinite(condition).all() or not torch.isfinite(noise).all():
        raise TechnicalRejection('nonfinite_condition')
    original_mean = condition.mean(dim=(1,2),keepdim=True)
    original_std = condition.std(dim=(1,2),correction=0,keepdim=True)
    corrupted = math.sqrt(gamma)*condition + .1*math.sqrt(1-gamma)*noise
    corrupted_mean = corrupted.mean(dim=(1,2),keepdim=True)
    corrupted_std = corrupted.std(dim=(1,2),correction=0,keepdim=True)
    if not torch.isfinite(original_std).all() or not torch.isfinite(corrupted_std).all() or (original_std == 0).any() or (corrupted_std == 0).any():
        raise TechnicalRejection('invalid_condition_std')
    result = (corrupted-corrupted_mean)/corrupted_std*original_std+original_mean
    if not torch.isfinite(result).all():
        raise TechnicalRejection('nonfinite_rescaled_condition')
    return result


def combine_predictions(baseline, positive, negative, *, gamma):
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    positive = torch.as_tensor(positive, dtype=torch.float32, device=baseline.device)
    negative = torch.as_tensor(negative, dtype=torch.float32, device=baseline.device)
    if baseline.ndim != 4 or baseline.shape != positive.shape or baseline.shape != negative.shape:
        raise ValueError('expected matching BCHW noise predictions')
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not all(torch.isfinite(x).all() for x in (baseline,positive,negative)):
        raise ValueError('invalid prediction or gamma')
    cads = baseline + 2.0 * (positive-baseline)
    d_pos = (baseline-cads).square().sum(dim=(1,2,3))
    d_neg = (baseline-negative).square().sum(dim=(1,2,3))
    if not torch.isfinite(d_pos).all() or not torch.isfinite(d_neg).all():
        raise TechnicalRejection('nonfinite_guidance_distance')
    tau = .8 * math.sqrt(gamma)
    w_pos = 4 * torch.sigmoid(tau*d_pos)
    w_neg = -4 * torch.sigmoid(-tau*d_neg)
    shape = (-1,1,1,1)
    ccfg = baseline + w_pos.view(shape)*(positive-baseline) + w_neg.view(shape)*(negative-baseline)
    output = .8*cads+.2*ccfg
    if not torch.isfinite(output).all():
        raise TechnicalRejection('nonfinite_guidance_output')
    return output, {'d_pos':d_pos,'d_neg':d_neg,'w_pos':w_pos,'w_neg':w_neg,'gamma':gamma,'temperature':tau}


def _condition_seed(role, class_id, slot, attempt, step, branch):
    import hashlib
    key = f'geo-e3-cads-v1|314159|{role}|{class_id}|{slot}|{attempt}|{step}|{branch}'
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')


def predict_step(context):
    """Evaluate A1/A2/A3 guidance branches for one active timestep."""
    required = {'unet','latent','timestep','index','count','positive','confuser','baseline',
                'role','class_id','slot','attempt','variant'}
    if not isinstance(context, dict) or set(context) != required:
        raise ValueError('invalid guidance context')
    variant, role = context['variant'], context['role']
    if variant not in {'A1','A2','A3'} or role not in {'reference','final'}:
        raise ValueError('invalid guidance role')
    if role == 'reference' and variant != 'A1':
        raise ValueError('references use CADS only')
    for key in ('class_id','slot','attempt'):
        if type(context[key]) is not int or context[key] < 0:
            raise ValueError('invalid guidance job identity')
    if context['attempt'] >= 5:
        raise ValueError('attempt outside confirmed cap')
    time = active_time(context['index'], context['count'])
    gamma = gamma_schedule(time)
    device = context['positive'].device
    def condition(embedding, branch):
        seed = _condition_seed(role,context['class_id'],context['slot'],context['attempt'],context['index'],branch)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(embedding.shape, generator=generator, device=device, dtype=torch.float32)
        corrupted=corrupt_condition(embedding,noise,gamma)
        stats={'original_mean':float(embedding.float().mean()),'original_std':float(embedding.float().std(correction=0)),
               'corrupted_mean':float(corrupted.mean()),'corrupted_std':float(corrupted.std(correction=0))}
        return corrupted, seed, stats
    positive, positive_seed,positive_stats = condition(context['positive'],'positive')
    baseline = context['unet'](context['latent'],context['timestep'],encoder_hidden_states=context['baseline']).sample
    raw_positive = context['unet'](context['latent'],context['timestep'],encoder_hidden_states=positive).sample
    diagnostics = {'t_norm':time,'gamma':gamma,'positive_noise_seed':positive_seed,'confuser_noise_seed':None,
                   'd_pos':None,'d_neg':None,'w_pos':None,'w_neg':None,'temperature':None,'alpha':None,
                   'positive_statistics':positive_stats,'confuser_statistics':None,
                   'baseline_type':'artifact_negative' if variant=='A3' else 'empty',
                   'baseline_norm':float(torch.linalg.vector_norm(baseline.float())),
                   'positive_prediction_norm':float(torch.linalg.vector_norm(raw_positive.float()))}
    if variant == 'A1':
        result = baseline.float() + 2*(raw_positive.float()-baseline.float())
    else:
        if context['confuser'] is None:
            raise ValueError('confuser embedding required')
        confuser, confuser_seed,confuser_stats = condition(context['confuser'],'confuser')
        raw_negative = context['unet'](context['latent'],context['timestep'],encoder_hidden_states=confuser).sample
        result, stats = combine_predictions(baseline,raw_positive,raw_negative,gamma=gamma)
        diagnostics.update({key: value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value for key,value in stats.items()})
        diagnostics['confuser_noise_seed'] = confuser_seed
        diagnostics['confuser_statistics']=confuser_stats
        diagnostics['confuser_prediction_norm']=float(torch.linalg.vector_norm(raw_negative.float()))
        diagnostics['alpha']=.8
    if not torch.isfinite(result).all():
        raise TechnicalRejection('nonfinite_guidance_result')
    return result, diagnostics
