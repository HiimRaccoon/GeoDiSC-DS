"""Shared native-resolution SD1.5 img2img inference primitives."""
import torch
from ..images import pad_native, crop_native

BASE_PROMPT = 'A realistic street-level road photograph in {country_name}, look like picture in Google Street View.'
SCHEDULER_SETTINGS = {'algorithm_type':'dpmsolver++', 'solver_order':2, 'solver_type':'midpoint',
                      'timestep_spacing':'linspace', 'lower_order_final':True, 'thresholding':False,
                      'use_karras_sigmas':False}


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


def generate_image(image, country, strength, *, vae, unet, tokenizer, text_encoder, scheduler, device, vae_seed, latent_seed):
    """One deterministic source/attempt, returning native RGB and provenance."""
    if strength not in (0.25, 0.35, 0.5) or not isinstance(country, str) or not country:
        raise ValueError('invalid strength or country')
    config = getattr(scheduler, 'config', None)
    if config is None or any(config.get(key) != value for key, value in SCHEDULER_SETTINGS.items()):
        raise ValueError('scheduler differs from confirmed configuration')
    for model in (vae, unet, text_encoder):
        if isinstance(model, torch.nn.Module):
            if model.training or any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError('generation models must use FP32 eval mode')
    padded, padding = pad_native(image)
    pixels = padded.unsqueeze(0).to(device=device, dtype=torch.float32)
    positive_prompt = BASE_PROMPT.format(country_name=country)
    with torch.inference_mode():
        baseline = _embedding(tokenizer, text_encoder, '', device)
        positive = _embedding(tokenizer, text_encoder, positive_prompt, device)
        posterior = vae.encode(pixels * 2 - 1).latent_dist
        vae_rng = torch.Generator(device=device).manual_seed(vae_seed)
        latents = posterior.sample(generator=vae_rng) * vae.config.scaling_factor
        if latents.dtype != torch.float32 or not torch.isfinite(latents).all():
            raise ValueError('invalid VAE latents')
        scheduler.set_timesteps(50, device=device)
        active_steps = min(int(50 * strength), 50)
        begin = 50 - active_steps
        timesteps = scheduler.timesteps[begin:]
        if hasattr(scheduler, 'set_begin_index'):
            scheduler.set_begin_index(begin)
        latent_rng = torch.Generator(device=device).manual_seed(latent_seed)
        noise = torch.randn(latents.shape, generator=latent_rng, device=device, dtype=torch.float32)
        latents = scheduler.add_noise(latents, noise, timesteps[:1])
        for timestep in timesteps:
            model_input = scheduler.scale_model_input(latents, timestep)
            unconditioned = unet(model_input, timestep, encoder_hidden_states=baseline).sample
            conditioned = unet(model_input, timestep, encoder_hidden_states=positive).sample
            predicted = unconditioned + 2.0 * (conditioned - unconditioned)
            if not torch.isfinite(predicted).all():
                raise ValueError('nonfinite diffusion prediction')
            latents = scheduler.step(predicted, timestep, latents).prev_sample
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        if decoded.dtype != torch.float32 or not torch.isfinite(decoded).all():
            raise ValueError('invalid VAE output')
        output = crop_native((decoded[0] / 2 + .5).clamp(0, 1), padding)
    return output, {'padding': padding, 'configured_steps': 50, 'active_steps': len(timesteps),
                    'timesteps': [int(t) for t in timesteps], 'strength': strength,
                    'vae_seed': vae_seed, 'latent_seed': latent_seed, 'prompt': positive_prompt}
