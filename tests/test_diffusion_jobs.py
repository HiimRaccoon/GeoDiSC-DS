import torch
import pytest
from PIL import Image
from geobench.augmentation.diffusion import generate_image


class Posterior:
    def sample(self, generator=None): return torch.zeros(1,4,2,2)
class VAE:
    class config: scaling_factor=1.0
    def encode(self, x):
        assert x.shape == (1,3,16,16)
        return type('Enc',(),{'latent_dist':Posterior()})()
    def decode(self, x): return type('Dec',(),{'sample':torch.zeros(1,3,16,16)})()
class Tokenizer:
    model_max_length=77
    def __call__(self, text, **kwargs):
        return {'input_ids':torch.ones(1,77,dtype=torch.long)}
class Text:
    def __call__(self, ids): return (torch.zeros(1,77,2),)
class UNet:
    def __call__(self, latent, timestep, encoder_hidden_states=None):
        return type('Pred',(),{'sample':torch.zeros_like(latent)})()
class Scheduler:
    def __init__(self):
        self.calls=0
        self.config={'algorithm_type':'dpmsolver++','solver_order':2,'solver_type':'midpoint',
                     'timestep_spacing':'linspace','lower_order_final':True,'thresholding':False,'use_karras_sigmas':False}
    def set_timesteps(self,n,device=None): self.timesteps=torch.arange(n-1,-1,-1)
    def set_begin_index(self,i): self.begin=i
    def add_noise(self,latents,noise,timestep): return latents+noise
    def scale_model_input(self,latents,timestep): return latents
    def step(self,pred,timestep,latents):
        self.calls+=1
        return type('Step',(),{'prev_sample':latents})()


def test_native_geometry_active_steps_and_seed_replay():
    image=Image.new('RGB',(15,15),'white'); scheduler=Scheduler()
    args=dict(vae=VAE(),unet=UNet(),tokenizer=Tokenizer(),text_encoder=Text(),scheduler=scheduler,device='cpu',vae_seed=3,latent_seed=4)
    output, metadata=generate_image(image,'Example',.25,**args)
    assert output.shape==(3,15,15)
    assert metadata['active_steps']==12 and scheduler.calls==12 and scheduler.begin==38
    assert torch.allclose(output,torch.full((3,15,15),.5))


def test_invalid_strength_fails_before_model_call():
    with pytest.raises(ValueError):
        generate_image(Image.new('RGB',(15,15)), 'Example', .6, vae=None,unet=None,tokenizer=None,text_encoder=None,scheduler=None,device='cpu',vae_seed=1,latent_seed=2)


def test_latent_noise_is_paired_across_strengths():
    class Capture(Scheduler):
        def add_noise(self, latents, noise, timestep):
            self.noise=noise.clone()
            return super().add_noise(latents,noise,timestep)
    captures=[]
    for strength in (.25,.35,.5):
        scheduler=Capture()
        generate_image(Image.new('RGB',(15,15)), 'Example', strength, vae=VAE(), unet=UNet(), tokenizer=Tokenizer(), text_encoder=Text(), scheduler=scheduler, device='cpu', vae_seed=3, latent_seed=4)
        captures.append(scheduler.noise)
    assert all(torch.equal(captures[0], noise) for noise in captures[1:])


def test_prompt_overflow_blocks_before_vae():
    class TooLong(Tokenizer):
        def __call__(self,text,**kwargs): return {'input_ids':torch.ones(1,78,dtype=torch.long)}
    with pytest.raises(ValueError,match='prompt exceeds'):
        generate_image(Image.new('RGB',(15,15)), 'Example', .25, vae=None,unet=UNet(),tokenizer=TooLong(),text_encoder=Text(),scheduler=Scheduler(),device='cpu',vae_seed=1,latent_seed=2)


def test_sampler_rejects_incompatible_scheduler():
    scheduler=Scheduler()
    scheduler.config['solver_order']=1
    with pytest.raises(ValueError,match='scheduler'):
        generate_image(Image.new('RGB',(15,15)), 'Example', .25, vae=VAE(),unet=UNet(),tokenizer=Tokenizer(),text_encoder=Text(),scheduler=scheduler,device='cpu',vae_seed=1,latent_seed=2)


def test_sampler_rejects_train_mode_module():
    model=torch.nn.Linear(2,2)
    assert model.training
    with pytest.raises(ValueError,match='FP32 eval'):
        generate_image(Image.new('RGB',(15,15)), 'Example', .25, vae=model,unet=UNet(),tokenizer=Tokenizer(),text_encoder=Text(),scheduler=Scheduler(),device='cpu',vae_seed=1,latent_seed=2)
