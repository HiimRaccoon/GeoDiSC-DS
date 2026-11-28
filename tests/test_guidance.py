import pytest
import torch
from geobench.augmentation.guidance import active_time, gamma_schedule, corrupt_condition, combine_predictions


def test_schedule_endpoints_and_single_step():
    assert active_time(0,1)==0
    assert active_time(0,3)==1 and active_time(2,3)==0
    assert gamma_schedule(1)==0 and gamma_schedule(.5)==1
    assert gamma_schedule(.7)==pytest.approx(.5)


def test_condition_statistics_are_per_sample_population():
    condition=torch.tensor([[[0.,1.],[2.,3.]], [[10.,12.],[14.,16.]]])
    noise=torch.tensor([[[1.,-1.],[.5,-.5]], [[-1.,2.],[0.,1.]]])
    output=corrupt_condition(condition,noise,.5)
    assert torch.allclose(output.mean((1,2)),condition.mean((1,2)),atol=1e-6)
    assert torch.allclose(output.std((1,2),correction=0),condition.std((1,2),correction=0),atol=1e-6)
    with pytest.raises(ValueError): corrupt_condition(torch.ones(1,2,2),noise[:1],.5)


def test_positive_distance_uses_cfg_sum():
    baseline=torch.zeros(1,1,1,2); positive=torch.ones_like(baseline); negative=torch.zeros_like(baseline)
    result,stats=combine_predictions(baseline,positive,negative,gamma=1)
    assert stats['d_pos'].item()==8
    assert torch.isfinite(result).all()


def test_predict_step_uses_independent_condition_streams_and_geo_baseline():
    from geobench.augmentation.guidance import predict_step
    class Net:
        def __init__(self): self.conditions=[]
        def __call__(self, latent, timestep, encoder_hidden_states):
            self.conditions.append(encoder_hidden_states.clone())
            value=encoder_hidden_states.mean()
            return type('Prediction',(),{'sample':torch.ones_like(latent)*value})()
    positive=torch.tensor([[[0.,1.],[2.,3.]]]); confuser=positive+10
    net=Net(); context={'unet':net,'latent':torch.zeros(1,1,2,2),'timestep':99,'index':0,'count':3,
                        'positive':positive,'confuser':confuser,'baseline':torch.ones_like(positive),
                        'role':'final','class_id':1,'slot':2,'attempt':0,'variant':'A3'}
    result,diagnostics=predict_step(context)
    assert len(net.conditions)==3 and torch.equal(net.conditions[0],context['baseline'])
    assert diagnostics['positive_noise_seed'] != diagnostics['confuser_noise_seed']
    assert diagnostics['gamma']==0 and result.shape==(1,1,2,2)
    second,details=predict_step({**context,'unet':Net()})
    assert torch.equal(result,second) and details['positive_noise_seed']==diagnostics['positive_noise_seed']


def test_invalid_gamma_and_zero_corrupted_statistics_rejected():
    condition=torch.tensor([[[0.,1.],[2.,3.]]])
    with pytest.raises(ValueError): corrupt_condition(condition,torch.ones_like(condition),1.1)
    with pytest.raises(ValueError): corrupt_condition(condition,torch.ones_like(condition),0.)
    with pytest.raises(ValueError): combine_predictions(torch.zeros(1,1,1,1),torch.ones(1,1,1,1),torch.zeros(1,1,1,1),gamma=float('nan'))
