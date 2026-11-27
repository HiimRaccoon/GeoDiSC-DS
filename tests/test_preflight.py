import pytest
from geobench.preflight import validate_profile, readiness_by_role


def valid():
    return {'artifact_root':'/tmp/benchmark','cpu_count':8,'memory':'64G','walltime':'3-00:00:00',
            'account':'uit','partition':'normal','qos':'mig3-71','gres':'gpu:3g.71gb:1'}


@pytest.mark.parametrize('field', ['artifact_root','cpu_count','memory','walltime'])
def test_unknown_resources_are_rejected(field):
    profile=valid(); profile[field]=None
    with pytest.raises(ValueError): validate_profile(profile)


def test_invalid_gpu_pair_and_research_override_rejected():
    profile=valid(); profile['gres']='gpu:7g.141gb:1'
    with pytest.raises(ValueError): validate_profile(profile)
    profile=valid(); profile['batch_size']=32
    with pytest.raises(ValueError): validate_profile(profile)


def test_missing_dino_does_not_block_classifier_and_diffusion():
    roles=readiness_by_role({'site':True,'storage':True,'base_models':True,'dino':False})
    assert roles['classifier'] and roles['diffusion'] and not roles['dino']
