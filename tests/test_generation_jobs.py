import pytest
from geobench.augmentation.jobs import seed_for, source_for_slot, first_valid


def test_shuffle_cycles_and_paired_attempt_seeds():
    sources=['b','a','c']
    assert set(source_for_slot(sources, 2, slot) for slot in range(3)) == set(sources)
    assert source_for_slot(sources,2,3) == source_for_slot(list(reversed(sources)),2,3)
    assert seed_for('E2',2,7,'vae') == seed_for('E2',2,7,'vae')
    assert seed_for('E2',2,7,'vae') != seed_for('E2',2,7,'latent')


def test_first_valid_and_shortage():
    attempts=[]
    def attempt(i):
        attempts.append(i)
        return i == 2
    assert first_valid(attempt) == 2
    assert attempts == [0,1,2]
    with pytest.raises(ValueError): first_valid(lambda i: False)


def test_training_group_contract_rejects_duplicate_or_mislabeled_sources():
    from geobench.augmentation.jobs import validate_training_groups
    row={'class_id':'0','class_name':'A','image_id':'id','split':'train','is_synthetic':'false'}
    group={'class_id':'0','class_name':'A','n_train':'1','n_original':'2','singleton':'false','group':'extreme-tail'}
    assert len(validate_training_groups([row],[group])[0])==1
    with pytest.raises(ValueError): validate_training_groups([row,row],[group])
    with pytest.raises(ValueError): validate_training_groups([{**row,'class_name':'B'}],[group])


def test_generation_requires_matching_locked_research_config(tmp_path):
    from geobench.augmentation.jobs import validate_shared_research
    from geobench.config import verified_configuration
    from geobench.artifacts import claim_writer,staging_path,publish_lock
    import json
    config=verified_configuration(); folder=tmp_path/'splits'; token=claim_writer(folder,'common-data:fixture'); stage=staging_path(folder,token)
    (stage/'configuration.json').write_text(json.dumps(config))
    lock=publish_lock(folder,{'identity':'common-data:fixture','files':['configuration.json'],'dependencies':{},'claim':token})
    assert validate_shared_research(tmp_path,lock)['sha256']['methods']==config['sha256']['methods']
    changed=json.loads((folder/'configuration.json').read_text()); changed['values']['benchmark']['classifier']['batch_size']=64
    (folder/'configuration.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError): validate_shared_research(tmp_path,lock)
