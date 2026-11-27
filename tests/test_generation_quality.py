import pytest
from geobench.augmentation.quality import audit_candidate, HoldoutCollision


def test_holdout_collision_blocks_even_if_training_duplicate():
    with pytest.raises(HoldoutCollision):
        audit_candidate('file','pixel', train_files={'file'}, train_pixels=set(), holdout_files=set(), holdout_pixels={'pixel'}, accepted_files=set(), accepted_pixels=set())


def test_training_and_pool_duplicates_reject_only_candidate():
    args={'train_files':{'train'}, 'train_pixels':{'t'}, 'holdout_files':set(), 'holdout_pixels':set(), 'accepted_files':{'done'}, 'accepted_pixels':{'p'}}
    assert audit_candidate('train','x',**args) == 'training_exact_duplicate'
    assert audit_candidate('new','p',**args) == 'pool_exact_duplicate'
    assert audit_candidate('new','unique',**args) is None
