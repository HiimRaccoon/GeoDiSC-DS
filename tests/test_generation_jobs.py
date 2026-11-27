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
