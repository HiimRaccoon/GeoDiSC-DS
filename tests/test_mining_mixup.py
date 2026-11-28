import pytest
import torch
from geobench.augmentation.mining import mine_confusers


def test_centroids_cosine_ties_exclude_self_and_choose_lowest_class_id():
    features=torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    result=mine_confusers(features,[0,0,1,2])
    assert result['confusers'] == {0:1, 1:0, 2:0}
    assert result['centroids'].shape == (3,2)
    assert result['similarities'].shape == (3,3)


def test_zero_or_nonfinite_embedding_blocks_mining():
    with pytest.raises(ValueError): mine_confusers(torch.tensor([[0.,0.],[1.,0.]]),[0,1])
    with pytest.raises(ValueError): mine_confusers(torch.tensor([[float('nan'),0.],[1.,0.]]),[0,1])


def test_mixup_changes_only_eligible_anchors_before_normalization():
    from geobench.augmentation.mixup import mix_batch
    batch={'images':torch.stack([torch.ones(3,2,2),torch.ones(3,2,2),torch.ones(3,2,2)]),
           'labels':torch.tensor([0,0,1]),'sample_type':['real','final_synthetic','final_synthetic'],
           'n_original':[2,2,1], 'n_train':[1,1,1], 'image_id':['r','s','single']}
    ctx={'selected':True,'real_pool':{0:{'images':torch.zeros(1,3,2,2),'ids':['partner']}},'training_seed':42,'epoch':0,'batch_index':0}
    result,diag=mix_batch(batch,ctx)
    assert torch.equal(result['labels'],batch['labels'])
    assert torch.equal(result['images'][0],batch['images'][0])
    assert torch.equal(result['images'][2],batch['images'][2])
    assert diag['mixed_count']==1 and 0 <= result['images'][1,0,0,0] <= 1
    assert diag['pairs'][0]['lambda'] in [i/20 for i in range(21)]


def test_mixup_selects_floor_half_batches_without_replacement():
    from geobench.augmentation.mixup import select_batches
    selected=select_batches(7,training_seed=42,epoch=0)
    assert len(selected)==3 and len(set(selected))==3
    assert selected==select_batches(7,training_seed=42,epoch=0)


@pytest.mark.parametrize('lambda_index,expected',[(0,1.0),(20,0.0)])
def test_mixup_lambda_endpoints_follow_real_weight(monkeypatch,lambda_index,expected):
    from geobench.augmentation.mixup import mix_batch
    batch={'images':torch.ones(1,3,2,2),'labels':torch.tensor([0]),'sample_type':['final_synthetic'],
           'n_original':[2],'n_train':[1],'image_id':['anchor']}
    draws=iter([0,lambda_index])
    monkeypatch.setattr(torch,'randint',lambda *args,**kwargs: torch.tensor([next(draws)]))
    ctx={'selected':True,'real_pool':{0:{'images':torch.zeros(1,3,2,2),'ids':['partner']}},'training_seed':42,'epoch':0,'batch_index':0}
    result,diag=mix_batch(batch,ctx)
    assert torch.all(result['images']==expected)


def test_mixup_pair_decisions_follow_anchor_identity_when_batch_order_changes():
    from geobench.augmentation.mixup import mix_batch
    images=torch.stack([torch.ones(3,2,2)*.8,torch.ones(3,2,2)*.6])
    batch={'images':images,'labels':torch.tensor([0,0]),'sample_type':['final_synthetic']*2,
           'n_original':[2,2],'n_train':[1,1],'image_id':['a','b']}
    ctx={'selected':True,'real_pool':{0:{'images':torch.stack([torch.zeros(3,2,2),torch.ones(3,2,2)*.3]),'ids':['r1','r2']}},
         'training_seed':42,'epoch':0,'batch_index':0}
    _, first=mix_batch(batch,ctx)
    reverse={**batch,'images':images.flip(0),'image_id':['b','a']}
    _, second=mix_batch(reverse,ctx)
    a={p['anchor_id']:(p['partner_id'],p['lambda']) for p in first['pairs']}
    b={p['anchor_id']:(p['partner_id'],p['lambda']) for p in second['pairs']}
    assert a==b
