import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from geobench.training import _train_epochs as train_epochs


def loaders():
    dataset=TensorDataset(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]),torch.tensor([0,1,0]))
    generator=torch.Generator().manual_seed(42)
    train=DataLoader(dataset,batch_size=2,shuffle=True,generator=generator)
    validation=DataLoader(dataset,batch_size=2,shuffle=False)
    return train,validation,generator


def test_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    def create(): return torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Dropout(.2),torch.nn.Linear(8,124))
    torch.manual_seed(1); initial=create().state_dict()
    def model():
        value=create(); value.load_state_dict(initial); return value
    a=model(); train,val,generator=loaders()
    torch.manual_seed(99)
    result=train_epochs(a,train,val,{'tail':[0,1]},generator,tmp_path/'full',identity='run-42')
    b=model(); train,val,generator=loaders()
    torch.manual_seed(99)
    def interrupt(epoch):
        if epoch == 3: raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        train_epochs(b,train,val,{'tail':[0,1]},generator,tmp_path/'resume',identity='run-42',on_epoch=interrupt)
    c=model(); train,val,generator=loaders()
    resumed=train_epochs(c,train,val,{'tail':[0,1]},generator,tmp_path/'resume',identity='run-42')
    assert result['history']==resumed['history'] and result['global_step']==40
    assert (tmp_path/'resume/best_checkpoint.pt').is_file()
    for left,right in zip(a.parameters(),c.parameters()): assert torch.equal(left,right)
    assert resumed['best_epoch'] == min(item['epoch'] for item in resumed['history'] if item['macro_f1']==max(x['macro_f1'] for x in resumed['history']))


def test_incompatible_checkpoint_identity_is_rejected(tmp_path):
    model=torch.nn.Linear(2,124); train,val,generator=loaders()
    def interrupt(epoch): raise RuntimeError('stop')
    with pytest.raises(RuntimeError): train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='first',on_epoch=interrupt)
    with pytest.raises(ValueError,match='identity'):
        train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='different')


def test_validation_rng_consumption_does_not_change_training(tmp_path):
    from torch.utils.data import Dataset
    class NoisyValidation(Dataset):
        def __len__(self): return 3
        def __getitem__(self,index):
            torch.rand(3)
            data=[([1.,0.],0),([0.,1.],1),([1.,1.],0)]
            return torch.tensor(data[index][0]),data[index][1]
    torch.manual_seed(1)
    reference=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Dropout(.2),torch.nn.Linear(8,124))
    initial=reference.state_dict()
    results=[]
    for use_noisy in (False,True):
        model=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Dropout(.2),torch.nn.Linear(8,124)); model.load_state_dict(initial)
        train,val,generator=loaders()
        if use_noisy: val=DataLoader(NoisyValidation(),batch_size=2,shuffle=False)
        torch.manual_seed(99)
        train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path/str(use_noisy),identity='fixture')
        results.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.equal(a,b) for a,b in zip(*results))


def test_official_trainer_rejects_nonbenchmark_loader(tmp_path):
    from geobench.training import train_epochs as official_train
    model=torch.nn.Linear(2,124); train,val,generator=loaders()
    with pytest.raises(ValueError,match='loader'):
        official_train(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='fixture')


def test_checkpoint_loading_is_weights_only(tmp_path, monkeypatch):
    model=torch.nn.Linear(2,124); train,val,generator=loaders()
    def interrupt(epoch): raise RuntimeError('stop')
    with pytest.raises(RuntimeError): train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='first',on_epoch=interrupt)
    original=torch.load
    def guarded(*args,**kwargs):
        assert kwargs.get('weights_only') is True
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',guarded)
    with pytest.raises(ValueError,match='identity'):
        train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='different')


def test_checkpoint_has_version_and_rejects_unrecognized_schema(tmp_path):
    model=torch.nn.Linear(2,124); train,val,generator=loaders()
    def interrupt(epoch): raise RuntimeError('stop')
    with pytest.raises(RuntimeError):
        train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='fixture',on_epoch=interrupt)
    path=tmp_path/'last_checkpoint.pt'
    checkpoint=torch.load(path,weights_only=True)
    assert checkpoint['schema_version']==1
    checkpoint['schema_version']=2
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match='schema'):
        train_epochs(model,train,val,{'tail':[0,1]},generator,tmp_path,identity='fixture')


def test_a4_training_mixes_anchor_metadata_before_normalization(tmp_path, monkeypatch):
    from torch.utils.data import Dataset
    import geobench.augmentation.mixup as mixing
    class Anchors(Dataset):
        def __len__(self): return 2
        def __getitem__(self, index):
            return {'images':torch.full((3,2,2), .25), 'labels':0, 'image_id':str(index),
                    'sample_type':'final_synthetic', 'n_original':2, 'n_train':1}
        def real_pool(self, labels):
            return {0:{'images':torch.full((1,3,2,2),.75),'ids':['real']}}
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer=torch.nn.Linear(12,124); self.seen=[]
        def forward(self,x):
            self.seen.append(x.detach().clone()); return self.layer(x.flatten(1))
    generator=torch.Generator().manual_seed(42)
    train=DataLoader(Anchors(),batch_size=1,shuffle=True,generator=generator)
    validation=DataLoader(TensorDataset(torch.zeros(1,3,2,2),torch.tensor([0])),batch_size=1)
    calls=[]; original=mixing.mix_batch
    def checked(batch, context):
        assert torch.all(batch['images']==.25) and len(batch['image_id'])==1
        result, evidence=original(batch,context); calls.append(evidence); return result,evidence
    monkeypatch.setattr(mixing,'mix_batch',checked)
    model=Network()
    def interrupt(epoch): raise RuntimeError('stop')
    with pytest.raises(RuntimeError,match='stop'):
        train_epochs(model,train,validation,{'tail':[0]},generator,tmp_path,identity='a4',on_epoch=interrupt,mixup_seed=42)
    assert len(calls)==2 and sum(item['selected'] for item in calls)==1
    assert sum(item['mixed_count'] for item in calls)==1
    assert not torch.all(model.seen[0]==.25)
    saved=torch.load(tmp_path/'last_checkpoint.pt',weights_only=True)
    assert saved['mixup_seed']==42 and len(saved['mixup_history'])==1
    assert saved['mixup_history'][0]['selected_batches']==[i for i, event in enumerate(calls) if event['selected']]
