import csv, json
from PIL import Image
import pytest
import torch
from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock
from geobench.images import decoded_pixel_sha256
from geobench.augmentation.diffusion import generate_pool, build_hybrid_pool


def locked(root,relative,identity,files,dependencies=None,name='lock.json'):
    import hashlib
    target=root/relative
    value=hashlib.sha256(b'mining\0fixture').hexdigest() if identity=='mining' else None
    token=claim_writer(target,identity,token=value); stage=staging_path(target,token)
    for filename,content in files.items():
        path=stage/filename; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
    return publish_lock(target,{'identity':identity,'files':list(files),'dependencies':dependencies or {},'artifact_root':root,'claim':token,'lock_name':name})


def csv_bytes(fields,rows):
    import io
    out=io.StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n'); writer.writeheader();writer.writerows(rows)
    return out.getvalue().encode()


@pytest.mark.parametrize('class_count',[1,2])
def test_e2_pool_builds_four_synthetics_from_verified_training_source(tmp_path, monkeypatch,class_count):
    dataset=tmp_path/'dataset'; dataset.mkdir(); image=dataset/'one.png';Image.new('RGB',(8,8),(50,60,70)).save(image)
    row={'image_id':'source','relative_path':'one.png','class_name':'x','class_id':'0','split':'train','sha256':sha256_file(image),'decoded_pixel_sha256':decoded_pixel_sha256(image),'is_synthetic':'false'}
    fields=list(row); groups=[{'class_id':'0','class_name':'x','n_original':'2','n_train':'1','singleton':'false','group':'extreme-tail'}]
    train_rows=[row]
    if class_count==2:
        other=dataset/'two.png';Image.new('RGB',(8,8),(80,90,100)).save(other)
        train_rows.append({**row,'image_id':'source2','relative_path':'two.png','class_name':'y','class_id':'1',
                           'sha256':sha256_file(other),'decoded_pixel_sha256':decoded_pixel_sha256(other)})
        groups.append({**groups[0],'class_id':'1','class_name':'y'})
    root=tmp_path/'artifacts'
    from geobench.config import verified_configuration
    data=locked(root,'splits','common-data:fixture',{'train.csv':csv_bytes(fields,train_rows),'val.csv':csv_bytes(fields,[]),'test.csv':csv_bytes(fields,[]),'class_groups.csv':csv_bytes(list(groups[0]),groups),'configuration.json':json.dumps(verified_configuration()).encode()})
    e0=locked(root,'pools/E0','e0:fixture',{'pool.json':json.dumps({'manifest_sha256':sha256_file(root/'splits/train.csv')}).encode()},{'splits/lock.json':sha256_file(data)},'pool_lock.json')
    e1=locked(root,'pools/E1','e1',{'manifest.csv':b'image_id,class_id,relative_path,sample_type\n'},{'splits/lock.json':sha256_file(data),'pools/E0/pool_lock.json':sha256_file(e0)},'pool_lock.json')
    model=locked(root,'models/sd15','sd15-snapshot',{'model.json':json.dumps({'family':'stable-diffusion-v1-5','repository':'fixture/sd15','revision':'a'*40,'snapshot':'snapshot'}).encode(), 'snapshot/config.json':b'{}'})
    calls=[]
    def fake(image,country,strength,**kwargs):
        from geobench.augmentation.quality import TechnicalRejection
        calls.append((kwargs['vae_seed'],kwargs['latent_seed']))
        if len(calls)==1:
            raise TechnicalRejection('nonfinite_pixels')
        return torch.full((3,8,8),.1+len(calls)*.1), {'configured_steps':50,'active_steps':12,'timesteps':list(range(12))}
    class Tokenizer:
        def __call__(self,text,truncation=False): return {'input_ids':[1,2,3]}
    monkeypatch.setattr('geobench.augmentation.diffusion._load_runtime',lambda path: {'tokenizer':Tokenizer()})
    monkeypatch.setattr('geobench.augmentation.diffusion.generate_image',fake)
    descriptor={'strength':.25,'model_lock':'models/sd15/lock.json','model_sha256':sha256_file(model),'dataset_root':str(dataset)}
    lock=generate_pool(root,descriptor,owner_token='fixture')
    assert verify_lock(lock,boundary=root)['identity']=='e2:0.25'
    with (lock.parent/'manifest.csv').open(newline='') as stream: output=list(csv.DictReader(stream))
    assert len(output)==4*class_count and {r['source_id'] for r in output}=={r['image_id'] for r in train_rows} and len(calls)==4*class_count+1
    ledger=root/'augmentation/E2/strength_0.25/candidates.jsonl'
    events=[json.loads(line) for line in ledger.read_text().splitlines()]
    assert events[0]['reason']=='nonfinite_pixels' and events[1]['attempt']==1
    assert calls[0] != calls[1]
    assert generate_pool(root,descriptor,owner_token='fixture')==lock
    assert len(calls)==4*class_count+1
    hybrid=build_hybrid_pool(root,.25,owner_token='fixture')
    assert verify_lock(hybrid,boundary=root)['identity']=='candidate-pool:0.25'
    with (hybrid.parent/'manifest.csv').open(newline='') as stream: combined=list(csv.DictReader(stream))
    assert len(combined)==5*class_count and [row['sample_type'] for row in combined].count('real')==class_count
    locked(root,'tuning/E2/selection','selected-strength',{'selected_strength.json':b'{"selected_strength":0.25}'},
           {lock.relative_to(root).as_posix():sha256_file(lock)},'selection_lock.json')
    generated=[]
    def guided(image,country,strength,**kwargs):
        generated.append(kwargs)
        return torch.full((3,8,8),len(generated)/255), {'guidance':[{'gamma':1,'t_norm':0}]}
    monkeypatch.setattr('geobench.augmentation.diffusion.generate_image',guided)
    from geobench.augmentation.e3 import generate_e3
    bank=generate_e3(root,descriptor,variant='A1',reference=True,owner_token='fixture')
    assert verify_lock(bank,boundary=root)['identity']=='reference-bank'
    with (bank.parent/'manifest.csv').open(newline='') as stream: refs=list(csv.DictReader(stream))
    assert len(refs)==20*class_count and {row['sample_type'] for row in refs}=={'reference'}
    assert generated[0]['vae_seed'] != calls[0][0]
    assert generate_e3(root,descriptor,variant='A1',reference=True,owner_token='fixture')==bank
    assert len(generated)==20*class_count
    final=generate_e3(root,descriptor,variant='A1',owner_token='fixture')
    assert verify_lock(final,boundary=root)['identity']=='e3-generation:A1'
    with (final.parent/'manifest.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==4*class_count and {r['sample_type'] for r in rows}=={'final_synthetic'}
    assert generated[20*class_count]['vae_seed']==calls[0][0]
    from geobench.augmentation.e3 import build_e3_pool
    pool=build_e3_pool(root,'A1',owner_token='fixture')
    assert verify_lock(pool,boundary=root)['identity']=='pool:E3-A1'
    with (pool.parent/'manifest.csv').open(newline='') as stream: membership=list(csv.DictReader(stream))
    assert len(membership)==5*class_count and not any(r['sample_type']=='reference' for r in membership)
    if class_count==2:
        locked(root,'augmentation/E3/A2','mining',{'confusers.json':b'{"0":1,"1":0}'},
               {bank.relative_to(root).as_posix():sha256_file(bank)},'mining_lock.json')
        for variant in ('A2','A3'):
            generated_lock=generate_e3(root,descriptor,variant=variant,owner_token='fixture')
            hybrid=build_e3_pool(root,variant,owner_token='fixture')
            assert verify_lock(hybrid,boundary=root)['identity']==f'pool:E3-{variant}'
            assert generated[-1]['variant']==variant and generated[-1]['confuser_country']=='x'
            assert generate_e3(root,descriptor,variant=variant,owner_token='fixture')==generated_lock


def test_generation_never_writes_inside_dataset_root(tmp_path):
    descriptor={'strength':.25,'model_lock':'models/lock.json','model_sha256':'0'*64,'dataset_root':str(tmp_path)}
    with pytest.raises(ValueError,match='separate'):
        generate_pool(tmp_path,descriptor,owner_token='fixture')


def test_e2_cli_requires_owner_before_reading_descriptor(tmp_path,monkeypatch):
    from geobench.cli import main
    monkeypatch.delenv('GEOBENCH_OWNER_TOKEN',raising=False)
    with pytest.raises(ValueError,match='owner token'):
        main(['generate-e2','--root',str(tmp_path),'--descriptor',str(tmp_path/'missing.json')])
