import pytest
import torch
from geobench.augmentation.traditional import select_transforms, apply_transforms


def test_selection_is_unique_ordered_and_reproducible():
    for seed in range(50):
        selected = select_transforms(seed)
        assert 1 <= len(selected) <= 2
        assert selected == sorted(set(selected))
        assert selected == select_transforms(seed)


def test_brightness_contrast_use_full_image_mean_and_clip():
    x=torch.tensor([[[.9,.1]], [[.9,.1]], [[.9,.1]]])
    actual=apply_transforms(x, [('brightness', 1.1), ('contrast', .9)])
    assert torch.allclose(actual, torch.tensor([[[.946,.154]], [[.946,.154]], [[.946,.154]]]), atol=1e-6)


def test_rejects_unsupported_or_out_of_order_transforms():
    with pytest.raises(ValueError):
        apply_transforms(torch.zeros(3,8,8), [('rotation', 0), ('brightness', 1)])


def test_blur_preserves_constant_and_rotation_preserves_canvas():
    x=torch.full((3,9,11), .5)
    assert torch.allclose(apply_transforms(x,[('blur', .6)]),x,atol=1e-6)
    assert torch.allclose(apply_transforms(x,[('rotation', 3)]),x,atol=1e-6)


def test_noise_replays_from_independent_seed_and_quantizes_half_up():
    from geobench.augmentation.traditional import quantize
    x=torch.full((3,8,8),.5)
    a=apply_transforms(x,[('noise',2/255)],generator=torch.Generator().manual_seed(42))
    b=apply_transforms(x,[('noise',2/255)],generator=torch.Generator().manual_seed(42))
    assert torch.equal(a,b)
    assert quantize(torch.tensor([.5,1.0,0.0])).tolist()==[128,255,0]


def test_rotation_zero_keeps_nonsquare_image_pixels():
    x=torch.arange(3*5*7,dtype=torch.float32).reshape(3,5,7)/104
    assert torch.allclose(apply_transforms(x,[('rotation',0)]),x,atol=1e-6)


def test_invalid_inputs_and_out_of_range_parameters():
    with pytest.raises(ValueError): apply_transforms(torch.ones(3,4,4), [('brightness', 2)])
    with pytest.raises(ValueError): apply_transforms(torch.ones(3,4,4), [('brightness', 1), ('brightness', 1)])


def test_e1_pool_reuses_verified_real_training_sources(tmp_path):
    import csv, json
    from PIL import Image
    from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock
    from geobench.images import decoded_pixel_sha256
    from geobench.config import verified_configuration
    from geobench.augmentation.traditional import generate_e1
    dataset=tmp_path/'dataset'; dataset.mkdir()
    Image.new('RGB',(12,8),(100,120,140)).save(dataset/'a.png')
    root=tmp_path/'artifacts'; split=root/'splits'; token=claim_writer(split,'common-data:fixture')
    stage=staging_path(split,token)
    fields=['image_id','relative_path','class_name','class_id','split','sha256','decoded_pixel_sha256','duplicate_group_id','is_synthetic']
    row={'image_id':'source','relative_path':'a.png','class_name':'a','class_id':'0','split':'train','sha256':sha256_file(dataset/'a.png'),'decoded_pixel_sha256':decoded_pixel_sha256(dataset/'a.png'),'duplicate_group_id':'source','is_synthetic':'false'}
    for name in ('train','val','test'):
        with (stage/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            if name=='train': writer.writerow(row)
    with (stage/'class_groups.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['class_id','class_name','n_original','n_train','singleton','group']);writer.writeheader();writer.writerow({'class_id':0,'class_name':'a','n_original':20,'n_train':20,'singleton':'false','group':'medium'})
    (stage/'configuration.json').write_text(json.dumps(verified_configuration()))
    data_lock=publish_lock(split,{'identity':'common-data:fixture','files':['train.csv','val.csv','test.csv','class_groups.csv','configuration.json'],'dependencies':{},'claim':token})
    e0=root/'pools/E0'; token=claim_writer(e0,'e0:fixture'); stage=staging_path(e0,token)
    (stage/'pool.json').write_text(json.dumps({'manifest_sha256':sha256_file(split/'train.csv')}))
    e0_lock=publish_lock(e0,{'identity':'e0:fixture','files':['pool.json'],'dependencies':{'splits/lock.json':sha256_file(data_lock)},'artifact_root':root,'claim':token,'lock_name':'pool_lock.json'})
    with pytest.raises(ValueError,match='count'):
        generate_e1(root,dataset_root=dataset,owner_token='fixture')


def test_e1_publishes_exact_quota_with_locked_real_membership(tmp_path, monkeypatch):
    import csv, json
    from PIL import Image
    from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock
    from geobench.images import decoded_pixel_sha256
    from geobench.config import verified_configuration
    from geobench.augmentation.traditional import generate_e1
    import geobench.augmentation.quality as quality
    original_validate = quality.validate_candidate_tensor
    calls = [0]
    def reject_first(candidate, expected):
        calls[0] += 1
        return 'nonfinite_pixels' if calls[0] == 1 else original_validate(candidate, expected)
    monkeypatch.setattr(quality, 'validate_candidate_tensor', reject_first)
    dataset=tmp_path/'dataset'; dataset.mkdir(); rows=[]
    for i in range(99):
        path=dataset/f'{i:03}.png'
        image=Image.new('RGB',(8,8)); image.putdata([((i*2+j)%256,(i+j*3)%256,(i*7+j*5)%256) for j in range(64)]);image.save(path)
        rows.append({'image_id':f'id{i}','relative_path':path.name,'class_name':'a','class_id':'0','split':'train','sha256':sha256_file(path),'decoded_pixel_sha256':decoded_pixel_sha256(path),'duplicate_group_id':f'id{i}','is_synthetic':'false'})
    root=tmp_path/'artifacts'; split=root/'splits'; token=claim_writer(split,'common-data:fixture'); stage=staging_path(split,token)
    fields=list(rows[0])
    for name in ('train','val','test'):
        with (stage/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            if name=='train': writer.writerows(rows)
    with (stage/'class_groups.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['class_id','class_name','n_original','n_train','singleton','group']);writer.writeheader();writer.writerow({'class_id':0,'class_name':'a','n_original':120,'n_train':99,'singleton':'false','group':'medium'})
    (stage/'configuration.json').write_text(json.dumps(verified_configuration()))
    data_lock=publish_lock(split,{'identity':'common-data:fixture','files':['train.csv','val.csv','test.csv','class_groups.csv','configuration.json'],'dependencies':{},'claim':token})
    e0=root/'pools/E0'; token=claim_writer(e0,'e0:fixture'); stage=staging_path(e0,token)
    (stage/'pool.json').write_text(json.dumps({'manifest_sha256':sha256_file(split/'train.csv')}))
    publish_lock(e0,{'identity':'e0:fixture','files':['pool.json'],'dependencies':{'splits/lock.json':sha256_file(data_lock)},'artifact_root':root,'claim':token,'lock_name':'pool_lock.json'})
    lock=generate_e1(root,dataset_root=dataset,owner_token='fixture')
    assert verify_lock(lock,boundary=root)['identity'] == 'e1'
    with (lock.parent/'manifest.csv').open(newline='') as stream: outputs=list(csv.DictReader(stream))
    assert len(outputs)==1 and outputs[0]['sample_type']=='traditional'
    assert outputs[0]['source_id'] in {r['image_id'] for r in rows}
    assert lock == root/'pools/E1/pool_lock.json'
    ledger=root/'augmentation/E1/.staging/candidates.jsonl'
    complete=ledger.read_bytes(); ledger.write_bytes(complete[:10])
    assert json.loads(complete.splitlines()[0])['reason']=='nonfinite_pixels'
    calls[0] = 0
    assert generate_e1(root,dataset_root=dataset,owner_token='fixture')==lock
    assert ledger.read_bytes()==complete
    previous_calls = calls[0]
    assert generate_e1(root,dataset_root=dataset,owner_token='fixture')==lock
    assert calls[0] == previous_calls


def test_nonfinite_candidate_is_a_technical_rejection():
    from geobench.augmentation.quality import validate_candidate_tensor
    assert validate_candidate_tensor(torch.full((3,4,4),float('nan')), (4,4)) == 'nonfinite_pixels'
    assert validate_candidate_tensor(torch.zeros(3,3,4), (4,4)) == 'geometry_mismatch'
    assert validate_candidate_tensor(torch.zeros(3,4,4), (4,4)) is None


def test_e1_cli_checks_owner_before_pool_access(tmp_path, monkeypatch):
    from geobench.cli import main
    monkeypatch.delenv('GEOBENCH_OWNER_TOKEN', raising=False)
    with pytest.raises(ValueError, match='owner token'):
        main(['generate-e1','--root',str(tmp_path)])


def test_policy_identity_changes_when_backend_changes(tmp_path, monkeypatch):
    from geobench.augmentation.traditional import e1_policy
    import geobench.augmentation.traditional as traditional
    first=e1_policy()
    monkeypatch.setattr(traditional, '__file__', str(tmp_path/'changed.py'))
    (tmp_path/'changed.py').write_text('different source')
    second=e1_policy()
    assert first['code_sha256'] != second['code_sha256']
    assert first['backend']['rng'] == 'python.random.Random'
