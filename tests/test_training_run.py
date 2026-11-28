import csv
import json
import torch
import pytest
from PIL import Image
from geobench.artifacts import claim_writer, staging_path, publish_lock, sha256_file, verify_lock
from geobench.images import decoded_pixel_sha256


def test_official_run_trains_resumes_and_publishes_verified_checkpoint(tmp_path, monkeypatch):
    from geobench import training
    from torch.utils.data import DataLoader
    root=tmp_path/'root'; root.mkdir(); dataset=tmp_path/'dataset'; dataset.mkdir()
    image=dataset/'a.png'; Image.new('RGB',(8,8),'red').save(image)
    entry={'image_id':'a','relative_path':'a.png','class_name':'x','class_id':'0','split':'train',
           'sha256':sha256_file(image),'decoded_pixel_sha256':decoded_pixel_sha256(image),'is_synthetic':'false'}
    def locked(relative, identity, files, dependencies=None, name='lock.json'):
        dest=root/relative; token=claim_writer(dest,identity); stage=staging_path(dest,token)
        for filename,content in files.items():
            path=stage/filename; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
        lock=publish_lock(dest,{'identity':identity,'files':list(files),'dependencies':dependencies or {},
                                'artifact_root':root,'claim':token,'lock_name':name})
        return {'path':lock.relative_to(root).as_posix(),'sha256':sha256_file(lock),'identity':identity}
    import io
    def table(rows):
        out=io.StringIO(newline=''); writer=csv.DictWriter(out,list(rows[0]));writer.writeheader();writer.writerows(rows);return out.getvalue().encode()
    groups=[{'class_id':str(c),'class_name':'x' if c==0 else str(c),'n_original':'2','n_train':'1' if c==0 else '0',
             'group':'extreme-tail','singleton':'false'} for c in range(124)]
    fingerprint='a'*64
    from geobench.config import verified_configuration
    readiness=locked('deployment/data_readiness','data-readiness',{'V01.json':json.dumps({'dataset_root':str(dataset)}).encode()})
    data=locked('splits','common-data:'+fingerprint,{'train.csv':table([entry]),'val.csv':table([{**entry,'split':'val'}]),
        'test.csv':table([{**entry,'split':'test'}]),
        'class_groups.csv':table(groups),'configuration.json':json.dumps(verified_configuration()).encode()}, {readiness['path']:readiness['sha256']})
    pool=locked('pools/E0','e0:'+fingerprint,{'pool.json':json.dumps({'manifest_sha256':sha256_file(root/'splits/train.csv')}).encode()},
                {data['path']:data['sha256']},'pool_lock.json')
    weights=locked('models/efficientnet','classifier-weights',{'weights.pt':b'fixture weights'})
    runtime=locked('runtime','shared-runtime',{'runtime.json':json.dumps({'weights':'models/efficientnet/weights.pt',
        'weights_sha256':sha256_file(root/'models/efficientnet/weights.pt')}).encode()}, {weights['path']:weights['sha256']})
    def model(): return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,124))
    buffer=io.BytesIO();torch.save(model().state_dict(),buffer)
    initial=locked('initializations/seed_42','initialization:42',{'initial_state.pt':buffer.getvalue()}, {runtime['path']:runtime['sha256']})
    spec=locked('specs/E0','spec:E0',{'spec.json':b'{}'})
    run=root/'runs/E0/seed_42';run.mkdir(parents=True)
    (run/'descriptor.json').write_text(json.dumps({'run_id':'E0/42','condition':'E0','seed':42,
        'dependencies':{'data':data,'pool':pool,'runtime':runtime,'initialization':initial,'spec':spec}}))
    monkeypatch.setattr(training,'_require_classifier_runtime',lambda seed: None)
    import geobench.classifier as classifier
    monkeypatch.setattr(classifier,'build_classifier',lambda *args,**kwargs:model())
    def fixture_loaders(train,val,*,seed):
        generator=torch.Generator().manual_seed(seed)
        return DataLoader(train,batch_size=1,shuffle=True,generator=generator),DataLoader(val,batch_size=1),generator
    import geobench.training_data as td
    monkeypatch.setattr(td,'make_loaders',fixture_loaders)
    attempts=[]
    def fixture_train(*args,**kwargs):
        kwargs['device']='cpu'
        if not attempts:
            attempts.append('interrupted')
            def interrupt(epoch):
                if epoch==2: raise RuntimeError('fixture interruption')
            kwargs['on_epoch']=interrupt
        return training._train_epochs(*args,**kwargs)
    monkeypatch.setattr(training,'train_epochs',fixture_train)
    import threading
    from geobench.artifacts import active_writer
    held=threading.Event(); release=threading.Event()
    def hold_root():
        with active_writer(root):
            held.set(); release.wait(timeout=30)
    holder=threading.Thread(target=hold_root); holder.start(); assert held.wait(timeout=10)
    try:
        with pytest.raises(RuntimeError, match='fixture interruption'):
            training.train_run(root,'E0/42',owner_token='fixture')
    finally:
        release.set(); holder.join(timeout=10)
    assert not (run/'checkpoint_lock.json').exists()
    import geobench.artifacts as artifacts
    original_link=artifacts.os.link
    def interrupt_publication(source,destination):
        if str(destination).endswith('checkpoint_lock.json'):
            raise OSError('fixture publication interruption')
        return original_link(source,destination)
    monkeypatch.setattr(artifacts.os,'link',interrupt_publication)
    with pytest.raises(ValueError,match='publish'):
        training.train_run(root,'E0/42',owner_token='fixture')
    assert not (run/'checkpoint_lock.json').exists()
    monkeypatch.setattr(artifacts.os,'link',original_link)
    lock=training.train_run(root,'E0/42',owner_token='fixture')
    record=verify_lock(lock,boundary=root)
    assert record['identity'].startswith('run:E0/42:')
    assert {'best_checkpoint.pt','last_checkpoint.pt','history.json','validation_at_best.json',
            'config.yaml','run_metadata.json','history.csv','lr_steps.csv','training_metrics.json'} <= set(record['files'])
    assert record['dependencies'][readiness['path']]==readiness['sha256']
    assert len(json.loads((run/'history.json').read_text()))==20
    with (run/'history.csv').open(newline='') as stream: history=list(csv.DictReader(stream))
    with (run/'lr_steps.csv').open(newline='') as stream: steps=list(csv.DictReader(stream))
    assert len(history)==len(steps)==20
    assert {'validation_loss','train_loss','samples','optimizer_steps'} <= set(history[0])
    assert {'step','epoch','step_in_epoch','lr'} <= set(steps[0])
    metadata=json.loads((run/'run_metadata.json').read_text())
    assert metadata['seed']==42 and metadata['counts']['real']==1
    assert any(event['event']=='failure' for event in metadata['execution_history'])
    assert any(event['event']=='resume' for event in metadata['execution_history'])
    assert json.loads((run/'metrics.json').read_text())['test'] is None
    assert training.train_run(root,'E0/42',owner_token='fixture')==lock
    from geobench import evaluation
    def predict_cpu(model,loader):
        model.eval()
        with torch.inference_mode(): return torch.cat([model(images) for images,labels in loader])
    monkeypatch.setattr(evaluation,'_predict',predict_cpu)
    evaluation_lock=evaluation.evaluate_run(root,'E0/42',owner_token='fixture')
    assert verify_lock(evaluation_lock,boundary=root)['files']['metrics.json']
    assert json.loads((run/'metrics.json').read_text())['test'] is not None
    assert evaluation.evaluate_run(root,'E0/42',owner_token='fixture')==evaluation_lock
    e1=locked('pools/E1','e1',{'manifest.csv':b'image_id\n'},
              {data['path']:data['sha256'],pool['path']:pool['sha256']},'pool_lock.json')
    locked('specs/E1E2','spec:E1E2',{'spec.json':b'{}'})
    for seed in (123,2026):
        locked(f'initializations/seed_{seed}',f'initialization:{seed}',{'initial_state.pt':buffer.getvalue()},
               {runtime['path']:runtime['sha256']})
    for strength in (.25,.35,.5):
        locked(f'pools/E1E2/candidates/strength_{strength}',f'candidate-pool:{strength}',
               {'manifest.csv':table([{**entry,'sample_type':'real'}]),'pool.json':b'{}'},
               {data['path']:data['sha256'],pool['path']:pool['sha256'],e1['path']:e1['sha256']},'pool_lock.json')
    from geobench.tuning import prepare_tuning
    protocol=prepare_tuning(root,owner_token='fixture')
    monkeypatch.setattr(td,'classifier_geometry',lambda image: torch.ones(3,8,8)*.5)
    references=[]
    for strength in (.25,.35,.5):
        for seed in (42,123,2026):
            tuning_lock=training.train_tuning(root,strength,seed,owner_token='fixture')
            tuning_record=verify_lock(tuning_lock,boundary=root)
            assert tuning_record['identity']==f'tuning:E2:{strength}:{seed}'
            assert tuning_record['dependencies']['tuning/E2/lock.json']==sha256_file(protocol)
            assert len(json.loads((tuning_lock.parent/'history.json').read_text()))==20
            references.append({'strength':strength,'seed':seed,'path':tuning_lock.relative_to(root).as_posix(),
                               'sha256':sha256_file(tuning_lock)})
    assert training.train_tuning(root,.25,42,owner_token='fixture')==root/references[0]['path']
    from geobench.selection import select_strength_from_locks
    assert select_strength_from_locks(root,references)['strength']==.25
    from geobench.selection import publish_strength_selection
    from geobench.bindings import freeze_selected_pool,prepare_run,bind_selected_runs
    publish_strength_selection(root,references,owner_token='fixture')
    official_pool=freeze_selected_pool(root,owner_token='fixture')
    assert verify_lock(official_pool,boundary=root)['identity']=='pool:E1E2'
    for seed in (42,123,2026): prepare_run(root,f'E1E2/{seed}',owner_token='fixture')
    bindings=bind_selected_runs(root,owner_token='fixture')
    assert len(verify_lock(bindings,boundary=root)['files'])==3
    evaluation.evaluate_run(root,'E1E2/42',owner_token='fixture')
    (run/'descriptor.json').write_text((run/'descriptor.json').read_text()+' ')
    with pytest.raises(ValueError,match='identity'):
        training.train_run(root,'E0/42',owner_token='fixture')


def test_training_cli_requires_owner(tmp_path,monkeypatch):
    from geobench.cli import main
    monkeypatch.delenv('GEOBENCH_OWNER_TOKEN',raising=False)
    with pytest.raises(ValueError,match='owner'):
        main(['train','--root',str(tmp_path),'--run','E0/42'])
