import json
import pytest
from geobench.artifacts import publish_records,sha256_file
from geobench.slurm import freeze_array,execute_job


def test_array_identity_and_incomplete_stage_rejection(tmp_path,monkeypatch):
    dep=publish_records(tmp_path,'dep','dependency',{'input.json':b'{}'},{},owner_token='fixture')
    tasks=[{'id':'E0/42','argv':['train','--run','E0/42'],'dependencies':{'dep/lock.json':sha256_file(dep)},
            'outputs':['out/lock.json']}]
    path=freeze_array(tmp_path,'train',tasks,owner_token='fixture')
    assert json.loads((path.parent/'array.json').read_text())==tasks
    calls=[]
    def fake(argv,**kwargs):
        calls.append((argv,kwargs));return None
    monkeypatch.setattr('geobench.slurm.subprocess.run',fake)
    with pytest.raises(ValueError):execute_job(tmp_path,path.relative_to(tmp_path).as_posix(),0)
    assert calls[0][1]['env']['PYTHONHASHSEED']=='42'
    publish_records(tmp_path,'out','run',{'done.json':b'{}'},{},owner_token='fixture')
    assert execute_job(tmp_path,path.relative_to(tmp_path).as_posix(),0)==0
    with pytest.raises(ValueError):freeze_array(tmp_path,'bad',tasks+tasks,owner_token='fixture')
    with pytest.raises(ValueError):execute_job(tmp_path,path.relative_to(tmp_path).as_posix(),1)


def test_submission_requires_separate_scientific_approval(tmp_path):
    from geobench.slurm import submit_array
    with pytest.raises(ValueError,match='approval'):
        submit_array(tmp_path,'missing',{},site_lock='missing',owner_token='fixture')


def test_submission_records_resources_afterok_and_replacement(tmp_path,monkeypatch):
    from geobench.slurm import submit_array
    profile={'artifact_root':'/fixture/root','cpu_count':8,'memory':'64G','walltime':'3-00:00:00',
             'account':'uit','partition':'normal','qos':'mig3-71','gres':'gpu:3g.71gb:1'}
    site=publish_records(tmp_path,'site','V03',{'evidence.json':json.dumps({'status':'verified','profile':profile,
        'max_walltime':profile['walltime'],'signal_grace_seconds':300,'worst_epoch_seconds':200,'checkpoint_seconds':20,
        'array_max_tasks':100,'array_concurrency':2}).encode()},{},owner_token='fixture')
    array=freeze_array(tmp_path,'fixture',[{'id':'run','argv':['train','--run','E0/42'],'dependencies':{},'outputs':['run/lock.json']}],owner_token='fixture')
    calls=[]
    def fake(argv,**kwargs):
        calls.append(argv);return type('Result',(),{'stdout':str(100+len(calls))+';cluster\n'})()
    monkeypatch.setattr('geobench.slurm.subprocess.run',fake)
    monkeypatch.setattr('geobench.preflight.assess_readiness',lambda root,evidence:{'roles':{'classifier':True,'diffusion':True,'dino':True}})
    kwargs={'site_lock':site.relative_to(tmp_path).as_posix(),'owner_token':'fixture','scientific_approved':True}
    first=submit_array(tmp_path,array.relative_to(tmp_path).as_posix(),profile,**kwargs)
    second=submit_array(tmp_path,array.relative_to(tmp_path).as_posix(),profile,
                        afterok=[first.relative_to(tmp_path).as_posix()],replacement=first.relative_to(tmp_path).as_posix(),**kwargs)
    assert '--dependency=afterok:101' in calls[-1]
    assert '--time=3-00:00:00' in calls[-1] and '--gres=gpu:3g.71gb:1' in calls[-1]
    record=json.loads((second.parent/'submission.json').read_text())
    assert record['replacement']==first.relative_to(tmp_path).as_posix() and record['job_id']=='102'
