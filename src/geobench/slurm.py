"""Frozen Slurm arrays, verified execution and explicit submission records."""
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
import math
from .artifacts import _inside, publish_records, sha256_file, verify_lock
from .bindings import encoded
from .preflight import validate_profile

COMMANDS={'prepare-data','generate-e1','generate-e2','generate-e3','mine-confusers','initialize','train','train-tuning',
          'prepare-tuning','select-tuning','prepare-run','freeze-selected-pool','bind-selected','evaluate','report','preflight',
          'probe-gpu','freeze-runtime','freeze-specs','plan-fresh-e1e2'}


def freeze_array(root,name,tasks,*,owner_token):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',name) or not isinstance(tasks,list) or not tasks:
        raise ValueError('array name and tasks required')
    deps={}; ids=set()
    for task in tasks:
        if (not isinstance(task,dict) or set(task)!={'id','argv','dependencies','outputs'} or not isinstance(task['id'],str)
                or not task['id'] or task['id'] in ids or not isinstance(task['argv'],list) or not task['argv']
                or any(not isinstance(a,str) or not a for a in task['argv']) or task['argv'][0] not in COMMANDS
                or '--root' in task['argv'] or not isinstance(task['dependencies'],dict) or not isinstance(task['outputs'],list) or not task['outputs']):
            raise ValueError('invalid or duplicate array task')
        ids.add(task['id'])
        for path,digest in task['dependencies'].items():
            lock=_inside(root,path);verify_lock(lock,boundary=root)
            if sha256_file(lock)!=digest: raise ValueError('task dependency mismatch')
            deps[path]=digest
        for path in task['outputs']: _inside(root,path)
    return publish_records(root,f'deployment/jobs/arrays/{name}',f'job-array:{name}',{'array.json':encoded(tasks)},deps,owner_token=owner_token)


def execute_job(root,manifest,index):
    import sys
    root=Path(root).resolve();path=_inside(root,manifest);lock=verify_lock(path,boundary=root)
    if not lock['identity'].startswith('job-array:') or 'array.json' not in lock['files']:
        raise ValueError('frozen array required')
    tasks=json.loads((path.parent/'array.json').read_text())
    if type(index) is not int or not 0<=index<len(tasks): raise ValueError('array index out of range')
    task=tasks[index]
    for relative,digest in task['dependencies'].items():
        dep=_inside(root,relative);verify_lock(dep,boundary=root)
        if sha256_file(dep)!=digest: raise ValueError('execution dependency mismatch')
    if task['argv'][0] not in COMMANDS or '--root' in task['argv']: raise ValueError('unsupported array command')
    argv=task['argv'];seed=314159
    if argv[0] in ('train','evaluate'):
        seed=int(argv[argv.index('--run')+1].rsplit('/',1)[1])
    elif argv[0]=='train-tuning': seed=int(argv[argv.index('--seed')+1])
    env=dict(os.environ,PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    subprocess.run([sys.executable,'-m','geobench',argv[0],'--root',str(root),*argv[1:]],env=env,check=True)
    for output in task['outputs']: verify_lock(_inside(root,output),boundary=root)
    return 0


def submit_array(root,manifest,profile,*,site_lock,owner_token,kind='gpu',afterok=(),replacement=None,scientific_approved=False):
    """No jobs are submitted by preparation or tests; the operator calls this explicitly."""
    if scientific_approved is not True: raise ValueError('separate scientific execution approval required')
    if kind not in ('cpu','gpu'): raise ValueError('unknown resource role')
    profile=validate_profile(profile);root=Path(root).resolve()
    site=_inside(root,site_lock);site_record=verify_lock(site,boundary=root)
    if site_record['identity']!='V03' or 'evidence.json' not in site_record['files']: raise ValueError('verified site evidence required')
    evidence=json.loads((site.parent/'evidence.json').read_text())
    if evidence.get('status')!='verified' or evidence.get('profile')!=profile or evidence.get('max_walltime')!=profile['walltime']:
        raise ValueError('resource profile or maximum walltime not verified')
    grace=evidence.get('signal_grace_seconds');epoch=evidence.get('worst_epoch_seconds');checkpoint=evidence.get('checkpoint_seconds')
    if (type(grace) is not int or not 1<=grace<=65535 or type(epoch) not in (int,float) or type(checkpoint) not in (int,float)
            or not math.isfinite(epoch) or not math.isfinite(checkpoint) or epoch<=0 or checkpoint<=0 or grace<epoch+checkpoint):raise ValueError('measured epoch/checkpoint signal grace required')
    array=_inside(root,manifest);verify_lock(array,boundary=root)
    tasks=json.loads((array.parent/'array.json').read_text())
    if not tasks: raise ValueError('empty array')
    from .preflight import assess_readiness
    readiness=evidence.get('readiness')
    roles=assess_readiness(root,readiness if isinstance(readiness,dict) else {})['roles']
    for task in tasks:
        command=task['argv'][0]
        role=('classifier' if command in ('train','train-tuning','evaluate','initialize') else
              'diffusion' if command in ('generate-e2','generate-e3') else 'dino' if command=='mine-confusers' else None)
        if role is not None and not roles[role]:raise ValueError(f'{role} deployment verification incomplete')
    maximum=evidence.get('array_max_tasks');concurrency=evidence.get('array_concurrency')
    if (type(maximum) is not int or type(concurrency) is not int or maximum<1 or concurrency<1
            or len(tasks)>maximum or concurrency>maximum):raise ValueError('verified array limits required')
    predecessors=[]
    for path in afterok:
        previous=_inside(root,path);verify_lock(previous,boundary=root)
        record=json.loads((previous.parent/'submission.json').read_text())
        if not re.fullmatch(r'[0-9]+',record['job_id']): raise ValueError('invalid predecessor JobID')
        predecessors.append(record['job_id'])
    if replacement is not None: verify_lock(_inside(root,replacement),boundary=root)
    logs=root/'logs/slurm';logs.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parents[2]/f'scripts/slurm/{kind}.sbatch'
    argv=['sbatch','--parsable','--ntasks=1',f'--cpus-per-task={profile["cpu_count"]}',f'--mem={profile["memory"]}',
          f'--time={profile["walltime"]}',f'--account={profile["account"]}',f'--partition={profile["partition"]}',
          f'--signal=USR1@{grace}',
          f'--array=0-{len(tasks)-1}%{concurrency}',f'--output={logs}/%A_%a.out',f'--error={logs}/%A_%a.err']
    if kind=='gpu': argv += [f'--qos={profile["qos"]}',f'--gres={profile["gres"]}']
    if predecessors: argv.append('--dependency=afterok:'+':'.join(predecessors))
    argv += [str(script),str(root),manifest]
    attempt=uuid.uuid4().hex;relative=f'deployment/jobs/submissions/{attempt}'
    deps={manifest:sha256_file(array),site_lock:sha256_file(site)}
    for previous in [*afterok,*([replacement] if replacement is not None else [])]:
        deps[previous]=sha256_file(_inside(root,previous))
    intent={'argv':argv,'replacement':replacement,'predecessors':list(afterok),'state':'submission_pending'}
    publish_records(root,relative+'/intent','submission-intent',{'intent.json':encoded(intent)},deps,owner_token=owner_token)
    result=subprocess.run(argv,text=True,capture_output=True,check=True)
    job_id=result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'[0-9]+',job_id): raise RuntimeError('unknown submission result; inspect pending intent before retrying')
    record={**intent,'state':'submitted','job_id':job_id,'array_sha256':sha256_file(array)}
    return publish_records(root,relative,'submission',{'submission.json':encoded(record)},deps,owner_token=owner_token)
