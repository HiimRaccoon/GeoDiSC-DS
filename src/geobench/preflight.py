"""Validation of explicit deployment inputs; no implicit site discovery."""
import re

PAIRS = {'mig3-71': 'gpu:3g.71gb:1', 'mig7-141': 'gpu:7g.141gb:1'}
FIELDS = {'artifact_root', 'cpu_count', 'memory', 'walltime', 'account', 'partition', 'qos', 'gres'}


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != FIELDS:
        raise ValueError('deployment profile requires exactly the declared fields')
    if any(not isinstance(profile[k], str) or not profile[k].strip() for k in FIELDS - {'cpu_count'}):
        raise ValueError('deployment fields must be explicit')
    if type(profile['cpu_count']) is not int or profile['cpu_count'] < 1:
        raise ValueError('CPU count must be positive')
    if not profile['artifact_root'].startswith('/') or not re.fullmatch(r'[1-9][0-9]*[KMGTP]', profile['memory']):
        raise ValueError('invalid artifact root or host memory')
    if (not re.fullmatch(r'[0-9]+-[0-2][0-9]:[0-5][0-9]:[0-5][0-9]', profile['walltime'])
            or int(profile['walltime'].split('-')[1][:2]) > 23):
        raise ValueError('walltime must be an explicit Slurm duration')
    if PAIRS.get(profile['qos']) != profile['gres']:
        raise ValueError('unsupported QoS/GRES combination')
    return dict(profile)


def readiness_by_role(evidence):
    required = {'site', 'storage', 'base_models', 'dino'}
    if not isinstance(evidence, dict) or set(evidence) != required or any(type(value) is not bool for value in evidence.values()):
        raise ValueError('evidence status must be explicit')
    common = evidence['site'] and evidence['storage']
    return {'classifier': common and evidence['base_models'],
            'diffusion': common and evidence['base_models'],
            'dino': common and evidence['dino']}


def verify_snapshot(snapshot, descriptor, *, cache_root=None):
    """Verify a materialized snapshot without downloading or selecting revisions."""
    from pathlib import Path
    from .artifacts import sha256_file
    root = Path(snapshot).resolve()
    cache = Path(cache_root).resolve() if cache_root is not None else root
    if not root.is_relative_to(cache):
        raise ValueError('snapshot is outside the trusted cache')
    if not isinstance(descriptor, dict) or set(descriptor) != {'repository', 'revision', 'files'}:
        raise ValueError('invalid snapshot descriptor')
    if not isinstance(descriptor['repository'], str) or not descriptor['repository'] or not re.fullmatch(r'[0-9a-f]{40}', descriptor['revision']):
        raise ValueError('immutable model revision required')
    if not isinstance(descriptor['files'], dict) or not descriptor['files']:
        raise ValueError('snapshot file hashes are required')
    if not root.is_dir():
        raise ValueError('snapshot directory missing')
    paths = list(root.rglob('*'))
    if any(path.is_symlink() and (path.is_dir() or not path.resolve().is_relative_to(cache)) for path in paths):
        raise ValueError('snapshot symlink escapes the trusted cache or points to a directory')
    if {path.relative_to(root).as_posix() for path in paths if path.is_file()} != set(descriptor['files']):
        raise ValueError('snapshot inventory mismatch')
    for name, expected in descriptor['files'].items():
        try:
            if (not isinstance(name, str) or not name or '\\' in name
                    or any(part in {'', '.', '..'} for part in name.split('/'))
                    or Path(name).is_absolute()):
                raise ValueError('invalid snapshot path')
            target = (root / name).resolve(strict=True)
            if not target.is_relative_to(cache) or sha256_file(target) != expected:
                raise ValueError('snapshot file hash mismatch')
        except OSError as exc:
            raise ValueError('snapshot file unavailable') from exc
    return dict(descriptor)


def inspect_deployment(root, profile, *, owner_token):
    """Publish an inspection record, keeping unmeasured checks pending."""
    import json
    import platform
    from pathlib import Path
    from .artifacts import claim_writer, staging_path, publish_lock, complete_temporary
    from .config import verified_configuration
    root = Path(root).resolve()
    if not isinstance(profile, dict) or not {'profile'} <= set(profile) or set(profile) - {'profile', 'evidence'}:
        raise ValueError('invalid inspection descriptor')
    resolved = None if profile['profile'] is None else validate_profile(profile['profile'])
    config = verified_configuration()
    evidence = profile.get('evidence', {})
    assessment = assess_readiness(root, evidence)
    result = {'status': {'site': assessment['checks']['V01'], 'dino': assessment['checks']['V06']},
              'verification': assessment['checks'], 'roles': assessment['roles'],
              'python': platform.python_version(), 'configuration_sha256': config['sha256'],
              'profile_resolved': resolved is not None,
              'environment': inspect_environment(['torch', 'torchvision', 'diffusers', 'transformers', 'numpy', 'Pillow'])}
    root.mkdir(parents=True,exist_ok=True)
    result['storage']=inspect_storage(root)
    import hashlib
    version=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    target = root / 'deployment' / 'preflight'
    if (target/'lock.json').exists():target=target/'inspections'/version
    token = claim_writer(target, 'deployment-preflight', token=owner_token, resume=(target/'.writer').exists())
    try:
        path = staging_path(target, token) / 'preflight.json'
        complete_temporary(path, (json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n').encode())
        dependencies = {reference['path']: reference['sha256'] for reference in evidence.values()}
        return publish_lock(target, {'identity': 'deployment-preflight', 'files': ['preflight.json'], 'dependencies': dependencies, 'artifact_root': root, 'claim': token})
    finally:
        token.close()


def inspect_site(root,*,owner_token):
    """Collect read-only SSH observations using a separately verified private site."""
    import hashlib,json,shlex,subprocess
    from pathlib import Path
    from .data import _private_site
    from .artifacts import publish_records,sha256_file
    if not isinstance(owner_token,str) or not owner_token:raise ValueError('owner token required')
    root=Path(root).resolve();site,path=_private_site(root)
    target=site['ssh_target'];workspace=site['workspace']
    match=re.fullmatch(r'([A-Za-z0-9_.@-]+)(?::([0-9]+))?',target)
    if match is None or target.startswith('-'):
        raise ValueError('invalid private SSH target')
    host,port=match.groups()
    if port is not None and not 1<=int(port)<=65535:raise ValueError('invalid private SSH port')
    commands={'login_identity':'id -un','login_host':'hostname -f','workspace_storage':'df -Pk '+shlex.quote(workspace),
              'workspace_access':'test -d '+shlex.quote(workspace)+' && test -r '+shlex.quote(workspace)}
    observations={}
    for name,command in commands.items():
        argv=['ssh','-o','BatchMode=yes',*(['-p',port] if port is not None else []),host,command]
        result=subprocess.run(argv,text=True,capture_output=True,timeout=60,check=False)
        observations[name]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    record={'status':'observed','observations':observations,'site_sha256':sha256_file(path)}
    content=(json.dumps(record,sort_keys=True,separators=(',', ':'))+'\n').encode()
    version=hashlib.sha256(content).hexdigest()
    return publish_records(root,f'deployment/inspections/site/{version}','site-inspection',{'inspection.json':content},
        {path.relative_to(root).as_posix():sha256_file(path)},owner_token=owner_token)


def probe_gpu(root, role):
    """Invoke a shared provider only from a scheduler allocation."""
    import importlib
    import os
    if role not in {'classifier', 'diffusion', 'dino'}:
        raise ValueError('unknown probe role')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('GPU probes require a Slurm allocation')
    module = {'classifier': 'geobench.classifier', 'diffusion': 'geobench.augmentation.diffusion', 'dino': 'geobench.augmentation.mining'}[role]
    try:
        provider = importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f'{role} shared provider is not available') from exc
    if not hasattr(provider, 'probe_gpu'):
        raise RuntimeError(f'{role} shared probe provider is not available')
    return provider.probe_gpu(root)


def inspect_environment(packages):
    """Collect local interpreter and installed package metadata without GPU use."""
    import importlib.metadata
    import importlib.util
    import platform
    import sys
    if not isinstance(packages, list) or any(not isinstance(name, str) or not name for name in packages):
        raise ValueError('explicit package names required')
    found = {}
    for name in packages:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            spec = importlib.util.find_spec(name)
            origin = spec.origin if spec else None
        except (ValueError, ImportError, ModuleNotFoundError):
            origin = None
        found[name] = {'status': 'observed' if version else 'unavailable', 'version': version, 'origin': origin}
    return {'python': {'version': platform.python_version(), 'executable': sys.executable}, 'packages': found}


def inspect_storage(path):
    """Observe available space; quota and compute-node visibility remain pending."""
    import os
    import shutil
    from pathlib import Path
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError('storage inspection requires an existing directory')
    usage = shutil.disk_usage(path)
    try:
        stat = os.statvfs(path)
        inodes = stat.f_favail
    except (AttributeError, OSError):
        inodes = None
    return {'status': 'observed', 'free_bytes': usage.free, 'free_inodes': inodes,
            'quota': 'pending', 'compute_visibility': 'pending', 'atomic_semantics': 'pending'}


def verify_model_snapshot(snapshot, role, descriptor, *, cache_root=None):
    if role not in {'classifier', 'diffusion', 'dino'}:
        raise ValueError('unknown model role')
    if role == 'dino' and (descriptor.get('repository') != 'facebook/dinov3-vitb16-pretrain-lvd1689m'
                           or descriptor.get('revision') != '5931719e67bbdb9737e363e781fb0c67687896bc'):
        raise ValueError('DINO snapshot does not match the confirmed model and revision')
    return verify_snapshot(snapshot, descriptor, cache_root=cache_root)


ROLE_REQUIREMENTS = {
    'classifier': {'V01','V02','V03','V04','V05','V09','V13'},
    'diffusion': {'V01','V02','V03','V04','V05','V07_base','V08','V13'},
    'dino': {'V01','V02','V03','V04','V06','V10','V13'},
}

CONFORMANCE_CHECKS = {
    'V01':{'private_site_matches','dataset_path_verified','separate_code_checkout'},
    'V02':{'quota_bytes_inodes_verified','compute_visibility','atomic_publication','exclusive_ownership'},
    'V03':{'account_partition_qos_verified','maximum_walltime_verified','array_limits_verified','cpu_memory_verified'},
    'V04':{'inventory_49997_124','decoder_locked','file_pixel_hashes_verified','duplicates_split_supports_verified'},
    'V05':{'environment_exports_locked','cuda_driver_compatible','determinism_verified','base_model_snapshots_verified'},
    'V06':{'pinned_dinov3_weights_verified','official_processor_verified','processor_shape_verified'},
    'V07_base':{'all_base_prompts_audited','special_tokens_included','no_overflow','no_truncation'},
    'V08':{'shared_sampler_used','native_geometry_verified','strength_grid_verified','three_branches_verified','profile_fits'},
    'V09':{'shared_trainer_used','batch128_fp32_backward_update','loader_verified','a4_path_verified','pristine_state_reloaded','profile_fits'},
    'V10':{'shared_extractor_used','batch32_and_partial_verified','official_processor_used','fp32_frozen_cls','profile_fits'},
    'V13':{'compute_smoke_passed','signal_forwarding_verified','requeue_policy_verified','epoch_resume_verified','checkpoint_grace_measured'},
}


def assess_readiness(root, evidence):
    """Verify explicit locked evidence; absent observations remain pending."""
    from pathlib import Path
    import json
    from .artifacts import _inside, sha256_file, verify_lock
    root = Path(root).resolve()
    known = set().union(*ROLE_REQUIREMENTS.values())
    if not isinstance(evidence, dict) or set(evidence) - known:
        raise ValueError('unknown evidence roles')
    checks = {name: 'pending' for name in sorted(known)}
    for name, reference in evidence.items():
        if not isinstance(reference, dict) or set(reference) != {'path','sha256'}:
            raise ValueError('invalid evidence reference')
        path = _inside(root, reference['path'])
        try:
            if sha256_file(path) != reference['sha256']:
                raise ValueError('evidence hash mismatch')
            lock = verify_lock(path, boundary=root)
        except OSError as exc:
            raise ValueError('evidence missing') from exc
        if lock['identity'] != name:
            raise ValueError('evidence identity mismatch')
        if 'evidence.json' not in lock['files']:
            raise ValueError('evidence record missing')
        try:
            record = json.loads((path.parent / 'evidence.json').read_text(encoding='utf-8'))
            if (not isinstance(record, dict) or record.get('status') != 'verified'
                    or record.get('verification') != name or not isinstance(record.get('evidence_files'), list)
                    or not record['evidence_files']
                    or any(not isinstance(item, str) or item == 'evidence.json' or item not in lock['files'] for item in record['evidence_files'])):
                raise ValueError('verification record incomplete')
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('invalid verification record') from exc
        checks[name] = 'evidence_present'
        # Operator-reviewed, scoped observations must accompany the raw evidence.
        conformance=record.get('conformance')
        if (isinstance(conformance,dict) and set(conformance)==CONFORMANCE_CHECKS[name]
                and all(value is True for value in conformance.values())):
            if name in {'V08','V09','V10'}:
                role={'V08':'diffusion','V09':'classifier','V10':'dino'}[name]
                probe_path=record.get('probe_lock')
                if not isinstance(probe_path,str) or probe_path not in lock['dependencies']:
                    raise ValueError('shared probe dependency required')
                probe=verify_lock(_inside(root,probe_path),boundary=root)
                if probe['identity']!=f'probe:{role}' or 'probe.json' not in probe['files']:raise ValueError('probe role mismatch')
                observation=json.loads((_inside(root,probe_path).parent/'probe.json').read_text(encoding='utf-8'))
                if (observation.get('role')!=role or observation.get('state_discarded') is not True
                        or not observation.get('job_id') or observation.get('max_memory_bytes',0)<=0 or observation.get('seconds',0)<=0):
                    raise ValueError('invalid measured probe evidence')
            checks[name] = 'verified'
    return {'checks': checks, 'roles': {role: all(checks[key] == 'verified' for key in requirements)
                                      for role, requirements in ROLE_REQUIREMENTS.items()}}


def audit_prompts(tokenizer, prompts):
    """Return untruncated token evidence and role-scoped overflow reports."""
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError('rendered prompts are required')
    records, overflow = [], set()
    for role, texts in prompts.items():
        if not isinstance(role, str) or not role or not isinstance(texts, list) or not texts:
            raise ValueError('invalid prompt role')
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('prompt must be text')
            tokens = tokenizer(text, truncation=False)['input_ids']
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                raise ValueError('invalid tokenizer output')
            records.append({'role': role, 'text': text, 'token_ids': tokens, 'count': len(tokens)})
            if len(tokens) > 77:
                overflow.add(role)
    return {'records': records, 'overflow_roles': sorted(overflow)}
