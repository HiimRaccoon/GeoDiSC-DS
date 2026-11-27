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
