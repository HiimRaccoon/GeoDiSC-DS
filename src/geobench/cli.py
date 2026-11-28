"""Commands implemented in the P0 phase."""
import argparse
import json
import os
from pathlib import Path
from .artifacts import verify_lock


def main(argv=None):
    parser = argparse.ArgumentParser(prog='geobench')
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('lock', type=Path)
    verify.add_argument('--root', type=Path, help='trusted benchmark root for cross-stage dependencies')
    prepare = commands.add_parser('prepare-data')
    prepare.add_argument('--dataset-root', required=True, type=Path)
    prepare.add_argument('--root', required=True, type=Path)
    for name in ('benchmark', 'methods', 'deployment'):
        prepare.add_argument('--' + name + '-config', type=Path)
    preflight = commands.add_parser('preflight')
    preflight.add_argument('--root', required=True, type=Path)
    preflight.add_argument('--profile', type=Path)
    e1 = commands.add_parser('generate-e1')
    e1.add_argument('--root', required=True, type=Path)
    e2 = commands.add_parser('generate-e2')
    e2.add_argument('--root', required=True, type=Path)
    e2.add_argument('--descriptor', required=True, type=Path)
    initialize = commands.add_parser('initialize')
    initialize.add_argument('--root', required=True, type=Path)
    train = commands.add_parser('train')
    train.add_argument('--root', required=True, type=Path)
    train.add_argument('--run', required=True)
    tune = commands.add_parser('prepare-tuning')
    tune.add_argument('--root', required=True, type=Path)
    tune_run = commands.add_parser('train-tuning')
    tune_run.add_argument('--root', required=True, type=Path)
    tune_run.add_argument('--strength', required=True, type=float, choices=(.25,.35,.5))
    tune_run.add_argument('--seed', required=True, type=int, choices=(42,123,2026))
    select = commands.add_parser('select-tuning')
    select.add_argument('--root', required=True, type=Path)
    e3 = commands.add_parser('generate-e3')
    e3.add_argument('--root',required=True,type=Path)
    e3.add_argument('--descriptor',required=True,type=Path)
    e3.add_argument('--variant',required=True,choices=('A1','A2','A3'))
    e3.add_argument('--reference',action='store_true')
    mining = commands.add_parser('mine-confusers')
    mining.add_argument('--root',required=True,type=Path)
    mining.add_argument('--model-lock',required=True)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        print(json.dumps(verify_lock(args.lock, boundary=args.root), sort_keys=True))
    elif args.command == 'prepare-data':
        from .config import verified_configuration
        configuration = verified_configuration(args.benchmark_config, args.methods_config, args.deployment_config)
        from .data import prepare_data
        print(prepare_data(args.dataset_root, args.root, owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN'), configuration=configuration))
    elif args.command == 'preflight':
        from .preflight import inspect_deployment
        token = os.environ.get('GEOBENCH_OWNER_TOKEN')
        if not token:
            raise ValueError('GEOBENCH_OWNER_TOKEN is required')
        profile = None if args.profile is None else json.loads(args.profile.read_text(encoding='utf-8'))
        print(inspect_deployment(args.root, {'profile': profile}, owner_token=token))
    elif args.command == 'generate-e1':
        from .augmentation.traditional import generate_e1
        print(generate_e1(args.root, owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'generate-e2':
        from .augmentation.diffusion import generate_pool, build_hybrid_pool
        token = os.environ.get('GEOBENCH_OWNER_TOKEN')
        if not token:
            raise ValueError('owner token required')
        from .config import _unique_pairs
        descriptor = json.loads(args.descriptor.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        generate_pool(args.root, descriptor, owner_token=token)
        print(build_hybrid_pool(args.root, descriptor['strength'], owner_token=token))
    elif args.command == 'initialize':
        from .classifier import create_initializations
        print(create_initializations(args.root, owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'train':
        from .training import train_run
        print(train_run(args.root,args.run,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'prepare-tuning':
        from .tuning import prepare_tuning
        print(prepare_tuning(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'train-tuning':
        from .training import train_tuning
        print(train_tuning(args.root,args.strength,args.seed,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'select-tuning':
        from .artifacts import sha256_file
        from .selection import publish_strength_selection
        from .tuning import STRENGTHS, SEEDS
        references=[]
        for strength in STRENGTHS:
            for seed in SEEDS:
                path=f'tuning/E2/strength_{strength}/seed_{seed}/checkpoint_lock.json'
                references.append({'strength':strength,'seed':seed,'path':path,'sha256':sha256_file(args.root/path)})
        print(publish_strength_selection(args.root,references,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'generate-e3':
        from .augmentation.e3 import generate_e3, build_e3_pool
        from .config import _unique_pairs
        descriptor=json.loads(args.descriptor.read_text(encoding='utf-8'),object_pairs_hook=_unique_pairs)
        print(generate_e3(args.root,descriptor,variant=args.variant,reference=args.reference,
                          owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
        if not args.reference:
            print(build_e3_pool(args.root,args.variant,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'mine-confusers':
        from .augmentation.mining import extract_and_mine
        print(extract_and_mine(args.root,model_lock=args.model_lock,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    return 0
