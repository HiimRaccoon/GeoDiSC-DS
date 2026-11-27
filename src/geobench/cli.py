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
    return 0
