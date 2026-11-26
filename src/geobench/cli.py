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
    args = parser.parse_args(argv)
    if args.command == 'verify':
        print(json.dumps(verify_lock(args.lock, boundary=args.root), sort_keys=True))
    elif args.command == 'prepare-data':
        from .data import prepare_data
        print(prepare_data(args.dataset_root, args.root, owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    return 0
