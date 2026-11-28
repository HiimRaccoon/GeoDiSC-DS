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
    probe=commands.add_parser('probe-gpu');probe.add_argument('--root',required=True,type=Path)
    probe.add_argument('--role',required=True,choices=('classifier','diffusion','dino'))
    specs=commands.add_parser('freeze-specs');specs.add_argument('--root',required=True,type=Path)
    fresh=commands.add_parser('plan-fresh-e1e2');fresh.add_argument('--root',required=True,type=Path);fresh.add_argument('--reason',required=True)
    materialize=commands.add_parser('stage-model');materialize.add_argument('--root',required=True,type=Path)
    materialize.add_argument('--role',required=True,choices=('classifier','diffusion','dino'))
    materialize.add_argument('--snapshot',required=True,type=Path);materialize.add_argument('--descriptor',required=True,type=Path)
    materialize.add_argument('--cache-root',type=Path)
    runtime=commands.add_parser('freeze-runtime');runtime.add_argument('--root',required=True,type=Path)
    site=commands.add_parser('inspect-site');site.add_argument('--root',required=True,type=Path)
    probe_input=commands.add_parser('prepare-probe');probe_input.add_argument('--root',required=True,type=Path)
    probe_input.add_argument('--role',required=True,choices=('classifier','diffusion','dino'));probe_input.add_argument('--model-lock',required=True)
    for name in ('prepare-run','evaluate'):
        command=commands.add_parser(name);command.add_argument('--root',required=True,type=Path);command.add_argument('--run',required=True)
    for name in ('freeze-selected-pool','bind-selected','report'):
        command=commands.add_parser(name);command.add_argument('--root',required=True,type=Path)
    array=commands.add_parser('freeze-array');array.add_argument('--root',required=True,type=Path)
    array.add_argument('--name',required=True);array.add_argument('--tasks',required=True,type=Path)
    execute=commands.add_parser('execute-job');execute.add_argument('--root',required=True,type=Path)
    execute.add_argument('--manifest',required=True);execute.add_argument('--index',required=True,type=int)
    submit=commands.add_parser('submit-array');submit.add_argument('--root',required=True,type=Path)
    submit.add_argument('--manifest',required=True);submit.add_argument('--profile',required=True,type=Path)
    submit.add_argument('--site-lock',required=True);submit.add_argument('--kind',choices=('cpu','gpu'),default='gpu')
    submit.add_argument('--afterok',action='append',default=[]);submit.add_argument('--replacement')
    submit.add_argument('--scientific-approved',action='store_true')
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
    elif args.command == 'prepare-run':
        from .bindings import prepare_run
        print(prepare_run(args.root,args.run,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'freeze-specs':
        from .bindings import freeze_specs
        for path in freeze_specs(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')): print(path)
    elif args.command == 'plan-fresh-e1e2':
        from .bindings import plan_fresh_e1e2
        print(plan_fresh_e1e2(args.root,args.reason,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'probe-gpu':
        from .preflight import probe_gpu
        print(probe_gpu(args.root,args.role))
    elif args.command == 'prepare-probe':
        from .deployment import prepare_probe
        print(prepare_probe(args.root,args.role,args.model_lock,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'stage-model':
        from .deployment import stage_model
        from .config import _unique_pairs
        descriptor=json.loads(args.descriptor.read_text(encoding='utf-8'),object_pairs_hook=_unique_pairs)
        print(stage_model(args.root,args.role,args.snapshot,descriptor,cache_root=args.cache_root,
                          owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'freeze-runtime':
        from .deployment import freeze_runtime
        print(freeze_runtime(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'inspect-site':
        from .preflight import inspect_site
        print(inspect_site(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'freeze-selected-pool':
        from .bindings import freeze_selected_pool
        print(freeze_selected_pool(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'bind-selected':
        from .bindings import bind_selected_runs
        print(bind_selected_runs(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'evaluate':
        from .evaluation import evaluate_run
        print(evaluate_run(args.root,args.run,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'report':
        from .reporting import report
        print(report(args.root,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'freeze-array':
        from .slurm import freeze_array
        print(freeze_array(args.root,args.name,json.loads(args.tasks.read_text(encoding='utf-8')),
                           owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    elif args.command == 'execute-job':
        from .slurm import execute_job
        return execute_job(args.root,args.manifest,args.index)
    elif args.command == 'submit-array':
        from .slurm import submit_array
        print(submit_array(args.root,args.manifest,json.loads(args.profile.read_text(encoding='utf-8')),
            site_lock=args.site_lock,kind=args.kind,afterok=args.afterok,replacement=args.replacement,
            scientific_approved=args.scientific_approved,owner_token=os.environ.get('GEOBENCH_OWNER_TOKEN')))
    return 0
