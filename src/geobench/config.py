"""Confirmed research settings and separately scoped deployment inputs."""
import json
from pathlib import Path

RESEARCH = {
    'classifier': {
        'model': 'EfficientNet-B0', 'weights': 'IMAGENET1K_V1', 'full_backbone': True,
        'head': [1280, 124], 'dropout': 0.2, 'batch_size': 128, 'input_shape': [3, 384, 896],
        'epochs': 20, 'learning_rate': 0.0003, 'weight_decay': 0.0001, 'precision': 'FP32', 'amp': False,
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'amsgrad': False,
                      'maximize': False, 'foreach': False, 'fused': False, 'capturable': False,
                      'differentiable': False, 'single_parameter_group': True},
        'scheduler': {'warmup_epochs': 1, 'warmup_start_factor': 0.1, 'schedule': 'cosine', 'min_lr': 1e-6,
                      'step_unit': 'optimizer_step', 'apply_before_update': True},
        'loss': {'name': 'CrossEntropyLoss', 'reduction': 'mean', 'label_smoothing': 0,
                 'class_weights': False, 'focal': False},
        'runtime': {'device': 'single_cuda_gpu', 'tf32': False, 'grad_scaler': False, 'layout': 'NCHW',
                    'compile': False, 'gradient_accumulation_steps': 1, 'gradient_clipping': False,
                    'zero_grad_set_to_none': True, 'num_workers': 4, 'pin_memory': True,
                    'persistent_workers': False, 'prefetch_factor': 2, 'drop_last': False,
                    'weighted_sampler': False, 'train_shuffle': True, 'eval_shuffle': False,
                    'train_replacement': False, 'deterministic_algorithms': True,
                    'cudnn_benchmark': False, 'cudnn_deterministic': True, 'cublas_workspace_config': ':4096:8',
                    'default_mixup': False, 'cutmix': False},
        'preprocessing': {'decoder': 'Pillow_EXIF_RGB', 'resize': 'torchvision_tensor_bilinear_antialias',
                          'canvas_wh': [896, 384], 'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225], 'padding': 'center_imagenet_mean'},
    },
    'dataset': {'images': 49997, 'classes': 124, 'split_policy': 'geolocation_common_split_v2_exact_duplicates',
                'duplicate_policy': 'file_or_decoded_rgb_connected_components', 'head_min': 100, 'medium_min': 20},
    'seeds': {'training': [42, 123, 2026], 'split': 42, 'augmentation': 314159},
    'evaluation': {'ranking': 'descending_logit_ascending_class_id', 'support': 'ground_truth_positive',
                   'checkpoint': 'overall_validation_macro_f1_earliest_tie',
                   'aggregation': 'three_seed_sample_std', 'display': 'percent_2_decimals'},
}

METHODS = {
    'traditional': {'eligible_train_range': [20, 99], 'target': 100, 'transform_count': [1, 2],
                    'brightness': [0.9, 1.1], 'contrast': [0.9, 1.1], 'blur_sigma': [0.1, 0.6],
                    'blur_kernel': 5, 'noise_sigma_255': [1, 3], 'rotation_degrees': [-3, 3],
                    'rotation_interpolation': 'bilinear', 'rotation_padding': 'reflection'},
    'diffusion': {'backbone': 'Stable Diffusion v1.5', 'mode': 'img2img', 'solver': 'DPM-Solver++',
                  'steps': 50, 'guidance_scale': 2.0, 'batch_size': 1, 'precision': 'FP32', 'amp': False,
                  'strength_candidates': [0.25, 0.35, 0.5], 'resolution': 'native_reflection_pad_8',
                  'textual_inversion': False, 'lora': False, 'max_attempts': 5, 'acceptance': 'first_valid',
                  'negative_prompt': '', 'tail_multiplier': 5, 'tail_max': 100, 'singleton_final_synthetic': False},
    'cads': {'tau1': 0.5, 'tau2': 0.9, 'noise_scale': 0.1, 'rescale': 1.0},
    'ccfg': {'temperature': 0.8, 'cads_interpolation': 0.8},
    'mining': {'references_per_class': 20, 'model': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
               'revision': '5931719e67bbdb9737e363e781fb0c67687896bc', 'batch_size': 32, 'precision': 'FP32'},
    'mixup': {'selected_batch_fraction': 0.5, 'lambda_values': [i / 20 for i in range(21)],
              'partner': 'same_class_real_with_replacement', 'before_normalization': True},
    'variants': {'E0': [], 'E1E2': ['traditional', 'diffusion'], 'A1': ['traditional', 'diffusion', 'cads'],
                 'A2': ['traditional', 'diffusion', 'cads', 'mining', 'ccfg'],
                 'A3': ['traditional', 'diffusion', 'cads', 'mining', 'ccfg', 'geo'],
                 'A4': ['traditional', 'diffusion', 'cads', 'mining', 'ccfg', 'geo', 'mixup']},
}

METHODS['traditional'].update({
    'operator_order': ['brightness', 'contrast', 'blur', 'noise', 'rotation'],
    'selection': 'uniform_without_replacement', 'transform_count_distribution': 'uniform',
    'contrast_mean': 'scalar_rgb', 'clip_after_each': True, 'quantization': 'floor_255x_plus_0.5',
    'source_selection': 'deterministic_shuffle_cycle', 'seed_namespace': 'geo-e1e2-v2',
})
METHODS['diffusion'].update({
    'scheduler': {'class': 'DPMSolverMultistepScheduler', 'algorithm_type': 'dpmsolver++',
                  'solver_order': 2, 'solver_type': 'midpoint', 'timestep_spacing': 'linspace',
                  'lower_order_final': True, 'thresholding': False, 'karras_sigmas': False},
    'prompt': 'A realistic street-level road photograph in {country_name}, look like picture in Google Street View.',
    'attention': 'standard', 'xformers': False, 'vae_slicing': False, 'vae_tiling': False,
    'vae_posterior': 'sample_independent_rng', 'encoding': 'RGB8_PNG_lossless',
    'quality_filter': False, 'holdout_collision': 'block', 'seed_namespace': 'geo-e1e2-v2',
})
METHODS['cads'].update({'statistics': 'fp32_population_per_embedding_all_tokens',
                        'noise': 'independent_standard_normal_per_timestep', 'reference_namespace': 'geo-e3-ref-v1',
                        'noise_namespace': 'geo-e3-cads-v1'})
METHODS['ccfg'].update({'distance': 'squared_l2_sum_chw', 'positive': 'cfg_combined'})
METHODS['mining'].update({'processor': 'official', 'drop_last': False, 'tie_break': 'smallest_class_id',
                          'zero_norm': 'error'})
METHODS['geo'] = {
    'prompt': 'Realistic street-level road photograph in {country_name}. Authentic local architecture, traffic direction, lane markings, road signs, utility poles, vegetation and terrain. Preserve road geometry and geographic identity. Natural variations in lighting, weather and colors.',
    'negative_prompt': 'illustration, cartoon, CGI, blurry, warped geometry, reversed traffic, incorrect lane markings, fake road signs, altered text, distorted vehicles, unrealistic architecture, watermark, compression artifacts',
    'token_limit': 77, 'overflow': 'block',
}
METHODS['mixup'].update({'namespace': 'geo-e3-mixup-v1', 'batch_selection': 'uniform_without_replacement_floor',
                          'eligible_anchor': 'final_synthetic', 'labels': 'unchanged'})

DEPLOYMENT_KEYS = {'artifact_root', 'cpu_count', 'memory', 'walltime', 'account', 'partition'}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_config(path: Path) -> dict:
    with Path(path).open(encoding='utf-8-sig') as stream:
        config = json.load(stream, object_pairs_hook=_unique_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    return validate_config(config)


def validate_config(config, role=None):
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    if role is None:
        role = 'benchmark' if 'classifier' in config else 'methods' if ('diffusion' in config or 'variants' in config) else 'deployment'
    if role == 'deployment':
        if set(config) != DEPLOYMENT_KEYS:
            raise ValueError('invalid deployment schema')
        for name, value in config.items():
            if value is None:
                continue
            if name == 'cpu_count':
                if type(value) is not int or value <= 0:
                    raise ValueError('invalid deployment CPU count')
            elif not isinstance(value, str) or not value:
                raise ValueError('invalid deployment value')
    else:
        expected = {'benchmark': RESEARCH, 'methods': METHODS}.get(role)
        if expected is None or json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError('research config differs from confirmed settings')
    return config


def validate_configuration(snapshot):
    import hashlib
    roles = {'benchmark','methods','deployment'}
    if not isinstance(snapshot, dict) or set(snapshot) != {'values','sha256'}:
        raise ValueError('invalid configuration snapshot schema')
    if (not isinstance(snapshot['values'], dict) or set(snapshot['values']) != roles
            or not isinstance(snapshot['sha256'], dict) or set(snapshot['sha256']) != roles):
        raise ValueError('configuration role mismatch')
    for role, value in snapshot['values'].items():
        validate_config(value, role)
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if snapshot['sha256'][role] != digest:
            raise ValueError('configuration hash mismatch')
    return json.loads(json.dumps(snapshot, allow_nan=False))


def verified_configuration(benchmark=None, methods=None, deployment=None):
    import hashlib
    base = Path(__file__).resolve().parents[2] / 'configs'
    paths = {'benchmark': benchmark or base / 'benchmark.json',
             'methods': methods or base / 'methods.json', 'deployment': deployment or base / 'uit.json'}
    values = {name: load_config(path) for name, path in paths.items()}
    hashes = {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
              for name, value in values.items()}
    return validate_configuration({'values': values, 'sha256': hashes})


def resolve_run(root: Path, run_id: str) -> dict:
    from .artifacts import sha256_file, verify_lock
    conditions = {'E0', 'E1E2', 'E3-A1', 'E3-A2', 'E3-A3', 'E3-A4'}
    parts = run_id.split('/')
    if len(parts) != 2 or parts[0] not in conditions or parts[1] not in {'42', '123', '2026'}:
        raise ValueError('invalid official run identity')
    condition, seed = parts[0], int(parts[1])
    root = Path(root).resolve()
    path = (root / 'runs' / condition / f'seed_{seed}' / 'descriptor.json').resolve()
    if not path.is_relative_to(root):
        raise ValueError('run descriptor escapes artifact root')
    with path.open(encoding='utf-8') as stream:
        descriptor = json.load(stream, object_pairs_hook=_unique_pairs)
    if (not isinstance(descriptor, dict) or set(descriptor) != {'run_id', 'condition', 'seed', 'dependencies'}
            or descriptor['run_id'] != run_id or descriptor['condition'] != condition
            or type(descriptor['seed']) is not int or descriptor['seed'] != seed):
        raise ValueError('run descriptor identity/schema mismatch')
    required = {'data', 'pool', 'runtime', 'initialization', 'spec'}
    if condition != 'E0':
        required |= {'strength', 'e1'}
    if condition.startswith('E3-'):
        required |= {'references'}
    if condition in {'E3-A2', 'E3-A3', 'E3-A4'}:
        required |= {'mining'}
    dependencies = descriptor['dependencies']
    if not isinstance(dependencies, dict) or set(dependencies) != required:
        raise ValueError('required run dependency roles mismatch')
    identities = {'runtime': 'shared-runtime', 'initialization': f'initialization:{seed}',
                  'spec': f'spec:{condition}', 'strength': 'selected-strength', 'e1': 'e1',
                  'references': 'reference-bank', 'mining': 'mining'}
    verified = {}
    for role, record in dependencies.items():
        if not isinstance(record, dict) or set(record) != {'path', 'sha256', 'identity'}:
            raise ValueError('invalid dependency record')
        if any(not isinstance(value, str) or not value for value in record.values()):
            raise ValueError('invalid dependency value')
        identity = record['identity']
        if role == 'data':
            if not identity.startswith('common-data:') or len(identity.removeprefix('common-data:')) != 64:
                raise ValueError('invalid data dependency identity')
        elif role == 'pool':
            expected_pool = ('e0:' + dependencies['data']['identity'].removeprefix('common-data:')
                             if condition == 'E0' else 'pool:' + ('E3-A3' if condition == 'E3-A4' else condition))
            if identity != expected_pool:
                raise ValueError('wrong condition pool dependency')
        elif identity != identities[role]:
            raise ValueError('wrong dependency role/seed identity')
        relative = Path(record['path']); dependency = (root / relative).resolve()
        if relative.is_absolute() or not dependency.is_relative_to(root):
            raise ValueError('dependency escapes artifact root')
        try:
            if sha256_file(dependency) != record['sha256']:
                raise ValueError('run dependency hash mismatch')
            lock = verify_lock(dependency, boundary=root)
            verified[role] = lock
            if lock['identity'] != record['identity']:
                raise ValueError('run dependency identity mismatch')
        except OSError as exc:
            raise ValueError('missing run dependency') from exc
    lineage = {'pool': {'data'}, 'initialization': {'runtime'}}
    if condition != 'E0': lineage['pool'] |= {'strength', 'e1'}
    if condition.startswith('E3-'): lineage['pool'] |= {'references'}
    if condition in {'E3-A2','E3-A3','E3-A4'}: lineage['pool'] |= {'mining'}
    if 'references' in verified: lineage['references'] = {'data','strength'}
    if 'mining' in verified: lineage['mining'] = {'references'}
    for consumer, upstreams in lineage.items():
        edges = verified[consumer]['dependencies']
        for upstream in upstreams:
            record = dependencies[upstream]
            if edges.get(record['path']) != record['sha256']:
                raise ValueError('inconsistent run dependency lineage')
    return descriptor
