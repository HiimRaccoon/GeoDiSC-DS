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
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    if 'classifier' in config:
        expected = RESEARCH
    elif 'diffusion' in config or 'variants' in config:
        expected = METHODS
    elif set(config) == DEPLOYMENT_KEYS:
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))) for value in config.values()):
            raise ValueError('invalid deployment value')
        return config
    else:
        raise ValueError('unknown configuration schema')
    if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('research config differs from confirmed settings')
    return config


def resolve_run(root: Path, run_id: str) -> dict:
    from .artifacts import sha256_file, verify_lock
    if not run_id or '/' in run_id or '\\' in run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('invalid run identity')
    root = Path(root).resolve()
    path = (root / 'runs' / run_id / 'descriptor.json').resolve()
    if not path.is_relative_to(root):
        raise ValueError('run descriptor escapes artifact root')
    with path.open(encoding='utf-8') as stream:
        descriptor = json.load(stream, object_pairs_hook=_unique_pairs)
    if not isinstance(descriptor, dict) or descriptor.get('run_id') != run_id:
        raise ValueError('run identity mismatch')
    dependencies = descriptor.get('dependencies')
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError('run dependencies are required')
    for name, expected in dependencies.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError('invalid dependency')
        relative = Path(name)
        dependency = (root / relative).resolve()
        if relative.is_absolute() or not dependency.is_relative_to(root):
            raise ValueError('dependency escapes artifact root')
        try:
            if sha256_file(dependency) != expected:
                raise ValueError('run dependency hash mismatch')
            verify_lock(dependency, boundary=root)
        except OSError as exc:
            raise ValueError('missing run dependency') from exc
    return descriptor
