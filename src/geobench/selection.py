"""Strength selection over nine completed validation records."""
import math
import statistics


def select_strength(records):
    strengths = (.25, .35, .5)
    seeds = (42, 123, 2026)
    expected = {(strength, seed) for strength in strengths for seed in seeds}
    values = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'strength','seed','tail_f1','complete'}:
            raise ValueError('invalid candidate record')
        key = (record['strength'], record['seed'])
        value = record['tail_f1']
        if (type(record['seed']) is not int or type(record['strength']) not in (int, float)
                or key not in expected or key in values or record['complete'] is not True
                or isinstance(value, bool) or not isinstance(value, (int,float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('invalid, incomplete or duplicate candidate')
        values[key] = value
    if set(values) != expected:
        raise ValueError('all nine candidate runs are required')
    means = {strength: statistics.mean(values[strength, seed] for seed in seeds) for strength in strengths}
    winner = min(strengths, key=lambda strength: (-means[strength], strength))
    return {'strength': winner, 'means': means}


def select_strength_from_locks(root, references):
    """Verify nine run locks and select tail score at each overall-best epoch."""
    import json
    from pathlib import Path
    from .artifacts import _inside, sha256_file, verify_lock
    root = Path(root).resolve()
    if not isinstance(references, list) or len(references) != 9:
        raise ValueError('nine locked runs are required')
    numeric = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {'strength','seed','path','sha256'}:
            raise ValueError('invalid tuning reference')
        strength, seed = reference['strength'], reference['seed']
        if type(strength) not in (float,int) or strength not in (.25,.35,.5) or type(seed) is not int or seed not in (42,123,2026):
            raise ValueError('invalid tuning identity')
        try:
            path=_inside(root,reference['path'])
            if sha256_file(path) != reference['sha256']:
                raise ValueError('tuning lock hash mismatch')
            lock=verify_lock(path,boundary=root)
            if lock['identity'] != f'tuning:E2:{strength}:{seed}':
                raise ValueError('tuning identity mismatch')
            if not {'history.json','validation_at_best.json','best_checkpoint.pt'} <= set(lock['files']):
                raise ValueError('incomplete tuning artifacts')
            expected = {f'initializations/seed_{seed}/lock.json':f'initialization:{seed}',
                        f'pools/E1E2/candidates/strength_{strength}/pool_lock.json':f'candidate-pool:{strength}',
                        'runtime/lock.json':'shared-runtime'}
            for dep, identity in expected.items():
                if dep not in lock['dependencies']:
                    raise ValueError('missing tuning dependency')
                dependency = verify_lock(_inside(root,dep),boundary=root)
                if dependency['identity'] != identity:
                    raise ValueError('tuning dependency identity mismatch')
            history=json.loads((path.parent/'history.json').read_text(encoding='utf-8'))
            selected=json.loads((path.parent/'validation_at_best.json').read_text(encoding='utf-8'))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('invalid tuning evidence') from exc
        if not isinstance(history,list) or len(history)!=20:
            raise ValueError('complete 20-epoch history required')
        for index, row in enumerate(history,1):
            if (not isinstance(row,dict) or row.get('epoch') != index or isinstance(row.get('macro_f1'),bool)
                    or not isinstance(row.get('macro_f1'),(int,float)) or not math.isfinite(row['macro_f1'])
                    or not 0 <= row['macro_f1'] <= 1):
                raise ValueError('invalid validation history')
        best=max(history,key=lambda row: (row['macro_f1'], -row['epoch']))
        if (not isinstance(selected,dict) or selected.get('epoch') != best['epoch']
                or selected.get('macro_f1') != best['macro_f1']
                or selected.get('checkpoint_sha256') != lock['files']['best_checkpoint.pt']):
            raise ValueError('selected checkpoint is not overall-validation best')
        if 'tail_f1' not in best or selected.get('tail_f1') != best['tail_f1']:
            raise ValueError('selected tail score does not match best-epoch history')
        numeric.append({'strength':strength,'seed':seed,'tail_f1':selected.get('tail_f1'),'complete':True})
    return select_strength(numeric)


def publish_strength_selection(root, references, *, owner_token):
    import hashlib
    import json
    from pathlib import Path
    from .artifacts import active_writer, claim_writer, staging_path, complete_temporary, publish_lock
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    root = Path(root).resolve()
    with active_writer(root):
        result = select_strength_from_locks(root, references)
        dependencies = {entry['path']:entry['sha256'] for entry in references}
        if len(dependencies) != 9:
            raise ValueError('nine distinct run locks required')
        target = root/'tuning/E2/selection'
        token_value = hashlib.sha256(('selection\0'+owner_token).encode()).hexdigest()
        token = claim_writer(target,'selected-strength',token=token_value,resume=(target/'.writer').exists())
        try:
            stage = staging_path(target,token)
            record = {'selected_strength':result['strength'],
                      'means':{str(k):v for k,v in result['means'].items()},
                      'runs':sorted(references,key=lambda item:(item['strength'],item['seed']))}
            complete_temporary(stage/'selected_strength.json',(json.dumps(record,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
            return publish_lock(target,{'identity':'selected-strength','files':['selected_strength.json'],
                                       'dependencies':dependencies,'artifact_root':root,'claim':token,'lock_name':'selection_lock.json'})
        finally:
            token.close()
