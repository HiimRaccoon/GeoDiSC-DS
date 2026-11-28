"""Confirmed D02 seed namespaces and first-valid attempt ordering."""
import hashlib
import random


def seed_for(method, class_id, index, stream, *, reference=False):
    if method not in ('E1', 'E2') or type(class_id) is not int or class_id < 0 or type(index) is not int or index < 0:
        raise ValueError('invalid job identity')
    if stream not in ('source_cycle', 'transforms', 'vae', 'latent'):
        raise ValueError('invalid seed stream')
    namespace = 'geo-e3-ref-v1' if reference else 'geo-e1e2-v2'
    key = f'{namespace}|314159|{method}|{class_id}|{index}|{stream}'
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')


def source_for_slot(sources, class_id, slot, method='E2', *, reference=False):
    if not sources or len(set(sources)) != len(sources) or type(slot) is not int or slot < 0:
        raise ValueError('invalid source pool or slot')
    ordered = sorted(sources)
    cycle, position = divmod(slot, len(ordered))
    random.Random(seed_for(method, class_id, cycle, 'source_cycle',reference=reference)).shuffle(ordered)
    return ordered[position]


def first_valid(attempt):
    for index in range(5):
        if attempt(index):
            return index
    raise ValueError('quota shortage: five attempts rejected')


def validate_training_groups(train, groups):
    """Check real membership and class/group counts before generating anything."""
    by_class={}; seen=set(); group_ids=set()
    for row in train:
        image_id=row.get('image_id')
        if not image_id or image_id in seen or row.get('split')!='train' or row.get('is_synthetic','').lower()!='false':
            raise ValueError('invalid or duplicate real training membership')
        seen.add(image_id)
        by_class.setdefault(int(row['class_id']),[]).append(row)
    for group in groups:
        class_id=int(group['class_id']); count=int(group['n_train']); original=int(group['n_original'])
        expected_group='head' if count >= 100 else 'medium' if count >= 20 else 'extreme-tail'
        if class_id in group_ids or count < 1 or original < count:
            raise ValueError('invalid training class count')
        group_ids.add(class_id)
        if len(by_class.get(class_id,[])) != count:
            raise ValueError('training class count mismatch')
        if (group.get('group') != expected_group or group.get('singleton','').lower() != str(original == 1).lower()
                or any(row.get('class_name') != group.get('class_name') for row in by_class[class_id])):
            raise ValueError('class/group identity mismatch')
    if set(by_class) != group_ids:
        raise ValueError('class mapping mismatch')
    return by_class


def validate_shared_research(root, data_lock):
    import json
    from pathlib import Path
    from ..artifacts import verify_lock
    from ..config import verified_configuration, validate_configuration
    root=Path(root).resolve();data_lock=Path(data_lock).resolve()
    lock=verify_lock(data_lock,boundary=root)
    if 'configuration.json' not in lock['files']:
        raise ValueError('locked data configuration missing')
    saved=validate_configuration(json.loads((data_lock.parent/'configuration.json').read_text(encoding='utf-8')))
    current=verified_configuration()
    for role in ('benchmark','methods'):
        if saved['sha256'][role] != current['sha256'][role]:
            raise ValueError('locked research configuration mismatch')
    return saved
