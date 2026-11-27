"""Confirmed D02 seed namespaces and first-valid attempt ordering."""
import hashlib
import random


def seed_for(method, class_id, index, stream):
    if method not in ('E1', 'E2') or type(class_id) is not int or class_id < 0 or type(index) is not int or index < 0:
        raise ValueError('invalid job identity')
    if stream not in ('source_cycle', 'transforms', 'vae', 'latent'):
        raise ValueError('invalid seed stream')
    key = f'geo-e1e2-v2|314159|{method}|{class_id}|{index}|{stream}'
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')


def source_for_slot(sources, class_id, slot, method='E2'):
    if not sources or len(set(sources)) != len(sources) or type(slot) is not int or slot < 0:
        raise ValueError('invalid source pool or slot')
    ordered = sorted(sources)
    cycle, position = divmod(slot, len(ordered))
    random.Random(seed_for(method, class_id, cycle, 'source_cycle')).shuffle(ordered)
    return ordered[position]


def first_valid(attempt):
    for index in range(5):
        if attempt(index):
            return index
    raise ValueError('quota shortage: five attempts rejected')
