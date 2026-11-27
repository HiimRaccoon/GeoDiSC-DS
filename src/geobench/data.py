"""Inventory and deterministic exact-duplicate-aware fixed split."""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import __version__ as pillow_version
from .artifacts import sha256_file
from .images import decode, decoded_pixel_sha256

POLICY = 'geolocation_common_split_v2_exact_duplicates'


def quotas(n):
    if n < 1:
        raise ValueError('empty class')
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 0, 1)
    q = (n + 5) // 10 if n >= 10 else 1
    return n - 2 * q, q, q


def inventory(dataset_root, extensions=None):
    root = Path(dataset_root).resolve()
    extensions = {'.jpg'} if extensions is None else extensions
    entries = list(root.iterdir())
    if not entries or any(not p.is_dir() or p.is_symlink() for p in entries):
        raise ValueError('dataset root must contain class directories only')
    classes = sorted(p.name for p in entries)
    mapping = {name: i for i, name in enumerate(classes)}
    rows = []
    for name in classes:
        paths = sorted((root / name).iterdir())
        if not paths:
            raise ValueError(f'empty class: {name}')
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in extensions:
                raise ValueError(f'unexpected dataset entry: {path}')
            relative = path.relative_to(root).as_posix()
            image = decode(path)
            rows.append({'image_id': hashlib.sha256(relative.encode('utf-8')).hexdigest(),
                         'relative_path': relative, 'class_name': name, 'class_id': mapping[name],
                         'width': image.width, 'height': image.height, 'file_size': path.stat().st_size,
                         'sha256': sha256_file(path), 'decoded_pixel_sha256': decoded_pixel_sha256(path)})
    return rows, mapping


def _components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in ('sha256', 'decoded_pixel_sha256'):
            identity = (key, row[key])
            if identity in seen:
                parent[find(i)] = find(seen[identity])
            else:
                seen[identity] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    for component in components.values():
        if len({r['class_id'] for r in component}) != 1:
            raise ValueError('duplicate component has multiple labels')
        group = hashlib.sha256('\n'.join(sorted(r['image_id'] for r in component)).encode()).hexdigest()
        for row in component:
            row['duplicate_group_id'] = group
    return list(components.values())


def split_inventory(inventory_rows):
    rows = [dict(row) for row in inventory_rows]
    components = _components(rows)
    by_class = defaultdict(list)
    for component in components:
        by_class[component[0]['class_id']].append(component)
    for groups in by_class.values():
        class_rows = [r for group in groups for r in group]
        ordered = sorted(class_rows, key=lambda r: (hashlib.sha256(('geolocation-common-split-v1|42|' + r['relative_path']).encode()).hexdigest(), r['relative_path']))
        position = {r['image_id']: i for i, r in enumerate(ordered)}
        train, val, test = quotas(len(ordered))
        for i, row in enumerate(ordered):
            row['split'] = 'train' if i < train else 'val' if i < train + val else 'test'
        if any(len({r['split'] for r in group}) > 1 for group in groups):
            groups.sort(key=lambda g: (-len(g), min(position[r['image_id']] for r in g)))
            duplicate_groups = [g for g in groups if len(g) > 1]
            singles = len(groups) - len(duplicate_groups)
            reachable = [set() for _ in range(len(duplicate_groups) + 1)]
            reachable[-1].add((0, 0))
            for i in range(len(duplicate_groups) - 1, -1, -1):
                size = len(duplicate_groups[i])
                for v, t in reachable[i + 1]:
                    reachable[i].add((v, t))
                    if v + size <= val: reachable[i].add((v + size, t))
                    if t + size <= test: reachable[i].add((v, t + size))
            remaining = [train, val, test]
            def feasible(index, slots):
                return any(v <= slots[1] and t <= slots[2] and slots[1] - v + slots[2] - t <= singles
                           for v, t in reachable[index])
            if not feasible(0, remaining):
                raise ValueError(f'duplicate component quota infeasible for class {ordered[0]["class_name"]}: {quotas(len(ordered))}')
            for i, group in enumerate(groups):
                size = len(group)
                if size == 1: singles -= 1
                for j, name in enumerate(('train', 'val', 'test')):
                    if remaining[j] < size: continue
                    remaining[j] -= size
                    if feasible(min(i + 1, len(duplicate_groups)), remaining):
                        for row in group: row['split'] = name
                        break
                    remaining[j] += size
                else:
                    raise ValueError('duplicate component quota infeasible')
        for row in class_rows:
            row['is_synthetic'] = False
    return sorted(rows, key=lambda r: (r['class_id'], r['relative_path']))


EXPECTED_CLASSES = 124
EXPECTED_IMAGES = 49997
IMAGE_EXTENSIONS = {'.jpg'}


def inventory_fingerprint(rows, mapping):
    import json
    canonical = json.dumps({'class_to_idx': mapping, 'inventory': sorted(rows, key=lambda r: r['relative_path'])},
                           ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _private_site(root):
    """Read separately verified private deployment inputs; no public site defaults."""
    import json
    from .artifacts import verify_lock
    directory = Path(root).resolve() / 'deployment/private_site'
    path = directory / 'lock.json'
    lock = verify_lock(path, boundary=root)
    if lock['identity'] != 'private-site' or 'site.json' not in lock['files']:
        raise ValueError('private deployment input lock mismatch')
    site = json.loads((directory/'site.json').read_text(encoding='utf-8'))
    if not isinstance(site, dict) or set(site) != {'status','ssh_target','login_host','workspace','evidence_files'}:
        raise ValueError('invalid private deployment schema')
    if site['status'] != 'verified' or any(not isinstance(site[name], str) or not site[name] for name in ('ssh_target','login_host','workspace')):
        raise ValueError('private deployment inputs are not verified')
    evidence = site['evidence_files']
    if not isinstance(evidence, list) or not evidence or any(not isinstance(name,str) or name == 'site.json' or name not in lock['files'] for name in evidence):
        raise ValueError('missing locked private deployment evidence')
    return site, path


def _verify_publication_readiness(root, fingerprint, dataset_root, *, _pending=None):
    """Check locked V01/V02/V04 records under deployment/data_readiness.

    Each record references locked evidence_files and says status=verified. V01
    binds the dataset and UIT login identity; V02 binds the artifact root and
    measured storage checks; V04 binds the fingerprint, counts, decoder and
    inventory/split audit checks. Fixture evidence is not deployment evidence.
    """
    import json
    from .artifacts import verify_lock
    evidence = Path(root).resolve() / 'deployment' / 'data_readiness'
    lock_path = evidence / 'lock.json'
    try:
        if _pending is None:
            lock = verify_lock(lock_path, boundary=root)
        else:
            evidence, lock = _pending
        if lock['identity'] != 'data-readiness' or not {'V01.json', 'V02.json', 'V04.json'} <= lock['files'].keys():
            raise ValueError('invalid data readiness lock')
        records = {name: json.loads((evidence / (name + '.json')).read_text(encoding='utf-8'))
                   for name in ('V01', 'V02', 'V04')}
        if any(record.get('status') != 'verified' for record in records.values()):
            raise ValueError('readiness evidence is not verified')
        for name, record in records.items():
            references = record.get('evidence_files')
            if (not isinstance(references, list) or not references
                    or any(not isinstance(item, str) or item not in lock['files']
                           or item in {'V01.json', 'V02.json', 'V04.json'} for item in references)):
                raise ValueError(f'{name} has no locked supporting evidence')
        if records['V01'].get('dataset_root') != str(Path(dataset_root).resolve()):
            raise ValueError('V01 dataset identity mismatch')
        site, site_lock = _private_site(root)
        if any(records['V01'].get(name) != site[name] for name in ('ssh_target','login_host','workspace')):
            raise ValueError('V01 site identity mismatch')
        if lock.get('dependencies', {}).get('deployment/private_site/lock.json') != sha256_file(site_lock):
            raise ValueError('readiness private deployment dependency mismatch')
        if records['V02'].get('artifact_root') != str(Path(root).resolve()):
            raise ValueError('V02 artifact root mismatch')
        storage = records['V02']
        if (any(type(storage.get(key)) is not int or storage[key] <= 0
                for key in ('free_bytes', 'free_inodes', 'required_bytes', 'required_inodes'))
                or storage['free_bytes'] < storage['required_bytes']
                or storage['free_inodes'] < storage['required_inodes']
                or not isinstance(storage.get('model_cache_paths'), list) or not storage['model_cache_paths']
                or any(not isinstance(path, str) or not path for path in storage['model_cache_paths'])
                or any(storage.get(key) is not True for key in ('quota_ok', 'compute_visible', 'atomic_publication', 'single_writer'))):
            raise ValueError('V02 storage evidence incomplete')
        v04 = records['V04']
        checks = v04.get('checks')
        check_names = {'readable', 'file_hashes', 'pixel_hashes', 'duplicate_labels', 'quotas', 'leakage'}
        if (v04.get('fingerprint') != fingerprint or v04.get('images') != EXPECTED_IMAGES
                or v04.get('classes') != EXPECTED_CLASSES or v04.get('decoder') != pillow_version
                or not isinstance(checks, dict) or set(checks) != check_names
                or any(checks[name] is not True for name in check_names)):
            raise ValueError('V04 data identity mismatch')
        return lock_path
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('missing or invalid V01/V02/V04 evidence') from exc


def prepare_data(dataset_root, root, *, owner_token=None, configuration=None):
    from .artifacts import active_writer
    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError('a privately retained owner_token is required for recovery')
    dataset_root, root = Path(dataset_root).resolve(), Path(root).resolve()
    if root == dataset_root or root.is_relative_to(dataset_root) or dataset_root.is_relative_to(root):
        raise ValueError('dataset and artifact roots must be separate')
    from .config import verified_configuration, validate_configuration
    configuration = verified_configuration() if configuration is None else validate_configuration(configuration)
    with active_writer(root):
        return _prepare_data(dataset_root, root, owner_token=owner_token, configuration=configuration)


def _prepare_data(dataset_root, root, *, owner_token=None, configuration=None):
    """Freeze data after verified readiness; retain owner_token to resume safely."""
    import csv
    import io
    import json
    import secrets
    from . import artifacts, config, images
    from .artifacts import claim_writer, publish_lock, staging_path, complete_temporary

    dataset_root, root = Path(dataset_root).resolve(), Path(root).resolve()
    if root == dataset_root or root.is_relative_to(dataset_root) or dataset_root.is_relative_to(root):
        raise ValueError('dataset and artifact roots must be separate')
    rows, mapping = inventory(dataset_root, IMAGE_EXTENSIONS)
    if len(mapping) != EXPECTED_CLASSES or len(rows) != EXPECTED_IMAGES:
        raise ValueError('dataset inventory differs from confirmed image/class counts')
    split = split_inventory(rows)
    audit = {}
    for key in ('image_id', 'sha256', 'decoded_pixel_sha256'):
        sets = {name: {r[key] for r in split if r['split'] == name} for name in ('train', 'val', 'test')}
        if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('duplicate leakage across splits')
        audit[key] = 'no cross-split intersection'
    groups = []
    counts = {}
    for name, class_id in mapping.items():
        subset = [r for r in split if r['class_id'] == class_id]
        ntrain = sum(r['split'] == 'train' for r in subset)
        counts[name] = dict(Counter(r['split'] for r in subset))
        groups.append({'class_name': name, 'class_id': class_id, 'n_original': len(subset), 'n_train': ntrain,
                       'singleton': len(subset) == 1, 'group': 'head' if ntrain >= 100 else 'medium' if ntrain >= 20 else 'extreme-tail'})
    fingerprint = inventory_fingerprint(rows, mapping)
    readiness_lock = _verify_publication_readiness(root, fingerprint, dataset_root)
    if owner_token is not None and (not isinstance(owner_token, str) or not owner_token):
        raise ValueError('owner token must be a nonempty string')
    owner = owner_token or secrets.token_hex(32)
    def acquire(destination, identity, role):
        token = hashlib.sha256((role + '\0' + owner).encode('utf-8')).hexdigest()
        if (destination / '.writer').exists():
            if owner_token is None:
                raise ValueError('existing writer requires explicit owner token')
            return claim_writer(destination, identity, token=token, resume=True)
        return claim_writer(destination, identity, token=token)
    destination = root / 'splits'
    token = acquire(destination, 'common-data:' + fingerprint, 'common-data')
    stage = staging_path(destination, token)
    def write_exact(name, content):
        path = stage / name
        encoded = content.encode('utf-8')
        complete_temporary(path, encoded)
    def write_json(name, obj):
        write_exact(name, json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    def write_csv(name, records, fields):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows({key: str(value).lower() if isinstance(value, bool) else value for key, value in row.items()} for row in records)
        write_exact(name, stream.getvalue())
    fields = list(split[0])
    for name in ('train', 'val', 'test'):
        write_csv(name + '.csv', [r for r in split if r['split'] == name], fields)
    write_csv('inventory.csv', sorted(rows, key=lambda r: r['relative_path']), list(rows[0]))
    write_csv('class_groups.csv', groups, list(groups[0]))
    duplicates = defaultdict(list)
    for row in split:
        duplicates[row['duplicate_group_id']].append({'image_id': row['image_id'], 'relative_path': row['relative_path'], 'class_id': row['class_id'], 'split': row['split'], 'sha256': row['sha256'], 'decoded_pixel_sha256': row['decoded_pixel_sha256']})
    write_json('duplicate_audit.json', {'components': dict(duplicates), 'leakage': audit})
    config_bytes = json.dumps(config.RESEARCH, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':'), allow_nan=False).encode('utf-8')
    write_json('source_identity.json', {'research_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                                       'source_sha256': {module.__name__: sha256_file(module.__file__)
                                                         for module in (config, artifacts, images)},
                                       'geobench.data': sha256_file(__file__)})
    supports = {name: sorted({r['class_id'] for r in split if r['split'] == name}) for name in ('val', 'test')}
    group_by_id = {g['class_id']: g['group'] for g in groups}
    support_by_group = {}
    for name in ('val', 'test'):
        support_by_group[name] = {}
        for group in ('head', 'medium', 'extreme-tail'):
            selected = [r for r in split if r['split'] == name and group_by_id[r['class_id']] == group]
            support_by_group[name][group] = {'classes': sorted({r['class_id'] for r in selected}), 'samples': len(selected)}
    write_json('split_metadata.json', {'policy': POLICY, 'seed': 42, 'fingerprint': fingerprint, 'class_to_idx': mapping,
                                       'counts': counts, 'support_sets': supports,
                                       'support_by_group': support_by_group,
                                       'duplicate_audit_sha256': sha256_file(stage / 'duplicate_audit.json'),
                                       'source_identity_sha256': sha256_file(stage / 'source_identity.json'),
                                       'decoder': {'library': 'Pillow', 'version': pillow_version, 'exif_transpose': True, 'mode': 'RGB'}})
    write_json('configuration.json', configuration)
    files = ['configuration.json', 'train.csv', 'val.csv', 'test.csv', 'inventory.csv', 'class_groups.csv', 'duplicate_audit.json', 'source_identity.json', 'split_metadata.json']
    data_lock = publish_lock(destination, {'identity': 'common-data:' + fingerprint, 'files': files,
                             'dependencies': {'deployment/data_readiness/lock.json': sha256_file(readiness_lock)},
                             'artifact_root': root, 'claim': token})
    e0 = root / 'pools' / 'E0'
    e0_token = acquire(e0, 'e0:' + fingerprint, 'e0')
    descriptor = {'condition': 'E0', 'total_count': len([r for r in split if r['split'] == 'train']),
                  'per_class_counts': {str(g['class_id']): g['n_train'] for g in groups},
                  'non_real_count': 0, 'sample_type': 'real', 'manifest': str(destination / 'train.csv'),
                  'manifest_sha256': sha256_file(destination / 'train.csv'), 'data_lock': str(data_lock)}
    e0_stage = staging_path(e0, e0_token)
    e0_content = (json.dumps(descriptor, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    e0_descriptor = e0_stage / 'pool.json'
    complete_temporary(e0_descriptor, e0_content)
    publish_lock(e0, {'identity': 'e0:' + fingerprint, 'files': ['pool.json'], 'lock_name': 'pool_lock.json',
                      'dependencies': {'splits/lock.json': sha256_file(data_lock)}, 'artifact_root': root, 'claim': e0_token})
    return data_lock


def publish_readiness(dataset_root, root, records, evidence_files, *, owner_token):
    """Import operator observations, validating before persistent claims.

    No network checks run here. Fixtures exercise this interface with simulated
    observations. Production records must contain actual site/audit evidence.
    """
    import json, tempfile
    from .artifacts import active_writer, claim_writer, staging_path, publish_lock, _inside, complete_temporary
    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError('a privately retained owner_token is required')
    root, dataset_root = Path(root).resolve(), Path(dataset_root).resolve()
    if root == dataset_root or root.is_relative_to(dataset_root) or dataset_root.is_relative_to(root):
        raise ValueError('dataset and artifact roots must be separate')
    rows, mapping = inventory(dataset_root, IMAGE_EXTENSIONS)
    if len(rows) != EXPECTED_IMAGES or len(mapping) != EXPECTED_CLASSES:
        raise ValueError('dataset inventory differs from confirmed counts')
    split_inventory(rows)
    fingerprint = inventory_fingerprint(rows, mapping)
    if not isinstance(records, dict) or set(records) != {'V01', 'V02', 'V04'}:
        raise ValueError('readiness requires exactly V01/V02/V04 records')
    with tempfile.TemporaryDirectory() as temporary:
        pending = Path(temporary)
        files = []
        for name, content in evidence_files.items():
            if name in {'V01.json', 'V02.json', 'V04.json'} or not isinstance(content, bytes):
                raise ValueError('invalid supporting evidence')
            path = _inside(pending, name); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content); files.append(name)
        for name, record in records.items():
            filename = name + '.json'
            (pending / filename).write_text(json.dumps(record, sort_keys=True, allow_nan=False), encoding='utf-8')
            files.append(filename)
        _, site_lock = _private_site(root)
        dependencies = {'deployment/private_site/lock.json': sha256_file(site_lock)}
        _verify_publication_readiness(root, fingerprint, dataset_root,
                                      _pending=(pending, {'identity':'data-readiness','files':dict.fromkeys(files), 'dependencies':dependencies}))
        destination = root / 'deployment/data_readiness'
        with active_writer(root):
            token = claim_writer(destination, 'data-readiness', token=owner_token,
                                 resume=(destination / '.writer').exists())
            try:
                stage = staging_path(destination, token)
                for name in files:
                    path = _inside(stage, name); content = (pending/name).read_bytes()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    complete_temporary(path, content)
                return publish_lock(destination, {'identity':'data-readiness','files':files,'dependencies':dependencies,'claim':token,'artifact_root':root})
            finally:
                token.close()
