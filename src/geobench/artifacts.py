"""Content-addressed locks with persistent, exclusive writer ownership."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import io
from contextlib import contextmanager
import threading

_leases = threading.local()
AUTHORITATIVE_LOCK_NAMES = frozenset({
    'lock.json', 'pool_lock.json', 'checkpoint_lock.json', 'selection_lock.json',
    'bank_lock.json', 'feature_lock.json', 'mining_lock.json',
    'evaluation_lock.json', 'report_lock.json',
})


def publish_records(root, relative, identity, contents, dependencies, *, owner_token, lock_name='lock.json'):
    """Publish an immutable bundle of already computed output bytes."""
    if not isinstance(owner_token,str) or not owner_token:
        raise ValueError('owner token required')
    root=Path(root).resolve(); target=_inside(root,relative)
    value=hashlib.sha256((identity+'\0'+owner_token).encode()).hexdigest()
    with active_writer(target):
        token=claim_writer(target,identity,token=value,resume=(target/'.writer').exists())
        try:
            stage=staging_path(target,token)
            for name,content in contents.items():
                path=_inside(stage,name); path.parent.mkdir(parents=True,exist_ok=True)
                complete_temporary(path,content)
            return publish_lock(target,{'identity':identity,'files':list(contents),'dependencies':dependencies,
                'artifact_root':root,'claim':token,'lock_name':lock_name})
        finally:
            token.close()


def _complete_stream(path, source):
    """Complete only a matching prefix of a private, owner-controlled write."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('temporary path is a symlink')
    with path.open('r+b' if path.exists() else 'x+b') as target:
        while True:
            previous = target.read(1024 * 1024)
            if not previous:
                break
            if source.read(len(previous)) != previous:
                raise ValueError('temporary content mismatch')
        remaining = source.read(1024 * 1024)
        if remaining and os.fstat(target.fileno()).st_nlink != 1:
            raise ValueError('cannot extend a linked temporary file')
        while remaining:
            target.write(remaining)
            remaining = source.read(1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def complete_temporary(path, content):
    """Caller must hold the writer lease and authenticate its persistent claim."""
    _complete_stream(path, io.BytesIO(content))



@contextmanager
def active_writer(root):
    """Process-exit-safe OS lease, reentrant only on the owning thread."""
    root = Path(root).resolve()
    held = getattr(_leases, 'held', None)
    if held is None:
        held = _leases.held = {}
    if root not in held:
        root.mkdir(parents=True, exist_ok=True)
        stream = (root / '.active-writer').open('a+b')
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise ValueError('another writer is active') from exc
        held[root] = [stream, 0]
    held[root][1] += 1
    try:
        yield
    finally:
        held[root][1] -= 1
        if held[root][1] == 0:
            stream = held.pop(root)[0]
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, path):
    root = Path(root).resolve()
    if not isinstance(path, str) or not path or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("noncanonical artifact path")
    path = Path(path)
    if path.is_absolute():
        raise ValueError('absolute artifact path')
    result = (root / path).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('path escapes artifact root')
    return result


def _ownership(root, token):
    try:
        claim = json.loads((Path(root) / '.writer' / 'claim.json').read_text(encoding='utf-8'))
        if not isinstance(token, str) or not hmac.compare_digest(claim['token_sha256'], hashlib.sha256(token.encode()).hexdigest()):
            raise ValueError('writer token mismatch')
        return claim['identity']
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid writer claim') from exc


class WriterToken(str):
    """A token carrying the active lease until publication, close, or unwind."""
    def close(self):
        lease = getattr(self, '_lease', None)
        if lease is not None:
            self._lease = None
            getattr(_leases, 'claims', set()).discard(self._root)
            lease.__exit__(None, None, None)

    def __del__(self):
        self.close()


def claim_writer(root, identity, *, token=None, resume=False):
    root = Path(root).resolve()
    claims = getattr(_leases, 'claims', None)
    if claims is None:
        claims = _leases.claims = set()
    if root in claims:
        raise ValueError('writer claim is already active')
    lease = active_writer(root)
    lease.__enter__()
    try:
        value = _claim_writer(root, identity, token=token, resume=resume)
        result = WriterToken(value)
        result._lease = lease
        result._root = root
        claims.add(root)
        return result
    except BaseException:
        lease.__exit__(None, None, None)
        raise


def _claim_writer(root, identity, *, token=None, resume=False):
    if not isinstance(identity, str) or not identity:
        raise ValueError('writer identity required')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    claim = root / '.writer'
    intent = root / '.writer.intent'
    if claim.exists() and not resume:raise FileExistsError('writer claim already exists')
    if intent.exists():
        if not isinstance(token,str) or not token:raise ValueError('owner token required for interrupted claim')
        expected=json.dumps({'identity':identity,'token_sha256':hashlib.sha256(token.encode()).hexdigest()},sort_keys=True).encode()
        complete_temporary(intent,expected)
        if claim.is_symlink() or (root/'.staging').is_symlink():raise ValueError('linked writer initialization')
        claim.mkdir(exist_ok=True)
        if set(p.name for p in claim.iterdir())-{'claim.json'}:raise ValueError('unexpected writer initialization contents')
        complete_temporary(claim/'claim.json',expected)
        (root/'.staging').mkdir(exist_ok=True)
        return token
    if resume:
        if _ownership(root, token) != identity:
            raise ValueError('writer identity mismatch')
        if (root/'.staging').is_symlink():raise ValueError('linked staging directory')
        (root / '.staging').mkdir(exist_ok=True)
        return token
    token = secrets.token_hex(32) if token is None else token
    if not isinstance(token, str) or not token:
        raise ValueError('writer token required')
    if claim.exists():raise FileExistsError('writer claim already exists')
    expected=json.dumps({'identity':identity,'token_sha256':hashlib.sha256(token.encode()).hexdigest()},sort_keys=True).encode()
    complete_temporary(intent,expected)
    claim.mkdir()
    with (claim / 'claim.json').open('x', encoding='utf-8') as stream:
        json.dump({'identity': identity, 'token_sha256': hashlib.sha256(token.encode()).hexdigest()}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    (root / '.staging').mkdir()
    return token


def staging_path(root, token):
    if Path(root).resolve() not in getattr(_leases, 'held', {}):
        raise ValueError('staging requires an active writer lease')
    _ownership(root, token)
    return Path(root).resolve() / '.staging'


def verify_lock(path, boundary=None, _seen=None, _verified=None):
    path = Path(path)
    if path.is_symlink() or path.resolve().name not in AUTHORITATIVE_LOCK_NAMES or any(part.startswith('.') for part in path.resolve().parts):
        raise ValueError('not an authoritative published lock target')
    if path.name not in AUTHORITATIVE_LOCK_NAMES or any(part.startswith(".") for part in path.parts):
        raise ValueError("not an authoritative published lock")
    return _validate_lock(path, boundary=boundary, _seen=_seen, _verified=_verified)


def _validate_lock(path, boundary=None, _seen=None, _verified=None):
    path = Path(path).resolve()
    boundary = Path(boundary).resolve() if boundary is not None else path.parent
    if not path.is_relative_to(boundary):
        raise ValueError('lock escapes trusted root')
    seen = set() if _seen is None else _seen
    verified = {} if _verified is None else _verified
    if path in seen:
        raise ValueError('cyclic lock dependency')
    seen.add(path)
    try:
        from .config import _unique_pairs
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs)
        if set(data) != {'identity', 'files', 'dependencies', 'artifact_root'} or not isinstance(data['identity'], str) or not data['identity'] or not isinstance(data['files'], dict) or not data['files'] or not isinstance(data['dependencies'], dict):
            raise ValueError('invalid lock schema')
        if not isinstance(data['artifact_root'], str) or Path(data['artifact_root']).is_absolute():
            raise ValueError('invalid root reference')
        artifact_root = (path.parent / data['artifact_root']).resolve()
        if not path.is_relative_to(artifact_root) or not artifact_root.is_relative_to(boundary):
            raise ValueError('lock outside artifact root')
        for name, digest in list(data['files'].items()) + list(data['dependencies'].items()):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid SHA-256 schema')
        for name, digest in data['files'].items():
            if not isinstance(name, str) or name.startswith('.') or name in AUTHORITATIVE_LOCK_NAMES:
                raise ValueError('reserved output path')
            if sha256_file(_inside(path.parent, name)) != digest:
                raise ValueError('artifact hash mismatch')
        if path in verified:
            if verified[path] != data:
                raise ValueError('lock changed during verification')
            return data
        for name, digest in data['dependencies'].items():
            dep = _inside(artifact_root, name)
            if sha256_file(dep) != digest:
                raise ValueError('dependency hash mismatch')
            verify_lock(dep, boundary=boundary, _seen=seen, _verified=verified)
        verified[path] = data
        return data
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid or missing artifact') from exc
    finally:
        seen.remove(path)


def publish_lock(root, descriptor):
    try:
        with active_writer(root):
            return _publish_lock(root, descriptor)
    finally:
        token = descriptor.get('claim') if isinstance(descriptor, dict) else None
        if isinstance(token, WriterToken):
            token.close()


def _publish_lock(root, descriptor):
    root = Path(root).resolve()
    if not isinstance(descriptor, dict) or set(descriptor) - {'identity', 'files', 'dependencies', 'claim', 'artifact_root', 'lock_name'} or not {'identity', 'files', 'dependencies', 'claim'} <= set(descriptor):
        raise ValueError('invalid descriptor')
    try:
        if _ownership(root, descriptor['claim']) != descriptor['identity']:
            raise ValueError('writer identity mismatch')
        stage = staging_path(root, descriptor['claim'])
        artifact_root = Path(descriptor.get('artifact_root', root)).resolve()
        if not root.is_relative_to(artifact_root):
            raise ValueError('stage outside artifact root')
        files = descriptor['files']
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise ValueError('unique outputs required')
        if any(not isinstance(name, str) or not name or name.split('/')[0].startswith('.') or name in AUTHORITATIVE_LOCK_NAMES for name in files):
            raise ValueError('invalid output path')
        staged = {name: _inside(stage, name) for name in files}
        outputs = {name: _inside(root, name) for name in files}
        hashes = {name: sha256_file(path) for name, path in sorted(staged.items())}
        deps = {}
        if not isinstance(descriptor['dependencies'], dict):
            raise ValueError('invalid dependencies')
        for name, digest in descriptor['dependencies'].items():
            dep = _inside(artifact_root, name)
            if sha256_file(dep) != digest:
                raise ValueError('dependency hash mismatch')
            verify_lock(dep, boundary=artifact_root)
            deps[name] = digest
        data = {'identity': descriptor['identity'], 'files': hashes, 'dependencies': deps,
                'artifact_root': os.path.relpath(artifact_root, root).replace(os.sep, '/')}
        lock_name = descriptor.get('lock_name', 'lock.json')
        if lock_name not in AUTHORITATIVE_LOCK_NAMES:
            raise ValueError('invalid published lock name')
        if (root / lock_name).exists():
            if verify_lock(root / lock_name, boundary=artifact_root) != data:
                raise ValueError('published lock mismatch')
            return root / lock_name
        copies = stage / '.publishing'
        copies.mkdir(exist_ok=True)
        for name in sorted(staged):
            outputs[name].parent.mkdir(parents=True, exist_ok=True)
            if outputs[name].exists():
                if sha256_file(outputs[name]) != hashes[name]:
                    raise ValueError('existing output mismatch')
                continue
            copy = copies / hashlib.sha256(name.encode('utf-8')).hexdigest()
            with staged[name].open('rb') as source:
                _complete_stream(copy, source)
            if sha256_file(copy) != hashes[name]:
                raise ValueError('staged copy mismatch')
            os.link(copy, outputs[name])
            copy.unlink()
        for name, digest in hashes.items():
            if sha256_file(outputs[name]) != digest:
                raise ValueError('output changed during publication')
        temp = root / '.lock.staging'
        encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        complete_temporary(temp, encoded.encode('utf-8'))
        _validate_lock(temp, boundary=artifact_root)
        os.link(temp, root / lock_name)
        temp.unlink()
        return root / lock_name
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('cannot publish artifact') from exc
