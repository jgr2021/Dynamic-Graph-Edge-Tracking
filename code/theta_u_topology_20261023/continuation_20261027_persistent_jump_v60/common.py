"""V60 immutable-source admission and exclusive output helpers."""
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def _json(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=_json)
        f.write('\n')

def live(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=_json)
        f.write('\n')
    os.replace(str(temp), str(path))

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def tick():
    deadline = os.environ.get('V60_DEADLINE')
    if deadline is not None and time.monotonic() > float(deadline):
        raise TimeoutError('V60 fixed parent budget expired')

def verify():
    manifest = read(ROOT/'FREEZE.json')
    count = {}
    for group, base in [('sources', ROOT), ('inputs', PROJECT)]:
        entries = manifest[group]
        for relative, expected in entries.items():
            tick()
            path = (base/relative).resolve()
            if base.resolve() not in path.parents:
                raise ValueError('Admission path escaped workspace: '+relative)
            if sha(path) != expected:
                raise RuntimeError('Frozen file changed: '+relative)
        count[group] = len(entries)
    return dict(counts=count, freeze_sha256=sha(ROOT/'FREEZE.json'))

def runtime():
    """Check the local numeric packages against the admitted V51 runtime."""
    import importlib
    import importlib.metadata
    expected = read(PROJECT/'continuation_20261025_joint_reference_v51/PROTOCOL_V51.json')['environment']
    receipt = {}
    for name in ('numpy', 'scipy', 'numba', 'threadpoolctl'):
        module = importlib.import_module(name)
        actual = dict(version=importlib.metadata.version(name),
                      origin=str(Path(module.__file__).resolve()), sha256=sha(module.__file__))
        pin = expected['packages'][name]
        if (actual['version'] != pin['version'] or
                Path(actual['origin']) != Path(pin['origin']).resolve() or
                actual['sha256'] != pin['sha256']):
            raise RuntimeError('Numeric package changed: '+name)
        receipt[name] = actual
    return receipt
