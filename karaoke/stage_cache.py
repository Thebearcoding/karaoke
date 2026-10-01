"""Content-checked checkpoints inside the existing, budgeted job directory."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class StageCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root/'checkpoints'
        self.directory.mkdir(exist_ok=True)
        self.reused = []

    def files(self, paths):
        records = {}
        for path in paths:
            path = Path(path)
            if not path.exists() or path.is_symlink():
                raise FileNotFoundError(path)
            files = sorted(path.rglob('*')) if path.is_dir() else [path]
            for file in files:
                if file.is_symlink():
                    raise ValueError(f'Checkpoint cannot follow a symlink: {file}')
                if file.is_file():
                    key = str(file.resolve().relative_to(self.root)) if file.resolve().is_relative_to(self.root) else str(file.resolve())
                    records[key] = dict(bytes=file.stat().st_size, sha256=digest(file))
        if not records:
            raise ValueError('Checkpoint has no files')
        return records

    @contextmanager
    def stage(self, name, *, inputs, outputs, config=None, code=()):
        if any(Path(p).is_symlink() for p in outputs):
            raise ValueError('Checkpoint output cannot be a symlink')
        outputs = [Path(p).resolve() for p in outputs]
        if any(not p.is_relative_to(self.root) or p == self.root or p.is_relative_to(self.directory) for p in outputs):
            raise ValueError('Checkpoint outputs must be managed job files')
        signature = dict(schema_version=1, inputs=self.files(inputs),
                         code=self.files(code) if code else {}, config=config)
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        checkpoint = self.directory/(name+'.json')
        try:
            old = json.loads(checkpoint.read_text())
            current = self.files(outputs)
            valid = old['fingerprint'] == fingerprint and bool(old['outputs']) and all(
                current.get(key) == value for key, value in old['outputs'].items())
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if valid:
            self.reused.append(name)
            yield True
            return
        # Invalidate before running: a crashed or half-written stage can never be reused.
        checkpoint.unlink(missing_ok=True)
        for output in outputs:
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink(missing_ok=True)
        started = time.perf_counter()
        yield False
        record = dict(fingerprint=fingerprint, **signature, outputs=self.files(outputs),
                      seconds=round(time.perf_counter()-started, 3))
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(checkpoint)


def retained_files(folder):
    """Failed jobs retain successful stage outputs; normal LRU can still evict them."""
    retained = set()
    for checkpoint in (folder/'checkpoints').glob('*.json'):
        try:
            outputs = json.loads(checkpoint.read_text())['outputs']
            if isinstance(outputs, dict):
                retained.update(key for key in outputs if not Path(key).is_absolute() and '..' not in Path(key).parts)
                retained.add(checkpoint.relative_to(folder).as_posix())
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return retained
