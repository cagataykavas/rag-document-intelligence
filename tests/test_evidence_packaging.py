"""Package integrity and conservative restoration tests, without model inference."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools.restore_evidence import MANIFEST, restore


@pytest.fixture
def package(tmp_path):
    source = Path(__file__).absolute().parents[1]
    shutil.copytree(source / 'evidence/packed', tmp_path / 'evidence/packed')
    record = json.loads((tmp_path / MANIFEST).read_text())
    (tmp_path / record['path']).parent.mkdir(parents=True, exist_ok=True)
    return tmp_path, record


def save(root, record):
    (root / MANIFEST).write_text(json.dumps(record))


def test_exact_roundtrip_and_idempotence(package):
    root, record = package
    restore(root)
    data = (root / record['path']).read_bytes()
    assert len(data) == record['bytes']
    assert hashlib.sha256(data).hexdigest() == record['sha256']
    assert restore(root) == record


@pytest.mark.parametrize('mutation', ['missing', 'corrupt', 'reordered', 'size', 'digest'])
def test_reject_bad_chunks(package, mutation):
    root, record = package
    first = root / record['chunks'][0]['path']
    if mutation == 'missing':
        first.unlink()
    elif mutation == 'corrupt':
        first.write_bytes(b'corrupt')
    elif mutation == 'reordered':
        record['chunks'].reverse()
    elif mutation == 'size':
        record['chunks'][0]['bytes'] += 1
    else:
        record['sha256'] = '0' * 64
    save(root, record)
    with pytest.raises((OSError, ValueError)):
        restore(root)
    assert not (root / record['path']).exists()


@pytest.mark.parametrize('path', ['../escape', '/tmp/escape', 'evidence/../escape',
                                  'evidence//escape', 'evidence/packed/manifest.json'])
def test_reject_unsafe_destination(package, path):
    root, record = package
    record['path'] = path
    save(root, record)
    with pytest.raises(ValueError):
        restore(root)


def test_reject_existing_unrelated_file(package):
    root, record = package
    target = root / record['path']
    target.write_bytes(b'keep me')
    with pytest.raises(ValueError):
        restore(root)
    assert target.read_bytes() == b'keep me'


@pytest.mark.parametrize('location', ['output', 'parent', 'chunk'])
def test_reject_symlinks(package, location, tmp_path):
    root, record = package
    outside = tmp_path / 'outside'
    outside.mkdir()
    if location == 'output':
        (root / record['path']).symlink_to(outside / 'untouched')
    elif location == 'parent':
        record['path'] = 'evidence/link/file'
        (root / 'evidence/link').symlink_to(outside, target_is_directory=True)
        save(root, record)
    else:
        chunk = root / record['chunks'][0]['path']
        copy = outside / 'copy'
        chunk.rename(copy)
        chunk.symlink_to(copy)
    with pytest.raises((OSError, ValueError)):
        restore(root)
    assert not (outside / 'untouched').exists()
    assert not (outside / 'file').exists()
