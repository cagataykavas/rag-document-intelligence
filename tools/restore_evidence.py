"""Restore the public evidence package without dependencies or overwriting files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


MANIFEST = 'evidence/packed/manifest.json'


def safe_parts(value):
    if not isinstance(value, str):
        raise ValueError('path must be text')
    parts = PurePosixPath(value).parts
    if (not parts or parts[0] != 'evidence' or '..' in parts
            or '/'.join(parts) != value or '\\' in value):
        raise ValueError('unsafe evidence path')
    return parts


def parent_fd(root_fd, value):
    parts = safe_parts(value)
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def read_safe(root_fd, value):
    fd, name = parent_fd(root_fd, value)
    try:
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(child, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('expected regular file')
            return stream.read()
    finally:
        os.close(fd)


def checked(data, record):
    data.decode('utf-8')
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('evidence byte count or digest mismatch')
    return data


def restore(root):
    """POSIX directory-fd traversal refuses symlinks, even in parent directories."""
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        manifest = json.loads(read_safe(root_fd, MANIFEST))
        if manifest['format'] != 1 or not manifest['chunks']:
            raise ValueError('unsupported or empty package')
        output = manifest['path']
        safe_parts(output)
        if output.startswith('evidence/packed/'):
            raise ValueError('output overlaps package')
        parts = []
        for index, chunk in enumerate(manifest['chunks']):
            if chunk['path'] != f'evidence/packed/part-{index:04d}.txt':
                raise ValueError('chunk order/path mismatch')
            if not 0 < chunk['bytes'] <= 48000:
                raise ValueError('invalid chunk size')
            parts.append(checked(read_safe(root_fd, chunk['path']), chunk))
        data = checked(b''.join(parts), manifest)
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != manifest['git_blob_sha1']:
            raise ValueError('Git blob digest mismatch')
        fd, name = parent_fd(root_fd, output)
        try:
            try:
                target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o644, dir_fd=fd)
            except FileExistsError:
                if read_safe(root_fd, output) != data:
                    raise ValueError('refusing to overwrite existing output') from None
            else:
                try:
                    with os.fdopen(target, 'wb') as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    os.unlink(name, dir_fd=fd)
                    raise
        finally:
            os.close(fd)
        return manifest
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).absolute().parent.parent)
    args = parser.parse_args()
    try:
        result = restore(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Evidence restoration refused: {exc}\n')
    print(f"Verified {result['path']}: {result['bytes']} bytes, SHA256 {result['sha256']}")


if __name__ == '__main__':
    main()
