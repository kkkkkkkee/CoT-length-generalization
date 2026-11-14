#!/usr/bin/env python3
"""Preserve experiment runs as checksum-verified GitHub Release assets.

Requires Python 3.10+ and authenticated gh. Each compressed shard is generated,
uploaded, downloaded for verification, then removed from the temporary cache.
Original experiment files are never deleted. Restore verifies every member.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def gh(*args):
    exe = shutil.which('gh') or '/root/.local/bin/gh'
    return subprocess.check_output([exe, *map(str, args)], text=True).strip()


def safe_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe archive path: {relative}')
    p = (root / rel).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes destination: {relative}')
    return p


def prepare(root, manifest, size):
    files = []
    for p in sorted((root / 'runs').rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink requires manual review: {p}')
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in {'.pyc', '.lock'}:
            continue
        files.append({'path': str(p.relative_to(root)), 'size': p.stat().st_size, 'sha256': sha(p)})
        if len(files) % 1000 == 0:
            print(f'Hashed {len(files)} files', flush=True)
    parts, group, total = [], [], 0
    for row in files:
        if group and total + row['size'] > size:
            parts.append({'name': f'runs-{len(parts)+1:03d}.tar.gz', 'files': group, 'raw_bytes': total})
            group, total = [], 0
        group.append(row)
        total += row['size']
    if group:
        parts.append({'name': f'runs-{len(parts)+1:03d}.tar.gz', 'files': group, 'raw_bytes': total})
    result = {'format': 1, 'scope': 'all runs except caches and transient file locks',
              'files': len(files), 'raw_bytes': sum(x['size'] for x in files), 'parts': parts}
    write(manifest, result)
    return result


def pack(root, part, target):
    with tarfile.open(target, 'w:gz', compresslevel=1) as tar:
        for row in part['files']:
            p = safe_path(root, row['path'])
            if p.stat().st_size != row['size'] or sha(p) != row['sha256']:
                raise RuntimeError(f'Source changed since manifest: {p}')
            info = tar.gettarinfo(str(p), arcname=row['path'])
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            with p.open('rb') as f:
                tar.addfile(info, f)


def unpack(root, part, archive):
    expected = {r['path']: r for r in part['files']}
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise RuntimeError(f'Unexpected archive member: {member.name}')
            row = expected[member.name]
            if member.size != row['size']:
                raise RuntimeError(f'Size mismatch: {member.name}')
            dest = safe_path(root, member.name)
            seen.add(member.name)
            if dest.exists():
                if sha(dest) != row['sha256']:
                    raise RuntimeError(f'Refusing to replace different existing file: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix='.restore-', dir=dest.parent)
            try:
                with os.fdopen(fd, 'wb') as f, tar.extractfile(member) as source:
                    shutil.copyfileobj(source, f)
                if sha(temp) != row['sha256']:
                    raise RuntimeError(f'Content mismatch: {member.name}')
                Path(temp).replace(dest)
            finally:
                Path(temp).unlink(missing_ok=True)
    if seen != set(expected):
        raise RuntimeError('Archive missing expected files')


def download(repo, tag, name, dest):
    gh('release', 'download', tag, '--repo', repo, '--pattern', name, '--output', dest, '--clobber')


def upload(root, manifest_path, repo, tag, cache):
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    cache.mkdir(parents=True, exist_ok=True)
    journal = cache / 'upload-state.json'
    state = json.loads(journal.read_text()) if journal.exists() else {
        'repo': repo, 'tag': tag, 'manifest_sha256': manifest_hash, 'verified': {}}
    if any(state[k] != v for k, v in {'repo': repo, 'tag': tag, 'manifest_sha256': manifest_hash}.items()):
        raise RuntimeError('Upload journal belongs to another archive')
    try:
        release = json.loads(gh('release', 'view', tag, '--repo', repo, '--json', 'assets,isDraft'))
    except subprocess.CalledProcessError:
        gh('release', 'create', tag, '--repo', repo, '--draft', '--title', 'Complete experiment migration archive',
           '--notes', 'All experiment runs, checkpoints and datasets. Restore with migration/github_archive.py; checksums in archive-manifest.json.')
        release = {'assets': [], 'isDraft': True}
    if not release['isDraft']:
        raise RuntimeError('Refusing to alter a published release; use a new tag')
    assets = {a['name']: a for a in release['assets']}
    for part in manifest['parts']:
        name = part['name']
        if name in state['verified'] and name in assets:
            saved = state['verified'][name]
            if saved.get('asset_id') == assets[name].get('id'):
                continue
        local, verify = cache / name, cache / ('verify-' + name)
        print(f'Packing {name} ({part["raw_bytes"] / 2**20:.1f} MiB raw)', flush=True)
        pack(root, part, local)
        digest = sha(local)
        if name not in assets:
            gh('release', 'upload', tag, local, '--repo', repo)
        download(repo, tag, name, verify)
        # Compressed headers may differ after resuming. Verify member contents,
        # rather than replacing an existing asset on a gzip-byte mismatch.
        if sha(verify) != digest:
            with tarfile.open(verify, 'r:gz') as tar:
                expected = {r['path']: r for r in part['files']}
                seen = set()
                for member in tar:
                    if member.name not in expected or member.name in seen or not member.isfile():
                        raise RuntimeError('Remote archive member mismatch')
                    row = expected[member.name]
                    with tar.extractfile(member) as f:
                        h = hashlib.sha256()
                        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                            h.update(block)
                    if member.size != row['size'] or h.hexdigest() != row['sha256']:
                        raise RuntimeError('Remote archive content mismatch')
                    seen.add(member.name)
                if seen != set(expected):
                    raise RuntimeError('Remote archive missing files')
        fresh = json.loads(gh('release', 'view', tag, '--repo', repo, '--json', 'assets'))
        assets = {a['name']: a for a in fresh['assets']}
        state['verified'][name] = {'sha256': sha(verify), 'bytes': verify.stat().st_size,
                                   'asset_id': assets[name]['id']}
        write(journal, state)
        local.unlink()
        verify.unlink()
        print(f'Uploaded and download-verified {name}', flush=True)
    metadata = cache / 'archive-manifest.json'
    manifest['archive_assets'] = state['verified']
    write(metadata, manifest)
    if 'archive-manifest.json' not in assets:
        gh('release', 'upload', tag, metadata, '--repo', repo)
    verify = cache / 'verify-manifest.json'
    download(repo, tag, 'archive-manifest.json', verify)
    if json.loads(verify.read_text()) != manifest:
        raise RuntimeError('Remote manifest mismatch')
    gh('release', 'edit', tag, '--repo', repo, '--draft=false')
    state['complete'] = True
    write(journal, state)
    print(f'https://github.com/{repo}/releases/tag/{tag}', flush=True)


def restore(root, repo, tag, cache):
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'archive-manifest.json'
    download(repo, tag, path.name, path)
    manifest = json.loads(path.read_text())
    for part in manifest['parts']:
        if all(safe_path(root, r['path']).is_file() and sha(safe_path(root, r['path'])) == r['sha256'] for r in part['files']):
            continue
        archive = cache / part['name']
        download(repo, tag, part['name'], archive)
        if sha(archive) != manifest['archive_assets'][part['name']]['sha256']:
            raise RuntimeError(f'Archive checksum mismatch: {archive}')
        unpack(root, part, archive)
        archive.unlink()
        print(f'Restored and verified {part["name"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'upload', 'restore'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--manifest', type=Path, default=Path('migration/archive-manifest.json'))
    p.add_argument('--repo', help='OWNER/REPOSITORY')
    p.add_argument('--tag', default='experiment-archive-2026-11-14')
    p.add_argument('--cache', type=Path, default=Path('/tmp/cot-github-archive'))
    p.add_argument('--part-mib', type=int, default=512)
    a = p.parse_args()
    if a.action == 'prepare':
        if a.manifest.exists():
            p.error('Manifest already exists; choose a new path for a new snapshot')
        r = prepare(a.root.resolve(), a.manifest, a.part_mib * 2**20)
        print(json.dumps({k: r[k] for k in ['files', 'raw_bytes']}), 'parts', len(r['parts']))
    else:
        if not a.repo:
            p.error('--repo OWNER/REPOSITORY is required')
        if a.action == 'upload':
            upload(a.root.resolve(), a.manifest, a.repo, a.tag, a.cache)
        else:
            restore(a.root.resolve(), a.repo, a.tag, a.cache)


if __name__ == '__main__':
    main()
