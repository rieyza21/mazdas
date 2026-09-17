"""Verify a downloaded dataset without PyTorch, training, or changing any files."""
import hashlib
import json
from pathlib import Path


def verify_dataset(root):
    root = Path(root).expanduser().resolve()
    index = root / 'results/dataset_sha256.json'
    if not index.is_file():
        raise ValueError(
            f'Wrong or incomplete ROOT: {root}\n'
            'Expected results/dataset_sha256.json beside the five class folders. '
            'Clone/download the complete repository and set ROOT to its extracted directory.')
    hashes = json.loads(index.read_text(encoding='utf-8'))
    classes = ('eairh', 'eh', 'heh', 'neh', 'owh')
    actual = {p.relative_to(root).as_posix() for name in classes
              for folder in ('data_train_video', 'data_test_video')
              for p in (root/name/folder).glob('*.mp4') if p.is_file()}
    expected = set(hashes)
    missing, extra = sorted(expected-actual), sorted(actual-expected)
    if missing or extra:
        details = [f'Dataset does not match the recorded experiment. ROOT: {root}',
                   f'Expected {len(expected)} videos; found {len(actual)}.']
        for title, paths in [('Missing', missing), ('Unexpected', extra)]:
            if paths:
                details.append(f'{title} ({len(paths)}):\n  ' + '\n  '.join(paths[:20]))
                if len(paths) > 20:
                    details.append(f'  ... and {len(paths)-20} more')
        details.append(
            'Use a fresh clone of https://github.com/rieyza21/mazdas.git or a fully '
            'extracted ZIP from the same commit. Set ROOT to the folder containing '
            'mazdas.ipynb, results/, and the five class folders. Do not mix a Drive '
            'dataset or new clips with recorded artifacts. Do not regenerate the '
            'hash manifest or disable this check to reproduce the published experiment.')
        raise ValueError('\n'.join(details))
    changed = []
    for relative, expected_hash in hashes.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Manifest path outside ROOT: {relative}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_hash:
                changed.append(relative)
    if changed:
        raise ValueError('Modified or incomplete video downloads:\n  ' + '\n  '.join(changed)
                         + '\nDownload these files from the same repository commit; '
                         'do not replace recorded hashes.')
    manifest = json.loads((root/'results/current_dataset_manifest.json').read_text(encoding='utf-8'))
    paths = [r['path'] for r in manifest['samples']]
    if len(paths) != len(set(paths)) or set(paths) != actual:
        raise ValueError('Split manifest and media index disagree. Download one complete repository revision.')
    print(f'Verified {len(hashes)} videos against recorded SHA-256 hashes. ROOT: {root}')
    return hashes, manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    try:
        verify_dataset(args.root)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + '\n')
