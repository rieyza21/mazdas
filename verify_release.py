"""Check dataset layout and split membership without hashing media."""
import json
from pathlib import Path


def find_project_root(start=None):
    start = Path(start or Path.cwd()).expanduser().resolve()
    def valid(path):
        return (path/'results/current_dataset_manifest.json').is_file() and all(
            (path/name).is_dir() for name in ('eairh', 'eh', 'heh', 'neh', 'owh'))
    for parent in (start, *start.parents):
        if valid(parent):
            return parent
    candidates = [p for p in start.iterdir() if p.is_dir() and valid(p)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError('Multiple dataset folders found. Open the notebook from the intended project folder.')
    raise ValueError('Project folder not found. Extract the complete repository and open the notebook from that folder.')


def verify_dataset(root):
    root = Path(root).expanduser().resolve()
    index = root/'results/current_dataset_manifest.json'
    if not index.is_file():
        raise ValueError(f'Incomplete project folder: {root}')
    manifest = json.loads(index.read_text(encoding='utf-8'))
    paths = [r['path'] for r in manifest['samples']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in split manifest.')
    actual = {p.relative_to(root).as_posix()
              for name in ('eairh', 'eh', 'heh', 'neh', 'owh')
              for folder in ('data_train_video', 'data_test_video')
              for p in (root/name/folder).glob('*.mp4') if p.is_file()}
    missing, extra = sorted(set(paths)-actual), sorted(actual-set(paths))
    if missing or extra:
        raise ValueError(
            f'Dataset layout differs from the recorded split in {root}.\n'
            f'Missing ({len(missing)}): {missing[:20]}\n'
            f'Unexpected ({len(extra)}): {extra[:20]}\n'
            'Download the complete dataset from the same revision as the notebook. '
            'Changed datasets need new split assignments and freshly prepared features.')
    print(f'Found {len(actual)} videos matching the split manifest. File contents are not checked.')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default=None)
    args = parser.parse_args()
    try:
        verify_dataset(Path(args.root) if args.root else find_project_root(Path(__file__).parent))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + '\n')
