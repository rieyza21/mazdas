import json
from pathlib import Path
import tempfile
import unittest

from verify_release import verify_dataset, find_project_root


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'eairh/data_train_video/example.mp4'
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'test fixture, not a real video')
        (self.root/'results').mkdir()
        relative = self.video.relative_to(self.root).as_posix()
        (self.root/'results/current_dataset_manifest.json').write_text(
            json.dumps({'samples': [{'path': relative}]}))

    def test_matching_release(self):
        manifest = verify_dataset(self.root)
        self.assertEqual(len(manifest['samples']), 1)

    def test_wrong_root(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete project folder'):
            verify_dataset(self.root/'wrong')

    def test_missing_and_unexpected_paths(self):
        self.video.rename(self.video.with_name('renamed.mp4'))
        with self.assertRaisesRegex(ValueError, r'Missing \(1\)') as caught:
            verify_dataset(self.root)
        self.assertIn('Unexpected (1)', str(caught.exception))
        self.assertIn('example.mp4', str(caught.exception))
        self.assertIn('renamed.mp4', str(caught.exception))

    def test_content_is_not_hashed(self):
        self.video.write_bytes(b'changed')
        verify_dataset(self.root)

    def test_split_manifest_mismatch(self):
        (self.root/'results/current_dataset_manifest.json').write_text('{"samples": []}')
        with self.assertRaisesRegex(ValueError, 'Dataset layout'):
            verify_dataset(self.root)

    def test_root_discovery(self):
        for name in ('eh', 'heh', 'neh', 'owh'):
            (self.root/name).mkdir()
        self.assertEqual(find_project_root(self.root), self.root.resolve())
        self.assertEqual(find_project_root(self.video.parent), self.root.resolve())

    def test_child_discovery_and_ambiguity(self):
        for name in ('eh', 'heh', 'neh', 'owh'):
            (self.root/name).mkdir()
        with tempfile.TemporaryDirectory() as outer:
            import shutil
            child = Path(outer)/'mazdas-main'
            shutil.copytree(self.root, child)
            self.assertEqual(find_project_root(outer), child.resolve())
            shutil.copytree(self.root, Path(outer)/'another-copy')
            with self.assertRaisesRegex(ValueError, 'Multiple dataset folders'):
                find_project_root(outer)


if __name__ == '__main__':
    unittest.main()
