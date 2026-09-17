import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_release import verify_dataset


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
        self.hashes = {relative: hashlib.sha256(self.video.read_bytes()).hexdigest()}
        (self.root/'results/dataset_sha256.json').write_text(json.dumps(self.hashes))
        (self.root/'results/current_dataset_manifest.json').write_text(
            json.dumps({'samples': [{'path': relative}]}))

    def test_matching_release(self):
        hashes, _ = verify_dataset(self.root)
        self.assertEqual(hashes, self.hashes)

    def test_wrong_root(self):
        with self.assertRaisesRegex(ValueError, 'Wrong or incomplete ROOT'):
            verify_dataset(self.root/'wrong')

    def test_missing_and_unexpected_paths(self):
        self.video.rename(self.video.with_name('renamed.mp4'))
        with self.assertRaisesRegex(ValueError, r'Missing \(1\)') as caught:
            verify_dataset(self.root)
        self.assertIn('Unexpected (1)', str(caught.exception))
        self.assertIn('example.mp4', str(caught.exception))
        self.assertIn('renamed.mp4', str(caught.exception))

    def test_modified_content(self):
        self.video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Modified or incomplete'):
            verify_dataset(self.root)

    def test_split_manifest_mismatch(self):
        (self.root/'results/current_dataset_manifest.json').write_text('{"samples": []}')
        with self.assertRaisesRegex(ValueError, 'Split manifest'):
            verify_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
