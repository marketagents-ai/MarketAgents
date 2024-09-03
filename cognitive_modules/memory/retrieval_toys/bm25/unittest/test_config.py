import unittest
import tempfile
import os
from config import Config, DataSource

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        config_content = """
        data_sources:
          - type: text
            path: /path/to/text/file.txt
            format: plain
          - type: json
            path: /path/to/json/file.json
            format: json
        index_path: /path/to/index.pkl
        chunk_size: 1000
        overlap: 200
        ensemble_weight: 0.7
        num_workers: 4
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(config_content)
            temp_file_path = temp_file.name

        try:
            config = Config.load(temp_file_path)
            self.assertEqual(len(config.data_sources), 2)
            self.assertIsInstance(config.data_sources[0], DataSource)
            self.assertEqual(config.data_sources[0].type, 'text')
            self.assertEqual(config.index_path, '/path/to/index.pkl')
            self.assertEqual(config.chunk_size, 1000)
            self.assertEqual(config.overlap, 200)
            self.assertEqual(config.ensemble_weight, 0.7)
            self.assertEqual(config.num_workers, 4)
        finally:
            os.unlink(temp_file_path)

if __name__ == '__main__':
    unittest.main()