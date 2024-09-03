import unittest
import asyncio
import tempfile
import os
import json
from data_loader import load_data, DataSource, DataItem

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_load_text(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("This is a test content.")
            temp_file_path = temp_file.name

        try:
            data_source = DataSource(type="text", path=temp_file_path, format="plain")
            result = self.loop.run_until_complete(load_data([data_source]))
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], DataItem)
            self.assertEqual(result[0].content, "This is a test content.")
            self.assertEqual(result[0].metadata["path"], temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_load_json(self):
        json_content = [{"key": "value1"}, {"key": "value2"}]
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(json_content, temp_file)
            temp_file_path = temp_file.name

        try:
            data_source = DataSource(type="json", path=temp_file_path, format="json")
            result = self.loop.run_until_complete(load_data([data_source]))
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], DataItem)
            self.assertEqual(result[0].content, '{"key": "value1"}')
            self.assertEqual(result[0].metadata["path"], temp_file_path)
            self.assertEqual(result[0].metadata["index"], 0)
        finally:
            os.unlink(temp_file_path)

    def test_load_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file1_path = os.path.join(temp_dir, "file1.txt")
            file2_path = os.path.join(temp_dir, "file2.txt")
            
            with open(file1_path, 'w') as f1, open(file2_path, 'w') as f2:
                f1.write("Content of file 1")
                f2.write("Content of file 2")

            data_source = DataSource(type="documents", path=temp_dir, format="plain")
            result = self.loop.run_until_complete(load_data([data_source]))
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], DataItem)
            self.assertIn(result[0].content, ["Content of file 1", "Content of file 2"])
            self.assertIn(result[0].metadata["path"], [file1_path, file2_path])

if __name__ == '__main__':
    unittest.main()