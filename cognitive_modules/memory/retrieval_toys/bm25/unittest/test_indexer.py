import unittest
import asyncio
import tempfile
import os
import pickle
from indexer import Indexer
from config import Config
from data_loader import DataItem

class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.config = Config({
            'chunk_size': 1000,
            'overlap': 200,
            'ensemble_weight': 0.7,
            'num_workers': 2
        })
        self.indexer = Indexer(self.config)

    def tearDown(self):
        self.loop.close()

    def test_build_index(self):
        data_items = [
            DataItem("This is the first document.", {"path": "doc1.txt"}),
            DataItem("This is the second document.", {"path": "doc2.txt"}),
            DataItem("And this is the third one.", {"path": "doc3.txt"})
        ]
        self.loop.run_until_complete(self.indexer.build_index(data_items))
        
        self.assertEqual(len(self.indexer.data_items), 3)
        self.assertIsNotNone(self.indexer.tfidf_matrix)
        self.assertGreater(len(self.indexer.bm25_scores), 0)

    def test_save_and_load_index(self):
        data_items = [
            DataItem("This is a test document.", {"path": "test.txt"})
        ]
        self.loop.run_until_complete(self.indexer.build_index(data_items))

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            self.indexer.save_index(temp_file_path)
            
            with open(temp_file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.assertIn('data_items', loaded_data)
            self.assertIn('tfidf_vectorizer', loaded_data)
            self.assertIn('tfidf_matrix', loaded_data)
            self.assertIn('bm25_scores', loaded_data)
            
            self.assertEqual(len(loaded_data['data_items']), 1)
            self.assertEqual(loaded_data['data_items'][0].content, "This is a test document.")
        finally:
            os.unlink(temp_file_path)

if __name__ == '__main__':
    unittest.main()