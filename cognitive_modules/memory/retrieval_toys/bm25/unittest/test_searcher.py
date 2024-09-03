import unittest
import asyncio
import tempfile
import os
import pickle
from searcher import Searcher
from indexer import Indexer
from config import Config
from data_loader import DataItem

class TestSearcher(unittest.TestCase):
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
        self.searcher = Searcher(self.config)

    def tearDown(self):
        self.loop.close()

    def test_search(self):
        data_items = [
            DataItem("The quick brown fox jumps over the lazy dog.", {"path": "doc1.txt"}),
            DataItem("A fast brown dog outpaces a quick red fox.", {"path": "doc2.txt"}),
            DataItem("The lazy cat sleeps all day long.", {"path": "doc3.txt"})
        ]
        self.loop.run_until_complete(self.indexer.build_index(data_items))

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            self.indexer.save_index(temp_file_path)
            self.searcher.load_index(temp_file_path)

            query = "quick brown fox"
            results = self.loop.run_until_complete(self.searcher.search(query, top_k=2))

            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0][0], DataItem)
            self.assertIsInstance(results[0][1], float)
            self.assertGreater(results[0][1], results[1][1])  # First result should have higher score
            self.assertIn("quick", results[0][0].content.lower())
            self.assertIn("brown", results[0][0].content.lower())
        finally:
            os.unlink(temp_file_path)

if __name__ == '__main__':
    unittest.main()