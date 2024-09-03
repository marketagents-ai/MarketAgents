import unittest
from long_term_memory import BM25Searcher

class TestBM25Searcher(unittest.TestCase):
    def setUp(self):
        self.searcher = BM25Searcher(None)

    def test_search(self):
        memories = [
            {"id": 1, "content": "The quick brown fox jumps over the lazy dog"},
            {"id": 2, "content": "The lazy cat sleeps all day"},
            {"id": 3, "content": "The brown dog barks at the moon"}
        ]
        results = self.searcher.search("quick fox", memories, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["relevance_score"], results[1]["relevance_score"])

if __name__ == '__main__':
    unittest.main()