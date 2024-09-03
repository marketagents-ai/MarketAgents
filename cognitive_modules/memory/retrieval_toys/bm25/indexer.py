import pickle
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import DataItem
from config import Config
import asyncio
from concurrent.futures import ProcessPoolExecutor

class Indexer:
    def __init__(self, config: Config):
        self.config = config
        self.data_items: List[DataItem] = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.bm25_scores: Dict[str, np.ndarray] = {}

    async def build_index(self, data_items: List[DataItem]):
        self.data_items = data_items
        contents = [item.content for item in data_items]

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(executor, self._calculate_bm25, contents, term) 
                     for term in self.tfidf_vectorizer.get_feature_names_out()]
            self.bm25_scores = dict(zip(self.tfidf_vectorizer.get_feature_names_out(), await asyncio.gather(*tasks)))

    def _calculate_bm25(self, contents: List[str], term: str) -> np.ndarray:
        k1, b = 1.5, 0.75
        tf = np.array([content.count(term) for content in contents])
        doc_len = np.array([len(content.split()) for content in contents])
        avg_doc_len = np.mean(doc_len)
        
        idf = np.log((len(contents) - (tf > 0).sum() + 0.5) / ((tf > 0).sum() + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        
        return idf * numerator / denominator

    def save_index(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'data_items': self.data_items,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'bm25_scores': self.bm25_scores
            }, f)