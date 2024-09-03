import pickle
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import DataItem
from config import Config

class Searcher:
    def __init__(self, config: Config):
        self.config = config
        self.data_items: List[DataItem] = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25_scores = {}

    def load_index(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.data_items = data['data_items']
            self.tfidf_vectorizer = data['tfidf_vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.bm25_scores = data['bm25_scores']

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[DataItem, float]]:
        query_vector = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        bm25_score = np.zeros(len(self.data_items))
        for term in query.split():
            if term in self.bm25_scores:
                bm25_score += self.bm25_scores[term]

        bm25_scores = (bm25_score - bm25_score.min()) / (bm25_score.max() - bm25_score.min() + 1e-8)
        cosine_scores = (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min() + 1e-8)

        ensemble_scores = self.config.ensemble_weight * bm25_scores + (1 - self.config.ensemble_weight) * cosine_scores

        top_indices = np.argsort(ensemble_scores)[::-1][:top_k]
        return [(self.data_items[i], float(ensemble_scores[i])) for i in top_indices]