import math
from collections import Counter
from typing import List, Dict, Any

class BM25Searcher:
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, memories: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        corpus = [mem["content"].split() for mem in memories]
        doc_freqs = Counter()
        for doc in corpus:
            doc_freqs.update(set(doc))

        idf = {}
        N = len(corpus)
        for word, freq in doc_freqs.items():
            idf[word] = math.log(N - freq + 0.5) - math.log(freq + 0.5)

        avg_dl = sum(len(doc) for doc in corpus) / N

        scores = []
        for i, doc in enumerate(corpus):
            score = self._score(query.split(), doc, idf, avg_dl, len(doc))
            scores.append((score, i))

        top_results = sorted(scores, reverse=True)[:top_k]
        return [
            {**memories[i], "relevance_score": score}
            for score, i in top_results
        ]

    def _score(self, query, doc, idf, avg_dl, doc_len):
        score = 0
        for word in query:
            if word not in idf:
                continue
            freq = doc.count(word)
            numerator = idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg_dl)
            score += numerator / denominator
        return score