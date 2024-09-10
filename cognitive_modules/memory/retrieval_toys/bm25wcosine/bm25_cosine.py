import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
import tiktoken
from openai import OpenAI

class BM25CosineEnsemble:
    def __init__(self, k1: float = 1.5, b: float = 0.75, ensemble_weight: float = 0.5, max_tokens: int = 8191):
        self.k1 = k1
        self.b = b
        self.ensemble_weight = ensemble_weight
        self.max_tokens = max_tokens
        self.idf: Dict[str, float] = {}
        self.avg_doc_length: float = 0
        self.doc_lengths: Dict[int, int] = {}
        self.total_docs: int = 0
        self.documents: List[str] = []
        self.client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def fit(self, documents: List[str]):
        self.documents = self._chunk_documents(documents)
        self.total_docs = len(self.documents)
        self.doc_lengths = {i: len(self.encoding.encode(doc)) for i, doc in enumerate(self.documents)}
        
        # Handle the case when there are no documents
        if self.total_docs == 0:
            self.avg_doc_length = 0
        else:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

        doc_freqs = {}
        for doc in self.documents:
            for word in set(doc.split()):
                doc_freqs[word] = doc_freqs.get(word, 0) + 1

        self.idf = {word: math.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1)
                    for word, freq in doc_freqs.items()}

    def _chunk_documents(self, documents: List[str]) -> List[str]:
        chunked_docs = []
        for doc in documents:
            sentences = doc.split('.')
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip() + '.'
                if len(self.encoding.encode(current_chunk + sentence)) <= self.max_tokens:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunked_docs.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunked_docs.append(current_chunk)
        return chunked_docs

    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="all-minilm",
            input=[text]
        )
        return response.data[0].embedding

    def _bm25_score(self, query: str, doc_id: int) -> float:
        score = 0
        doc = self.documents[doc_id]
        doc_length = self.doc_lengths[doc_id]
        for term in query.split():
            if term in self.idf:
                tf = doc.split().count(term)
                score += (self.idf[term] * tf * (self.k1 + 1) /
                          (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)))
        return score

    def _cosine_score(self, query: str) -> np.ndarray:
        query_embedding = self._get_embedding(query)
        doc_embeddings = [self._get_embedding(doc) for doc in self.documents]
        return cosine_similarity([query_embedding], doc_embeddings)[0]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.total_docs == 0:
            return []  # Return an empty list if there are no documents

        bm25_scores = np.array([self._bm25_score(query, i) for i in range(self.total_docs)])
        cosine_scores = self._cosine_score(query)

        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        cosine_scores = (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min() + 1e-8)

        ensemble_scores = self.ensemble_weight * bm25_scores + (1 - self.ensemble_weight) * cosine_scores

        top_indices = np.argsort(ensemble_scores)[::-1][:top_k]
        return [(int(i), float(ensemble_scores[i])) for i in top_indices]

class SearchableCollection:
    def __init__(self, items: List[Any], get_text_func: callable):
        self.items = items
        self.get_text = get_text_func
        self.ensemble = BM25CosineEnsemble()
        self.build_index()

    def build_index(self):
        texts = [self.get_text(item) for item in self.items]
        self.ensemble.fit(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
        results = self.ensemble.search(query, top_k)
        return [(self.items[i], score) for i, score in results]

if __name__ == "__main__":
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown dog outpaces a quick red fox",
        "The lazy cat sleeps all day long",
        "Quick foxes and lazy dogs are common in stories"
    ]

    collection = SearchableCollection(documents, lambda x: x)
    query = "quick brown fox"
    results = collection.search(query, top_k=2)

    print(f"Top 2 results for query '{query}':")
    for doc, score in results:
        print(f"Document: {doc} (Score: {score:.4f})")
