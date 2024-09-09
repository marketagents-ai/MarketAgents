import os
import argparse
import json
import pickle
from typing import List, Dict, Any, Tuple
import math
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import numpy as np
import hashlib
import sqlite3
from openai import OpenAI

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class EmbeddingCache:
    def __init__(self, cache_file: str):
        self.conn = sqlite3.connect(cache_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                              (hash TEXT PRIMARY KEY, embedding BLOB)''')
        self.conn.commit()

    def get(self, text: str) -> np.ndarray:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.cursor.execute("SELECT embedding FROM embeddings WHERE hash=?", (text_hash,))
        result = self.cursor.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float32)
        return None

    def set(self, text: str, embedding: np.ndarray):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_bytes = embedding.astype(np.float32).tobytes()
        self.cursor.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (text_hash, embedding_bytes))
        self.conn.commit()

    def close(self):
        self.conn.close()

class HybridIndex:
    def __init__(self, embedding_model: str = "mxbai-embed-large", k1: float = 1.5, b: float = 0.75, cache_file: str = "embedding_cache.db"):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.embedding_model = embedding_model
        self.embedding_cache = EmbeddingCache(cache_file)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        # BM25 parameters
        self.k1 = k1
        self.b = b
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf = {}
        self.doc_len = Counter()
        self.total_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stop_words]

    def _compute_embedding(self, text: str) -> np.ndarray:
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        self.embedding_cache.set(text, embedding)
        return embedding

    def add_document(self, doc_id: str, content: str, file_path: str):
        tokens = self._tokenize(content)
        self.doc_len[doc_id] = len(tokens)
        self.total_docs += 1
        
        for token in set(tokens):
            self.doc_freqs[token] += 1

        self.documents[doc_id] = {
            "content": content,
            "file_path": file_path,
            "tokens": tokens
        }
        
        self.embeddings[doc_id] = self._compute_embedding(content)
        self._update_idf()

    def _update_idf(self):
        self.avgdl = sum(self.doc_len.values()) / self.total_docs
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1)

    def _score_bm25(self, query_tokens: List[str], doc_id: str) -> float:
        score = 0.0
        doc_tokens = self.documents[doc_id]["tokens"]
        doc_len = self.doc_len[doc_id]
        for token in set(query_tokens):
            if token in self.idf:
                tf = doc_tokens.count(token)
                numerator = self.idf[token] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[str, float]]:
        query_tokens = self._tokenize(query)
        query_embedding = self._compute_embedding(query)
        
        bm25_scores = {}
        embedding_scores = {}
        
        for doc_id in self.documents:
            bm25_scores[doc_id] = self._score_bm25(query_tokens, doc_id)
            embedding_scores[doc_id] = np.dot(query_embedding, self.embeddings[doc_id]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings[doc_id])
            )
        
        # Advanced normalization: Z-score normalization
        bm25_mean = np.mean(list(bm25_scores.values()))
        bm25_std = np.std(list(bm25_scores.values()))
        embedding_mean = np.mean(list(embedding_scores.values()))
        embedding_std = np.std(list(embedding_scores.values()))
        
        combined_scores = {}
        for doc_id in self.documents:
            normalized_bm25 = (bm25_scores[doc_id] - bm25_mean) / bm25_std if bm25_std != 0 else 0
            normalized_embedding = (embedding_scores[doc_id] - embedding_mean) / embedding_std if embedding_std != 0 else 0
            combined_scores[doc_id] = alpha * normalized_bm25 + (1 - alpha) * normalized_embedding
        
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump((self.documents, self.embeddings, self.doc_freqs, self.idf, self.doc_len, self.total_docs, self.avgdl, self.embedding_model), f)

    @classmethod
    def load(cls, file_path: str, cache_file: str = "embedding_cache.db"):
        with open(file_path, 'rb') as f:
            documents, embeddings, doc_freqs, idf, doc_len, total_docs, avgdl, embedding_model = pickle.load(f)
        instance = cls(embedding_model, cache_file=cache_file)
        instance.documents = documents
        instance.embeddings = embeddings
        instance.doc_freqs = doc_freqs
        instance.idf = idf
        instance.doc_len = doc_len
        instance.total_docs = total_docs
        instance.avgdl = avgdl
        return instance

    def close(self):
        self.embedding_cache.close()

def process_file(file_path: str, root_folder: str) -> Tuple[str, str, str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc_id = os.path.relpath(file_path, root_folder)
        return doc_id, content, file_path
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def build_index(root_folder: str, output_file: str, embedding_model: str, cache_file: str):
    hybrid_index = HybridIndex(embedding_model, cache_file=cache_file)

    for root, _, files in os.walk(root_folder):
        for file in tqdm(files, desc="Indexing documents"):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                result = process_file(file_path, root_folder)
                if result:
                    doc_id, content, file_path = result
                    hybrid_index.add_document(doc_id, content, file_path)

    hybrid_index.save(output_file)
    hybrid_index.close()
    print(f"Index built and saved to {output_file}")

def update_index(index_file: str, new_folder: str, cache_file: str):
    hybrid_index = HybridIndex.load(index_file, cache_file)

    for root, _, files in os.walk(new_folder):
        for file in tqdm(files, desc="Updating index"):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                result = process_file(file_path, new_folder)
                if result:
                    doc_id, content, file_path = result
                    hybrid_index.add_document(doc_id, content, file_path)

    hybrid_index.save(index_file)
    hybrid_index.close()
    print(f"Index updated and saved to {index_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Hybrid BM25 and Embedding Search Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("folder", help="Root folder containing .md files")
    build_parser.add_argument("output", help="Output file for the pickled index")
    build_parser.add_argument("--embedding_model", default="mxbai-embed-large", help="Ollama model to use for embeddings")
    build_parser.add_argument("--cache_file", default="embedding_cache.db", help="SQLite file for caching embeddings")

    update_parser = subparsers.add_parser("update", help="Update the index with new documents")
    update_parser.add_argument("index", help="Existing index file to update")
    update_parser.add_argument("folder", help="Folder containing new .md files to add")
    update_parser.add_argument("--cache_file", default="embedding_cache.db", help="SQLite file for caching embeddings")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score (1-alpha for embedding score)")
    search_parser.add_argument("--snippet", action="store_true", help="Display a snippet of the matching content")
    search_parser.add_argument("--cache_file", default="embedding_cache.db", help="SQLite file for caching embeddings")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.folder, args.output, args.embedding_model, args.cache_file)
    elif args.command == "update":
        update_index(args.index, args.folder, args.cache_file)
    elif args.command == "search":
        hybrid_index = HybridIndex.load(args.index, args.cache_file)
        results = hybrid_index.search(args.query, args.top, args.alpha)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, (doc_id, score) in enumerate(results, 1):
            doc = hybrid_index.documents[doc_id]
            print(f"{i}. [{score:.4f}] {doc['file_path']}")
            if args.snippet:
                snippet = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                print(f"   Snippet: {snippet}\n")
        hybrid_index.close()

if __name__ == "__main__":
    main()