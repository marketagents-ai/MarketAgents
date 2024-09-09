import os
import argparse
import pickle
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import concurrent.futures
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf: Dict[str, float] = {}
        self.avg_doc_length: float = 0
        self.doc_lengths: Dict[str, int] = {}
        self.total_docs: int = 0
        self.index: Dict[str, Dict[str, int]] = {}
        self.stop_words = set(stopwords.words('english'))

    def fit(self, documents: List[Tuple[str, str, int]]):
        self.total_docs = len(documents)
        self.avg_doc_length = sum(doc[2] for doc in documents) / self.total_docs

        for doc_id, _, doc_length in documents:
            self.doc_lengths[doc_id] = doc_length

        word_doc_counts = Counter()

        for doc_id, content, _ in tqdm(documents, desc="Processing documents"):
            words = self._tokenize(content)
            word_doc_counts.update(set(words))

            for word in words:
                if word not in self.index:
                    self.index[word] = {}
                if doc_id not in self.index[word]:
                    self.index[word][doc_id] = 0
                self.index[word][doc_id] += 1

        for word, doc_count in word_doc_counts.items():
            self.idf[word] = math.log((self.total_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_words = self._tokenize(query)
        scores = Counter()

        for word in query_words:
            if word in self.index:
                idf = self.idf[word]
                for doc_id, tf in self.index[word].items():
                    doc_length = self.doc_lengths[doc_id]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                    scores[doc_id] += idf * numerator / denominator

        return scores.most_common(top_k)

    def _tokenize(self, text: str) -> List[str]:
        return [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in self.stop_words]

def process_file(file_path: str, root_folder: str) -> Optional[Tuple[str, str, int]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc_id = os.path.relpath(file_path, root_folder)
        token_length = len(content.split())
        return doc_id, content, token_length
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def build_index(root_folder: str) -> Tuple[BM25, Dict[str, Tuple[str, int]]]:
    documents = []
    metadata = {}

    file_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.md'):
                file_paths.append(os.path.join(root, file))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, root_folder) for file_path in file_paths]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(file_paths), desc="Processing files"):
            result = future.result()
            if result:
                doc_id, content, token_length = result
                documents.append((doc_id, content, token_length))
                metadata[doc_id] = (os.path.join(root_folder, doc_id), token_length)

    bm25 = BM25()
    bm25.fit(documents)
    return bm25, metadata

def save_index(bm25: BM25, metadata: Dict[str, Tuple[str, int]], output_file: str):
    with open(output_file, 'wb') as f:
        pickle.dump((bm25, metadata), f)

def load_index(input_file: str) -> Tuple[BM25, Dict[str, Tuple[str, int]]]:
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="BM25 Index Search Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("folder", help="Root folder containing .md files")
    build_parser.add_argument("output", help="Output file for the pickled index")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    search_parser.add_argument("--snippet", action="store_true", help="Display a snippet of the matching content")

    args = parser.parse_args()

    if args.command == "build":
        bm25, metadata = build_index(args.folder)
        save_index(bm25, metadata, args.output)
        print(f"Index built and saved to {args.output}")
    elif args.command == "search":
        bm25, metadata = load_index(args.index)
        results = bm25.search(args.query, args.top)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, (doc_id, score) in enumerate(results, 1):
            file_path, token_length = metadata[doc_id]
            print(f"{i}. [{score:.4f}] {file_path} (Tokens: {token_length})")
            if args.snippet:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Snippet: {snippet}\n")

if __name__ == "__main__":
    main()
