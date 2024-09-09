import os
import argparse
import json
import pickle
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import numpy as np
from openai import OpenAI

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class InvertedIndex:
    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        self.index: Dict[str, Dict[str, int]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.embedding_model = embedding_model
        self.client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',
        )

    def _tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stop_words]

    def _compute_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def add_document(self, doc_id: str, content: str, file_path: str):
        tokens = self._tokenize(content)
        token_length = len(tokens)
        
        # Update inverted index
        for token in set(tokens):
            if token not in self.index:
                self.index[token] = {}
            self.index[token][doc_id] = tokens.count(token)
        
        # Store document info
        self.documents[doc_id] = {
            "content": content,
            "file_path": file_path,
            "token_length": token_length
        }
        
        # Compute and store embedding
        self.embeddings[doc_id] = self._compute_embedding(content)

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[str, float]]:
        query_tokens = self._tokenize(query)
        query_embedding = self._compute_embedding(query)
        
        # Compute TF-IDF scores
        tfidf_scores = {}
        for token in query_tokens:
            if token in self.index:
                idf = np.log(len(self.documents) / len(self.index[token]))
                for doc_id, tf in self.index[token].items():
                    if doc_id not in tfidf_scores:
                        tfidf_scores[doc_id] = 0
                    tfidf_scores[doc_id] += tf * idf
        
        # Compute cosine similarities
        cos_similarities = {}
        for doc_id, embedding in self.embeddings.items():
            cos_similarities[doc_id] = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        
        # Combine scores
        combined_scores = {}
        for doc_id in set(list(tfidf_scores.keys()) + list(cos_similarities.keys())):
            tfidf_score = tfidf_scores.get(doc_id, 0)
            cos_sim = cos_similarities.get(doc_id, 0)
            combined_scores[doc_id] = alpha * tfidf_score + (1 - alpha) * cos_sim
        
        # Sort and return top results
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump((self.index, self.documents, self.embeddings), f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            index, documents, embeddings = pickle.load(f)
        instance = cls()
        instance.index = index
        instance.documents = documents
        instance.embeddings = embeddings
        return instance

def process_file(file_path: str, root_folder: str) -> Tuple[str, str, str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc_id = os.path.relpath(file_path, root_folder)
        return doc_id, content, file_path
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def build_index(root_folder: str, output_file: str):
    inverted_index = InvertedIndex()

    for root, _, files in os.walk(root_folder):
        for file in tqdm(files, desc="Indexing documents"):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                result = process_file(file_path, root_folder)
                if result:
                    doc_id, content, file_path = result
                    inverted_index.add_document(doc_id, content, file_path)

    inverted_index.save(output_file)
    print(f"Index built and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Inverted Index Search Tool with Vector Embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("folder", help="Root folder containing .md files")
    build_parser.add_argument("output", help="Output file for the pickled index")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for TF-IDF score (1-alpha for embedding score)")
    search_parser.add_argument("--snippet", action="store_true", help="Display a snippet of the matching content")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.folder, args.output)
    elif args.command == "search":
        inverted_index = InvertedIndex.load(args.index)
        results = inverted_index.search(args.query, args.top, args.alpha)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, (doc_id, score) in enumerate(results, 1):
            doc = inverted_index.documents[doc_id]
            print(f"{i}. [{score:.4f}] {doc['file_path']} (Tokens: {doc['token_length']})")
            if args.snippet:
                snippet = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                print(f"   Snippet: {snippet}\n")

if __name__ == "__main__":
    main()