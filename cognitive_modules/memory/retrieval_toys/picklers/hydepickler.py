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

class HyDEIndex:
    def __init__(self, llm_model: str = "llama2", embedding_model: str = "mxbai-embed-large"):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

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
        
        self.documents[doc_id] = {
            "content": content,
            "file_path": file_path,
            "token_length": token_length
        }
        
        self.embeddings[doc_id] = self._compute_embedding(content)

    def generate_hypothetical_document(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Generate a concise, factual paragraph that would be a perfect answer to the given query."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        hypothetical_doc = self.generate_hypothetical_document(query)
        query_embedding = self._compute_embedding(hypothetical_doc)
        
        similarities = {}
        for doc_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities[doc_id] = similarity
        
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump((self.documents, self.embeddings, self.llm_model, self.embedding_model), f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            documents, embeddings, llm_model, embedding_model = pickle.load(f)
        instance = cls(llm_model, embedding_model)
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

def build_index(root_folder: str, output_file: str, llm_model: str, embedding_model: str):
    hyde_index = HyDEIndex(llm_model, embedding_model)

    for root, _, files in os.walk(root_folder):
        for file in tqdm(files, desc="Indexing documents"):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                result = process_file(file_path, root_folder)
                if result:
                    doc_id, content, file_path = result
                    hyde_index.add_document(doc_id, content, file_path)

    hyde_index.save(output_file)
    print(f"Index built and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="HyDE Retrieval Search Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("folder", help="Root folder containing .md files")
    build_parser.add_argument("output", help="Output file for the pickled index")
    build_parser.add_argument("--llm_model", default="llama2", help="Ollama model to use for LLM tasks")
    build_parser.add_argument("--embedding_model", default="mxbai-embed-large", help="Ollama model to use for embeddings")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    search_parser.add_argument("--snippet", action="store_true", help="Display a snippet of the matching content")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.folder, args.output, args.llm_model, args.embedding_model)
    elif args.command == "search":
        hyde_index = HyDEIndex.load(args.index)
        results = hyde_index.search(args.query, args.top)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, (doc_id, score) in enumerate(results, 1):
            doc = hyde_index.documents[doc_id]
            print(f"{i}. [{score:.4f}] {doc['file_path']} (Tokens: {doc['token_length']})")
            if args.snippet:
                snippet = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                print(f"   Snippet: {snippet}\n")

if __name__ == "__main__":
    main()