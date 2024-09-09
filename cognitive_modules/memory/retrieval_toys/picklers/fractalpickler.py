import os
import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pickle
import logging
from textblob import TextBlob
import chromadb
from openai import OpenAI
import networkx as nx

nltk.download('punkt', quiet=True)

class FractalSearchConfig:
    def __init__(self, topk_results=8, initial_chunk_size=2, max_iter=8, min_chunk_size=1):
        self.TOPK_RESULTS = topk_results
        self.INITIAL_CHUNK_SIZE = initial_chunk_size
        self.MAX_ITER = max_iter
        self.MIN_CHUNK_SIZE = min_chunk_size

class SemanticGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, data=None):
        self.graph.add_node(node_id, data=data)

    def add_edge(self, node1_id, node2_id, weight=1.0):
        self.graph.add_edge(node1_id, node2_id, weight=weight)

    def get_graph(self):
        return self.graph

class FractalSearch:
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="fractal_search_docs")

    def read_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def preprocess_documents(self, document_files):
        for file_path in document_files:
            document = self.read_document(file_path)
            if document:
                self.documents.append((file_path, document))

    def sentence_vector(self, sentence):
        response = self.client.embeddings.create(input=[sentence], model="mxbai-embed-large")
        return response.data[0].embedding

    def mandelbrot_chunking(self, sentences, chunk_size, query_vector, start_offset=0):
        if len(sentences) <= self.config.MIN_CHUNK_SIZE or self.config.MAX_ITER == 0:
            chunk_text = ' '.join(sentences)
            return [(chunk_text, start_offset, start_offset + len(chunk_text))]

        chunks = []
        current_chunk = []
        current_offset = start_offset

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_vector = self.sentence_vector(chunk_text)
                divergence = np.linalg.norm(query_vector - chunk_vector)
                if divergence > 2:
                    sub_chunks = self.mandelbrot_chunking(current_chunk, chunk_size // 2, query_vector, current_offset)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append((chunk_text, current_offset, current_offset + len(chunk_text)))
                current_offset += len(chunk_text) + 1  # +1 for the space between sentences
                current_chunk = []

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_offset, current_offset + len(chunk_text)))

        return chunks

    def build_index(self, document_folder, output_file):
        document_files = [os.path.join(document_folder, file) for file in os.listdir(document_folder) if file.endswith('.txt') or file.endswith('.md')]
        self.preprocess_documents(document_files)
        
        for i, (file_path, doc) in enumerate(self.documents):
            embedding = self.sentence_vector(doc)
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                metadatas=[{"file_path": file_path}],
                documents=[doc]
            )

        with open(output_file, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Index built and saved to {output_file}")

    def load_index(self, input_file):
        with open(input_file, 'rb') as f:
            self.documents = pickle.load(f)
        # No need to load a separate model, as we're using Ollama for embeddings

    def search(self, query, top_k=10):
        query_vector = self.sentence_vector(query)

        chunked_documents = []
        for file_path, doc in self.documents:
            sentences = sent_tokenize(doc)
            chunks = self.mandelbrot_chunking(sentences, self.config.INITIAL_CHUNK_SIZE, query_vector)
            chunked_documents.extend([(file_path, chunk_text, start_char, end_char) for chunk_text, start_char, end_char in chunks])

        chunk_ids = [chunk_text for _, chunk_text, _, _ in chunked_documents]

        for file_path, chunk_text, start_char, end_char in chunked_documents:
            self.semantic_graph.add_node(chunk_text, data={'file_path': file_path, 'start_char': start_char, 'end_char': end_char})

        self.semantic_graph.add_node("query")

        chunk_vectors = [self.sentence_vector(chunk_text) for chunk_text in chunk_ids]
        similarities = []

        for chunk_vector in chunk_vectors:
            query_norm = np.linalg.norm(query_vector)
            chunk_norm = np.linalg.norm(chunk_vector)
            if query_norm == 0 or chunk_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm)
            similarities.append(similarity)

        for i, chunk1_id in enumerate(chunk_ids):
            self.semantic_graph.add_edge("query", chunk1_id, weight=similarities[i])

        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_chunks = [chunked_documents[i] for i in top_k_indices]

        results = []
        for file_path, chunk_text, start_char, end_char in top_k_chunks:
            sentiment = round(TextBlob(chunk_text).sentiment.polarity, 3)
            relevance_score = similarities[chunk_ids.index(chunk_text)]
            results.append({
                'file_path': file_path,
                'chunk_text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'sentiment': sentiment,
                'relevance_score': relevance_score
            })

        return results

def main():
    parser = argparse.ArgumentParser(description="Fractal Search Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("folder", help="Root folder containing .txt or .md files")
    build_parser.add_argument("output", help="Output file for the pickled index")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")

    args = parser.parse_args()

    config = FractalSearchConfig()
    fractal_search = FractalSearch(config)

    if args.command == "build":
        fractal_search.build_index(args.folder, args.output)
    elif args.command == "search":
        fractal_search.load_index(args.index)
        results = fractal_search.search(args.query, args.top)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['relevance_score']:.4f}] {result['file_path']}")
            print(f"   Snippet: {result['chunk_text'][:200]}...")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Character Range: {result['start_char']}-{result['end_char']}\n")

if __name__ == "__main__":
    main()