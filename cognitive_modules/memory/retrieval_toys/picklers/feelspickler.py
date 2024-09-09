import os
import argparse
import numpy as np
import pickle
from textblob import TextBlob
from sklearn.neighbors import KernelDensity
import logging
from openai import OpenAI
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SearchConfig:
    def __init__(self, topk_results=16, chunk_size=32, max_tokens=128, embedding_model="mxbai-embed-large"):
        self.TOPK_RESULTS = topk_results
        self.CHUNK_SIZE = chunk_size
        self.MAXTOKENS = max_tokens
        self.EMBEDDING_MODEL = embedding_model

class SemanticDensitySearch:
    def __init__(self, config):
        self.config = config
        self.chunks = []
        self.sentiments = []
        self.smooth_corpus_vectors = []
        self.adaptive_chunks = []
        self.density_map = None
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="semantic_density_docs")

    def preprocess_and_chunk(self, text, chunk_size=64, overlap=0.5, fallback_chunk_size=512):
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        text_offset = 0

        if len(sentences) <= 1 and '.' not in text:
            logging.warning("No valid punctuation for sentence splitting detected, using fallback method.")
            words = text.split()
            num_chunks = max(1, len(words) // fallback_chunk_size)
            for i in range(0, len(words), fallback_chunk_size):
                chunk_text = ' '.join(words[i:i+fallback_chunk_size])
                start_char = text_offset
                end_char = start_char + len(chunk_text)
                chunks.append({
                    'text': chunk_text,
                    'sentences': [words[i:i+fallback_chunk_size]],
                    'start_char': start_char,
                    'end_char': end_char
                })
                text_offset = end_char + 1
        else:
            words = [sentence.split() for sentence in sentences]
            step_size = max(1, int(chunk_size * (1 - overlap)))
            for i in range(0, len(words), step_size):
                chunk_words = [word for sentence in words[i:i + chunk_size] for word in sentence]
                chunk_text = ' '.join(chunk_words)
                start_char = text_offset
                end_char = start_char + len(chunk_text)
                chunks.append({
                    'text': chunk_text,
                    'sentences': words[i:i + chunk_size],
                    'start_char': start_char,
                    'end_char': end_char
                })
                text_offset = end_char + 1

        return chunks

    def compute_embedding(self, text):
        response = self.client.embeddings.create(input=[text], model=self.config.EMBEDDING_MODEL)
        return np.array(response.data[0].embedding)

    def analyze_sentiment(self, chunks):
        return np.array([TextBlob(chunk['text']).sentiment.polarity for chunk in chunks])

    def smooth_vectors(self, corpus_vectors, sentiments, window_size=5):
        weighted_vectors = corpus_vectors * sentiments.reshape(-1, 1)
        smoothed_vectors = np.zeros_like(corpus_vectors)
        for i in range(len(corpus_vectors)):
            start = max(0, i - window_size // 2)
            end = min(len(corpus_vectors), i + window_size // 2 + 1)
            smoothed_vectors[i] = np.mean(weighted_vectors[start:end], axis=0)
        return smoothed_vectors

    def semantic_density_mapping(self, corpus_vectors, interpolation_points, batch_size=1000):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(corpus_vectors)
        grid_points = np.array(np.meshgrid(interpolation_points[:, 0], interpolation_points[:, 1])).T.reshape(-1, 2)

        density_map = np.zeros((len(interpolation_points), len(interpolation_points)))
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_vectors = np.hstack([batch_points, np.zeros((len(batch_points), corpus_vectors.shape[1] - 2))])
            batch_density = np.exp(kde.score_samples(batch_vectors))

            start_row = i // len(interpolation_points)
            end_row = min((i + batch_size) // len(interpolation_points), len(interpolation_points))
            start_col = i % len(interpolation_points)
            end_col = min(start_col + batch_size, len(interpolation_points))

            rows = end_row - start_row
            cols = end_col - start_col
            density_map[start_row:end_row, start_col:end_col] = batch_density[:rows * cols].reshape(rows, cols)

        return density_map

    def adaptive_chunking(self, chunks, sentiments, density_map, min_chunk_size, max_chunk_size, sentiment_threshold=0.2, density_threshold=0.1):
        adaptive_chunks = []
        current_chunk = {'text': '', 'sentences': [], 'start_char': chunks[0]['start_char'], 'end_char': 0}
        for i in range(len(chunks)):
            current_chunk['text'] += chunks[i]['text'] + ' '
            current_chunk['sentences'].extend(chunks[i]['sentences'])
            current_chunk['end_char'] = chunks[i]['end_char']
            if len(current_chunk['sentences']) >= min_chunk_size and (
                len(current_chunk['sentences']) >= max_chunk_size or
                i == len(chunks) - 1 or
                abs(sentiments[i] - sentiments[i-1]) > sentiment_threshold or
                np.max(np.abs(density_map[i//len(density_map)] - density_map[(i-1)//len(density_map)])) > density_threshold
            ):
                adaptive_chunks.append(current_chunk)
                if i < len(chunks) - 1:
                    current_chunk = {'text': '', 'sentences': [], 'start_char': chunks[i+1]['start_char'], 'end_char': 0}
        return adaptive_chunks

    def build_index(self, file_path, output_file):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()

        self.chunks = self.preprocess_and_chunk(text, chunk_size=self.config.CHUNK_SIZE)
        self.sentiments = self.analyze_sentiment(self.chunks)
        corpus_vectors = np.array([self.compute_embedding(chunk['text']) for chunk in self.chunks])
        self.smooth_corpus_vectors = self.smooth_vectors(corpus_vectors, self.sentiments)
        
        interpolation_points = np.linspace(0, 9, 50)
        interpolation_points = np.array(np.meshgrid(interpolation_points, interpolation_points)).T.reshape(-1, 2)
        self.density_map = self.semantic_density_mapping(self.smooth_corpus_vectors, interpolation_points, batch_size=1000)
        
        self.adaptive_chunks = self.adaptive_chunking(self.chunks, self.sentiments, self.density_map, 
                                                      min_chunk_size=5, max_chunk_size=self.config.MAXTOKENS)

        for i, chunk in enumerate(self.adaptive_chunks):
            self.collection.add(
                ids=[str(i)],
                embeddings=[self.compute_embedding(chunk['text']).tolist()],
                metadatas=[{"start_char": chunk['start_char'], "end_char": chunk['end_char']}],
                documents=[chunk['text']]
            )

        with open(output_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'sentiments': self.sentiments,
                'smooth_corpus_vectors': self.smooth_corpus_vectors,
                'adaptive_chunks': self.adaptive_chunks,
                'density_map': self.density_map
            }, f)

        logging.info(f"Index built and saved to {output_file}")

    def load_index(self, input_file):
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.sentiments = data['sentiments']
        self.smooth_corpus_vectors = data['smooth_corpus_vectors']
        self.adaptive_chunks = data['adaptive_chunks']
        self.density_map = data['density_map']

    def search(self, query, top_k):
        query_vector = self.compute_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )

        processed_results = []
        for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            chunk_text = doc
            start_char = metadata['start_char']
            end_char = metadata['end_char']
            sentiment = round(TextBlob(chunk_text).sentiment.polarity, 2)
            relevance_score = 1 - distance  # Convert distance to similarity score

            processed_results.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'relevance_score': round(relevance_score, 2),
                'sentiment': sentiment
            })

        return processed_results

def main():
    parser = argparse.ArgumentParser(description="Semantic Density Search Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument("file", help="Input text file")
    build_parser.add_argument("output", help="Output file for the pickled index")
    build_parser.add_argument("--embedding_model", default="mxbai-embed-large", help="Ollama embedding model to use")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Pickled index file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top", type=int, default=10, help="Number of top results to display")

    args = parser.parse_args()

    config = SearchConfig(embedding_model=args.embedding_model if 'embedding_model' in args else "mxbai-embed-large")
    search_engine = SemanticDensitySearch(config)

    if args.command == "build":
        search_engine.build_index(args.file, args.output)
    elif args.command == "search":
        search_engine.load_index(args.index)
        results = search_engine.search(args.query, args.top)
        print(f"Top {args.top} results for query: '{args.query}'")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['relevance_score']:.4f}] (Sentiment: {result['sentiment']:.2f})")
            print(f"   Snippet: {result['text'][:200]}...")
            print(f"   Character Range: {result['start_char']}-{result['end_char']}\n")

if __name__ == "__main__":
    main()