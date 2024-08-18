import os
import re
import pickle
import logging
import tiktoken
from collections import defaultdict
from cache_manager import CacheManager

class InvertedIndexSearch:
    def __init__(self, repo_name, chunk_percentage=10, max_tokens=1000, context_chunks=4):
        self.cache_manager = CacheManager(repo_name)
        self.cache_dir = self.cache_manager.get_cache_dir('inverted_index')
        self.max_tokens = max_tokens
        self.context_chunks = context_chunks
        self.chunk_percentage = chunk_percentage
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.inverted_index = defaultdict(list)
        self.chunks = []
        self.file_names = []

    def chunk_text(self, text, file_name):
        total_tokens = self.count_tokens(text)
        token_threshold = max(100, int(total_tokens * (self.chunk_percentage / 100)))
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for token in tokens:
            current_chunk.append(token)
            current_tokens += 1

            if current_tokens >= token_threshold:
                chunk_text = self.tokenizer.decode(current_chunk)
                chunks.append((chunk_text, file_name))
                current_chunk = []
                current_tokens = 0

        if current_chunk:
            chunk_text = self.tokenizer.decode(current_chunk)
            chunks.append((chunk_text, file_name))

        return chunks

    def process_and_index_content(self, content, file_name):
        chunks = self.chunk_text(content, file_name)
        for chunk, file_name in chunks:
            self.add_to_index(chunk, file_name)

    def add_to_index(self, chunk, file_name):
        chunk_id = len(self.chunks)
        self.chunks.append((chunk, file_name))
        self.file_names.append(file_name)
        
        words = re.findall(r'\w+', chunk.lower())
        for word in words:
            self.inverted_index[word].append(chunk_id)

    def search(self, query, k=8):
        query_words = re.findall(r'\w+', query.lower())
        chunk_scores = defaultdict(int)

        for word in query_words:
            for chunk_id in self.inverted_index.get(word, []):
                chunk_scores[chunk_id] += 1
        
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        total_tokens = 0
        
        for chunk_id, _ in sorted_chunks[:k]:
            context_result = self.get_contextual_chunks(chunk_id, self.max_tokens - total_tokens)
            context_tokens = sum(self.count_tokens(chunk) for chunk, _ in context_result)

            if total_tokens + context_tokens > self.max_tokens:
                break

            results.append(context_result)
            total_tokens += context_tokens

        return results

    def get_contextual_chunks(self, chunk_id, max_tokens):
        start_id = max(0, chunk_id - self.context_chunks)
        end_id = min(len(self.chunks), chunk_id + self.context_chunks + 1)
        context_chunks = self.chunks[start_id:end_id]

        joined_context = []
        current_file = None
        current_chunk = ""
        tokens_used = 0

        for chunk, file_name in context_chunks:
            if file_name != current_file and current_chunk:
                joined_context.append((current_chunk.strip(), current_file))
                current_chunk = ""

            chunk_tokens = self.count_tokens(chunk)
            if tokens_used + chunk_tokens <= max_tokens:
                current_chunk += f"{chunk}\n"
                tokens_used += chunk_tokens
                current_file = file_name
            else:
                break

        if current_chunk:
            joined_context.append((current_chunk.strip(), current_file))

        return joined_context
    
    def clear_cache(self):
        self.inverted_index.clear()
        self.chunks.clear()
        self.file_names.clear()
        # Remove cache files
        for file_name in ['inverted_index.pkl', 'chunks.pkl', 'file_names.pkl']:
            cache_file = os.path.join(self.cache_dir, file_name)
            if os.path.exists(cache_file):
                os.remove(cache_file)
        logging.info("Inverted index cache cleared")
        
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def save_cache(self):
        with open(os.path.join(self.cache_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(self.inverted_index, f)
        with open(os.path.join(self.cache_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(self.cache_dir, 'file_names.pkl'), 'wb') as f:
            pickle.dump(self.file_names, f)
        logging.info("Cache saved successfully.")

    def load_cache(self):
        inverted_index_path = os.path.join(self.cache_dir, 'inverted_index.pkl')
        chunks_path = os.path.join(self.cache_dir, 'chunks.pkl')
        file_names_path = os.path.join(self.cache_dir, 'file_names.pkl')
        if os.path.exists(inverted_index_path) and os.path.exists(chunks_path) and os.path.exists(file_names_path):
            with open(inverted_index_path, 'rb') as f:
                self.inverted_index = pickle.load(f)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            with open(file_names_path, 'rb') as f:
                self.file_names = pickle.load(f)
            logging.info("Cache loaded successfully.")
            return True
        return False