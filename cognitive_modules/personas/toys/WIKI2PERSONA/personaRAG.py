import os
import argparse
import json
from typing import List, Dict, Union
from gpt4all import GPT4All
from nomic import embed
from openai import OpenAI
from groq import Groq
from colorama import init, Fore, Style
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import tiktoken
import nltk
import pickle
from collections import deque
import hashlib
import numpy as np
from typing import List, Dict, Tuple

nltk.download('punkt', quiet=True)

# Initialize colorama
init(autoreset=True)

class PersonaBasedDocQA:
    def __init__(self, config: dict):
        self.config = config
        self.api_type = config.get('api_type', 'ollama')
        self.client = self.get_api_client()
        
        # Define default models for each API type
        self.api_models = {
            'ollama': 'llama3.1:latest',
            'groq': 'llama-3.1-70b-versatile'
        }
        
        # Set the model based on config or default
        self.model = config.get('model', self.api_models.get(self.api_type))
        self.max_tokens = config.get('max_tokens', 512)  # Default to 1024 if not specified
        self.config = config
        self.client = self.get_api_client()
        self.model_name = config.get('model_name', 'orca-mini-3b-gguf2-q4_0.gguf')
        self.llm = GPT4All(self.model_name)
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.document_contents: Dict[str, str] = {}
        self.bm25 = None
        self.all_tokenized_chunks = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.setup_logging()
        self.persona = ""
        self.persona_name = ""
        self.queries = []
        self.chunk_size = config.get('chunk_size', 256)
        self.chunk_overlap = config.get('chunk_overlap', 128)
        self.top_k = config.get('top_k', 10)
        self.relevance_threshold = config.get('relevance_threshold', 0.9)
        self.cache_dir = config.get('cache_dir', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.conversation_log = []
        self.conversation_id = 0
        self.conversation_history = deque(maxlen=5)  # Limit history to last 5 interactions

    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "persona_qa_agent.log")
        
        self.logger = logging.getLogger("PersonaQAAgent")
        self.logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_api_client(self) -> Union[OpenAI, Groq]:
        if self.api_type == 'ollama':
            return OpenAI(
                base_url=self.config.get('api_base_url', 'http://localhost:11434/v1'),
                api_key=self.config.get('api_key', 'ollama')
            )
        elif self.api_type == 'groq':
            return Groq(
                api_key=os.environ.get("GROQ_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
        
    def load_persona(self, persona_file: str):
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                self.persona = f.read().strip()
            
            # Extract persona name from filename
            filename = os.path.basename(persona_file)
            self.persona_name = filename.replace('_persona.md', '').replace('_', ' ').title()
            
            print(f"{Fore.GREEN}Loaded persona '{self.persona_name}' from {persona_file}")
        except Exception as e:
            self.logger.error(f"Error loading persona file: {str(e)}")
            print(f"{Fore.RED}Error loading persona file: {str(e)}")

    def get_cache_filename(self, base_name: str) -> str:
        # Create a unique hash based on the persona name
        persona_hash = hashlib.md5(self.persona_name.encode()).hexdigest()[:8]
        return f"{base_name}_{persona_hash}.pkl"

    def process_folder(self, folder_path: str):
        processed_files = 0
        cache_file = os.path.join(self.cache_dir, self.get_cache_filename(os.path.basename(folder_path)))
        
        if os.path.exists(cache_file):
            print(f"{Fore.YELLOW}Loading cached data for {self.persona_name} from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.documents = cached_data['documents']
                self.embeddings = cached_data['embeddings']
                self.all_tokenized_chunks = cached_data['all_tokenized_chunks']
                self.document_contents = cached_data['document_contents']
            print(f"{Fore.GREEN}Loaded cached data successfully")
            self.recreate_bm25_index()
            return

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            content = f.read()
                        self.add_document(content, filename, file_path)
                        processed_files += 1
                        print(f"{Fore.GREEN}Processed: {file_path}")
                    except Exception as e:
                        print(f"{Fore.RED}Error processing {file_path}: {str(e)}")
        
        if processed_files > 0:
            self.recreate_bm25_index()
            print(f"{Fore.GREEN}Successfully processed {processed_files} .md files in: {folder_path}")
            
            # Cache the processed data
            cache_data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'all_tokenized_chunks': self.all_tokenized_chunks,
                'document_contents': self.document_contents
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"{Fore.GREEN}Cached processed data for {self.persona_name} to {cache_file}")
        else:
            print(f"{Fore.YELLOW}No .md files found in: {folder_path}")

    def add_document(self, content: str, filename: str, file_path: str):
        self.logger.info(f"Processing document: {filename}")
        chunks = self.create_chunks(content, filename)
        embeddings = self.create_embeddings(chunks)
        tokenized_chunks = [self.tokenize(chunk['content']) for chunk in chunks]
        
        self.documents.extend(chunks)
        self.embeddings.extend(embeddings)
        self.all_tokenized_chunks.extend(tokenized_chunks)
        self.document_contents[filename] = content

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in {'.', '!', '?'}:
            text += '.'
        return text

    def create_embeddings(self, chunks: List[Dict]) -> List[np.ndarray]:
        embeddings = []
        for chunk in tqdm(chunks, desc="Creating embeddings", unit="chunk"):
            embedding = embed.text([chunk['content']], model="nomic-embed-text-v1.5", inference_mode="local")['embeddings'][0]
            embeddings.append(embedding)
        return embeddings

    def recreate_bm25_index(self):
        print(f"{Fore.YELLOW}Recreating BM25 index for all documents...")
        bm25_cache = os.path.join(self.cache_dir, self.get_cache_filename("bm25_index"))
        
        if os.path.exists(bm25_cache):
            print(f"{Fore.YELLOW}Loading cached BM25 index for {self.persona_name} from {bm25_cache}")
            with open(bm25_cache, 'rb') as f:
                self.bm25 = pickle.load(f)
            print(f"{Fore.GREEN}Loaded cached BM25 index successfully")
        else:
            decoded_chunks = [self.tokenizer.decode(tc) for tc in self.all_tokenized_chunks]
            self.bm25 = BM25Okapi(decoded_chunks)
            with open(bm25_cache, 'wb') as f:
                pickle.dump(self.bm25, f)
            print(f"{Fore.GREEN}BM25 index recreated and cached for {self.persona_name}")

    def load_queries(self, query_file: str):
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                self.queries = f.read().split('\n\n')
            print(f"{Fore.GREEN}Loaded {len(self.queries)} queries from {query_file}")
        except Exception as e:
            self.logger.error(f"Error loading query file: {str(e)}")
            print(f"{Fore.RED}Error loading query file: {str(e)}")

    @staticmethod
    def safe_normalize(scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores)  # If all scores are the same, return an array of ones
        return (scores - min_score) / (max_score - min_score)

    def create_chunks(self, content: str, source: str) -> List[Dict]:
        sentences = nltk.sent_tokenize(content)
        chunks = []
        current_chunk = ""
        current_chunk_start = 0
        current_tokens = 0
        
        def finalize_chunk(chunk, start, end, tokens):
            return {
                "content": self.clean_text(chunk),
                "source": source,
                "start": start,
                "end": end,
                "tokens": tokens
            }

        for sentence in sentences:
            sentence_tokens = len(self.tokenize(sentence))
            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(finalize_chunk(current_chunk, current_chunk_start, current_chunk_start + len(current_chunk), current_tokens))
                current_chunk = sentence
                current_chunk_start = content.index(sentence)
                current_tokens = sentence_tokens
            else:
                current_chunk += (" " if current_chunk else "") + sentence
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(finalize_chunk(current_chunk, current_chunk_start, current_chunk_start + len(current_chunk), current_tokens))

        return chunks

    def get_neighboring_chunks(self, chunk_index: int, chunks: List[Dict], max_tokens: int) -> Tuple[str, int]:
        combined_chunk = chunks[chunk_index]['content']
        total_tokens = chunks[chunk_index]['tokens']
        left_index, right_index = chunk_index - 1, chunk_index + 1

        while total_tokens < max_tokens and (left_index >= 0 or right_index < len(chunks)):
            if left_index >= 0:
                left_chunk = chunks[left_index]
                if total_tokens + left_chunk['tokens'] <= max_tokens:
                    combined_chunk = left_chunk['content'] + " " + combined_chunk
                    total_tokens += left_chunk['tokens']
                    left_index -= 1
                else:
                    break

            if right_index < len(chunks):
                right_chunk = chunks[right_index]
                if total_tokens + right_chunk['tokens'] <= max_tokens:
                    combined_chunk += " " + right_chunk['content']
                    total_tokens += right_chunk['tokens']
                    right_index += 1
                else:
                    break

        return combined_chunk, total_tokens

    def search_documents(self, query: str) -> List[Dict]:
        query_embedding = embed.text([query], model="nomic-embed-text-v1.5", inference_mode="local")['embeddings'][0]
        embedding_similarities = [np.dot(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        
        tokenized_query = self.tokenize(query)
        tokenized_query_str = self.tokenizer.decode(tokenized_query)
        
        bm25_scores = self.bm25.get_scores(tokenized_query_str.split())
        
        norm_embedding_similarities = self.safe_normalize(embedding_similarities)
        norm_bm25_scores = self.safe_normalize(bm25_scores)
        
        combined_scores = 0.5 * norm_embedding_similarities + 0.5 * norm_bm25_scores
        
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        retrieved_chunks = []
        for i in sorted_indices[:self.top_k]:
            relevance = round(combined_scores[i], 2)
            
            if relevance < self.relevance_threshold:
                continue
            
            chunk = self.documents[i]
            expanded_content, total_tokens = self.get_neighboring_chunks(i, self.documents, self.max_tokens)
            
            retrieved_chunks.append({
                "content": expanded_content,
                "source": chunk['source'],
                "start": chunk['start'],
                "end": chunk['end'],
                "relevance": relevance,
                "tokens": total_tokens
            })
        
        return retrieved_chunks

    def format_chunks(self, chunks: List[Dict]) -> str:
        formatted_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            # Remove '.md' from the source name
            source = os.path.splitext(chunk['source'])[0]
            formatted_chunk = f"[{i}] \"{chunk['content']}\" (Source: {source}, Relevance: {chunk['relevance']:.2f})\n"
            formatted_chunks.append(formatted_chunk)
        return "\n".join(formatted_chunks)

    def format_conversation_history(self) -> str:
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = []
        for i, (query, answer) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Interaction {i}:")
            formatted_history.append(f"Human: {query}")
            formatted_history.append(f"{self.persona_name}: {answer}")
            formatted_history.append("")  # Empty line for readability
        
        return "\n".join(formatted_history)

    def answer_question(self, query: str) -> str:
        relevant_chunks = self.search_documents(query)
        
        print(f"\n{Fore.CYAN}Retrieved Chunks:{Style.RESET_ALL}")
        print(self.format_chunks(relevant_chunks))
        print()

        conversation_history = self.format_conversation_history()

        if not relevant_chunks:
            prompt = f"""
Persona: {self.persona_name}

{self.persona}

You are an AI assistant named {self.persona_name}, embodying the persona described above. Use your general knowledge and perspective to answer the question, as no relevant information was found in the documents. Always stay in character.

Recent conversation history:
{conversation_history}

Question: {query}

Answer (as {self.persona_name}):
"""
        else:
            prompt = f"""
Persona: {self.persona_name}

{self.persona}

You are an AI assistant named {self.persona_name}, embodying the persona described above. Use the following information to answer the question, always staying in character:

Recent conversation history:
{conversation_history}

Relevant information:
{self.format_chunks(relevant_chunks)}

Question: {query}

Answer (as {self.persona_name}):
"""

        # Display the input prompt to the terminal
        print(f"{Fore.YELLOW}Input Prompt:{Style.RESET_ALL}")
        print(prompt)
        print(f"{Fore.YELLOW}End of Input Prompt{Style.RESET_ALL}\n")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": f"You are {self.persona_name}, a helpful assistant with the following characteristics: {self.persona}"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            answer = f"As {self.persona_name}, I apologize, but I encountered an error while processing your question. Could you please rephrase or try a different question?"

        # Update conversation history
        self.conversation_history.append((query, answer))

        # Log the conversation turn with nested docs
        conversation_turn = {
            "conversations": [
                {"from": "system", "value": f"You are {self.persona_name}, a helpful assistant with the following characteristics: {self.persona}"},
                {"from": "human", "value": query},
                {"from": "gpt", "value": answer, "weight": 1}
            ],
            "docs": [
                {
                    "content": chunk['content'],
                    "source": chunk['source'],
                    "relevance": chunk['relevance']
                } for chunk in relevant_chunks
            ] if relevant_chunks else [],
            "id": self.conversation_id
        }
        self.conversation_log.append(conversation_turn)
        self.conversation_id += 1

        return answer

    def save_conversation_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.cache_dir, f"conversation_log_{self.persona_name}_{timestamp}.jsonl")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            for turn in self.conversation_log:
                json.dump(turn, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"{Fore.GREEN}Conversation log for {self.persona_name} saved to {log_file}")

    def run(self):
        for i, query in enumerate(tqdm(self.queries, desc="Processing queries", unit="query"), 1):
            query = query.strip()
            if query:
                print(f"\n{Fore.YELLOW}Query {i}: {Style.RESET_ALL}{query}")
                answer = self.answer_question(query)
                print(f"{Fore.MAGENTA}Assistant ({self.persona_name}): {Style.RESET_ALL}{answer}")
                print("\n" + "="*50 + "\n")  # Separator between queries

        # Save the conversation log to a JSONL file
        self.save_conversation_log()

def main():
    parser = argparse.ArgumentParser(description="Persona-based Document Q&A Agent")
    parser.add_argument("--config", help="Path to the configuration JSON file")
    parser.add_argument("--persona", required=True, help="Path to the persona file")
    parser.add_argument("--docs", required=True, help="Path to the folder containing documents")
    parser.add_argument("--queries", required=True, help="Path to the file containing queries")
    parser.add_argument("--nocache", action="store_true", help="Disable caching and reprocess all documents")
    parser.add_argument("--api", choices=['ollama', 'groq'], default='ollama', help="Choose the API to use (ollama or groq)")
    parser.add_argument("--model", help="Specify the model to use (overrides config and defaults)")
    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as config_file:
                config = json.load(config_file)
            print(f"{Fore.GREEN}Loaded configuration from {args.config}")
        except Exception as e:
            print(f"{Fore.YELLOW}Error loading config file: {str(e)}. Using default configuration.")

    # Add the chosen API type to the config
    config['api_type'] = args.api

    if args.model:
        config['model'] = args.model

    agent = PersonaBasedDocQA(config)
    
    agent.load_persona(args.persona)
    
    if args.nocache:
        print(f"{Fore.YELLOW}Caching disabled. Reprocessing all documents.")
        # Clear existing cache for this persona
        cache_file = os.path.join(agent.cache_dir, agent.get_cache_filename(os.path.basename(args.docs)))
        bm25_cache = os.path.join(agent.cache_dir, agent.get_cache_filename("bm25_index"))
        if os.path.exists(cache_file):
            os.remove(cache_file)
        if os.path.exists(bm25_cache):
            os.remove(bm25_cache)
    
    agent.process_folder(args.docs)
    agent.load_queries(args.queries)
    
    agent.run()

if __name__ == "__main__":
    main()