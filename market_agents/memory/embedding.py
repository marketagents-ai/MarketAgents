import os
import requests
import time
import tiktoken
import logging
from dotenv import load_dotenv

class MemoryEmbedder:
    """
    MemoryEmbedder embeds given text inputs from a specified embedding model.
    """
    def __init__(self, config):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_input = self.config.max_input - 1000
        logging.info(f"Initialized MemoryEmbedder with {config.embedding_provider} provider")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_input tokens using tiktoken."""
        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_input:
            logging.warning(f"Text truncated from {len(tokens)} to {self.max_input} tokens")
            tokens = tokens[:self.max_input]
            text = self.encoding.decode(tokens)
        return text

    def get_embeddings(self, texts):
        """Get embeddings with retry logic and batch processing."""
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else texts

        texts = [self._truncate_text(text) for text in texts]

        if self.config.embedding_provider == "openai":
            all_embeddings = self._get_openai_embeddings(texts)
        elif self.config.embedding_provider == "tei":
            all_embeddings = self._get_tei_embeddings(texts)
        else:
            raise NotImplementedError(
                f"Unknown embedding provider: {self.config.embedding_provider}"
            )

        return all_embeddings[0] if single_input else all_embeddings

    def _get_openai_embeddings(self, texts):
        """Embeddings from OpenAI API."""
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            payload = {
                "input": batch,
                "model": self.config.model
            }
            # Reuse the _send_embedding_request method
            response = self._send_embedding_request(payload, headers)
            response_json = response.json()
            batch_embeddings = [item["embedding"] for item in response_json.get("data", [])]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _get_tei_embeddings(self, texts):
        """Embeddings for local embedding model (TEI)."""
        headers = {"Content-Type": "application/json"}
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            payload = {
                "inputs": batch,
                "model": self.config.model
            }
            # Reuse the _send_embedding_request method
            response = self._send_embedding_request(payload, headers)
            all_embeddings.extend(response.json())

        return all_embeddings

    def _send_embedding_request(self, payload, headers):
        """
        Sends POST request to the embedding API with retry logic. 
        """
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.post(
                    self.config.embedding_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                if response is not None:
                    print(f"Response content: {response.content}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(self.config.retry_delay)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                time.sleep(self.config.retry_delay)

        raise RuntimeError("Unexpected error in _send_embedding_request")