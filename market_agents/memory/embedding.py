import os
import requests
import time
from dotenv import load_dotenv

class MemoryEmbedder:
    """
    MemoryEmbedder embeds given text inputs from a specified embedding model.
    """
    def __init__(self, config):
        self.config = config

    def get_embeddings(self, texts):
        """Get embeddings with retry logic and batch processing."""
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else texts

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

if __name__ == "__main__":
    # test run for embedding
    from config import load_config_from_yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "memory_config.yaml")

    config = load_config_from_yaml(config_path)
    embedder = MemoryEmbedder(config)
    emb = embedder.get_embeddings("This is a test sentence for embedding.")
    print("Embedding:", emb)