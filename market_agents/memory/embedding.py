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
            raise NotImplementedError(f"Unknown embedding provider: {self.config.embedding_provider}")

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
            batch = texts[i:i+self.config.batch_size]
            payload = {
                "input": batch,
                "model": self.config.model
            }
            for attempt in range(self.config.retry_attempts):
                try:
                    response = requests.post(
                        self.config.embedding_api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    # Extract embeddings from the response
                    response_json = response.json()
                    batch_embeddings = [item["embedding"] for item in response_json.get("data", [])]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise e
                    time.sleep(self.config.retry_delay)
        return all_embeddings


    def _get_tei_embeddings(self, texts):
        """Embeddings for local embedding model."""
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i+self.config.batch_size]
            payload = {
                "inputs": batch,
                "model": self.config.model
            }

            for attempt in range(self.config.retry_attempts):
                try:
                    response = requests.post(
                        self.config.embedding_api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    all_embeddings.extend(response.json())
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise e
                    time.sleep(self.config.retry_delay)
        return all_embeddings

if __name__ == "__main__":
    # test run for embedding
    import os
    from config import load_config_from_yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "memory_config.yaml")

    config = load_config_from_yaml(config_path)
    embedder = MemoryEmbedder(config)
    emb = embedder.get_embeddings("This is a test sentence for embedding.")
    print("Embedding:", emb)