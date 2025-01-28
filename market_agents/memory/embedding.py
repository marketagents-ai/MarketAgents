import asyncio
import logging
import os
import aiohttp
from dotenv import load_dotenv
import tiktoken


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

    async def get_embeddings(self, texts):
        """Get embeddings with retry logic and batch processing."""
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else texts

        texts = [self._truncate_text(text) for text in texts]

        if self.config.embedding_provider == "openai":
            all_embeddings = await self._get_openai_embeddings(texts)
        elif self.config.embedding_provider == "tei":
            all_embeddings = await self._get_tei_embeddings(texts)
        else:
            raise NotImplementedError(
                f"Unknown embedding provider: {self.config.embedding_provider}"
            )

        return all_embeddings[0] if single_input else all_embeddings

    async def _get_openai_embeddings(self, texts):
        print("DEBUG: Actually inside the real get_embeddings in MemoryEmbedder:", self)

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
            response_json = await self._send_embedding_request(payload, headers)
            batch_embeddings = [item["embedding"] for item in response_json.get("data", [])]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _get_tei_embeddings(self, texts):
        """Embeddings for local embedding model (TEI)."""
        headers = {"Content-Type": "application/json"}
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            payload = {
                "inputs": batch,
                "model": self.config.model
            }
            response_json = await self._send_embedding_request(payload, headers)
            all_embeddings.extend(response_json)

        return all_embeddings

    async def _send_embedding_request(self, payload, headers):
        """Sends POST request to the embedding API with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config.embedding_api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                await asyncio.sleep(self.config.retry_delay)

        raise RuntimeError("Unexpected error in _send_embedding_request")