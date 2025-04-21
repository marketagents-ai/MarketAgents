import asyncio
import aiohttp
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from market_agents.web_search.content_extractor import ContentExtractor
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FetchedResult(BaseModel):
    """A simplified container for fetched results before AI summary."""
    url: str
    title: str
    content: Dict[str, Any]
    extraction_method: str
    has_data: bool


class URLFetcher:
    def __init__(self, config, prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.content_extractor = ContentExtractor(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.headers = config.headers

    async def fetch_url(self, session: aiohttp.ClientSession, url: str, query_url_mapping: Dict[str, str]) -> Optional[FetchedResult]:
        try:
            async with self.semaphore:
                original_query = query_url_mapping.get(url, "Unknown query")
                logger.info(f"\n=== Processing URL ===\nURL: {url}\nOriginal Query: {original_query}")
                
                for method in self.config.methods:
                    try:
                        logger.info(f"Trying method {method}")
                        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            if method == "selenium":
                                title, content = await self.content_extractor.extract_with_selenium(url)
                            elif method == "playwright":
                                title, content = await self.content_extractor.extract_with_playwright(url)
                            else:
                                continue

                            if content and isinstance(content, dict):
                                logger.info(f"Successfully extracted content using {method}")
                                has_data = content.get('has_data', False)

                                return FetchedResult(
                                    url=url,
                                    title=title or url,
                                    content=content,
                                    extraction_method=method,
                                    has_data=has_data
                                )

                    except asyncio.TimeoutError:
                        logger.error(f"{method} timed out for {url}")
                        continue
                    except Exception as e:
                        logger.error(f"{method} failed: {str(e)}")
                        continue

                logger.error(f"All extraction methods failed for {url}")
                return None

        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            return None

    async def process_urls(self, urls: List[str], query_url_mapping: Dict[str, str]) -> List[FetchedResult]:
        """Process a list of URLs and return raw extracted results without summaries."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url, query_url_mapping) for url in urls]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
