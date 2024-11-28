import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import time
from playwright.async_api import async_playwright
import aiohttp
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from newspaper import Article
from requests_html import AsyncHTMLSession
import httpx
from parsel import Selector as ParselSelector
import mechanicalsoup
from googlesearch import search

# from market_agents.models.prompt_models import WebSearchPromptManager
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.logger_utils import *
from market_agents.insert_simulation_data import SimulationDataInserter

logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(logging.NullHandler())
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('web_search.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class WebSearchPrompts(BaseModel):
    basic_summary: str
    key_points: str
    market_impact: str
    trading_implications: str
    search_query_generation: str

class WebSearchPromptManager(BaseModel):
    prompts: WebSearchPrompts
    prompt_file: str = Field(default="web_search_prompt.yaml")

    def __init__(self, **data: Any):
        # Initialize with default prompts first
        default_prompts = {
            "basic_summary": "",
            "key_points": "",
            "market_impact": "",
            "trading_implications": "",
            "search_query_generation": ""
        }
        data["prompts"] = WebSearchPrompts(**default_prompts)
        
        # Initialize BaseModel
        super().__init__(**data)
        
        # Try to load prompts from file
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, self.prompt_file)
            
            with open(full_path, 'r') as file:
                prompt_data = yaml.safe_load(file)
                self.prompts = WebSearchPrompts(**prompt_data)
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {full_path}. Using default empty prompts.")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}. Using default empty prompts.")

    def format_prompt(self, prompt_type: str, variables: dict) -> str:
        """
        Format a prompt by replacing variables in the template.
        
        Args:
            prompt_type: The type of prompt to format (e.g., 'search_query_generation')
            variables: Dictionary of variables to replace in the prompt template
        """
        if not hasattr(self.prompts, prompt_type):
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        prompt_template = getattr(self.prompts, prompt_type)
        try:
            return prompt_template.format(**variables)
        except KeyError as e:
            raise KeyError(f"Missing variable in prompt template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {e}")

class WebSearchResult(BaseModel):
    url: str
    title: str
    content: str
    timestamp: datetime
    status: str
    summary: Optional[dict] = {}
    agent_id: str
    extraction_method: str = "unknown"

class WebSearchConfig(BaseSettings):
    max_concurrent_requests: int = 50
    rate_limit: float = 0.1
    content_max_length: int = 1000000
    request_timeout: int = 30
    search_mode: str = "url"
    urls_per_query: int = 5
    llm_configs: List[LLMConfig] = [
        LLMConfig(
            client="openai",
            model=os.getenv("OPENAI_MODEL"),
            max_tokens=10000,
            temperature=0.7,
            use_cache=True
        ),
    ]
    use_ai_summary: bool = True
    methods: List[str] = ["selenium", "playwright", "beautifulsoup", "newspaper3k", "scrapy", "requests_html", "mechanicalsoup", "httpx"]
    default_method: str = "newspaper3k"
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

class ContentExtractor:
    @staticmethod
    async def extract_with_playwright(url: str, timeout: int = 60000) -> tuple[str, str]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(url, wait_until='networkidle', timeout=timeout)
                await page.wait_for_load_state('networkidle')
                title = await page.title()
                content = await page.content()
                await browser.close()
                return title, content
            except Exception as e:
                logger.error(f"Playwright extraction failed: {str(e)}")
                await browser.close()
                return "", ""

    @staticmethod
    def extract_with_selenium(url: str, timeout: int = 30) -> tuple[str, str]:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(timeout)
        try:
            driver.get(url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            title = driver.title
            content = driver.page_source
            driver.quit()
            return title, content
        except Exception as e:
            logger.error(f"Selenium extraction failed: {str(e)}")
            driver.quit()
            return "", ""

    @staticmethod
    async def extract_with_newspaper3k(url: str) -> tuple[str, str]:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text

    @staticmethod
    async def extract_with_beautifulsoup(url: str, headers: Dict[str, str]) -> tuple[str, str]:
        try:
            response = httpx.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            content = ' '.join([p.text for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            return title, content
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return "", ""

    @staticmethod
    async def extract_with_scrapy(url: str) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            selector = ParselSelector(response.text)
            title = selector.css('h1::text').get() or selector.css('title::text').get()
            content = ' '.join(selector.css('p::text').getall())
            return title, content

    @staticmethod
    async def extract_with_requests_html(url: str) -> tuple[str, str]:
        session = AsyncHTMLSession()
        response = await session.get(url)
        await response.html.arender()
        title = response.html.find('title', first=True)
        content = ' '.join([p.text for p in response.html.find('p')])
        return title.text if title else url, content

    @staticmethod
    async def extract_with_mechanicalsoup(url: str) -> tuple[str, str]:
        browser = mechanicalsoup.StatefulBrowser()
        browser.set_user_agent('Mozilla/5.0')
        page = browser.get(url)
        title = page.soup.find('title')
        content = ' '.join([p.text for p in page.soup.find_all('p')])
        return title.text if title else url, content

    @staticmethod
    async def extract_with_httpx(url: str) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title')
            content = ' '.join([p.text for p in soup.find_all('p')])
            return title.text if title else url, content

class SearchManager:
    def __init__(self, ai_utils):
        self.ai_utils = ai_utils
        self.last_request_time = 0
        self.request_delay = 5  # Delay between requests
        self.max_retries = 3    # Add max_retries
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_urls_for_query(self, query: str, num_results: int = 5) -> List[str]:
        for attempt in range(self.max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                # Use the googlesearch library with more robust parameters
                urls = list(search(
                    query, 
                    num=num_results,
                    stop=num_results,
                    pause=2.0,
                    user_agent=self.headers['User-Agent']
                ))
                
                self.last_request_time = time.time()
                
                if not urls:
                    logger.warning(f"No URLs found for query: {query} on attempt {attempt + 1}")
                    continue
                    
                logger.info(f"Successfully retrieved {len(urls)} URLs for query: {query}")
                return urls
                
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.request_delay * (attempt + 1)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All attempts failed for query: {query}")
                    return []

    def get_urls_for_query(self, query: str, num_results: int = 5) -> List[str]:
        for attempt in range(self.max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                urls = list(search(query, num=num_results, stop=num_results, pause=2.0))
                self.last_request_time = time.time()
                
                logger.info(f"Successfully retrieved {len(urls)} URLs for query: {query}")
                return urls
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for query: {query}")
                    return []

class WebSearchAgent:
    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.results: List[WebSearchResult] = []
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.content_extractor = ContentExtractor()
        
        # Initialize ai_utils first
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=150000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None
        )
        
        # Then initialize search_manager with self.ai_utils
        self.search_manager = SearchManager(self.ai_utils)

    async def process_search_query(self, query: str) -> None:
        """Process a search query and extract content from resulting URLs"""
        try:
            # Get URLs from search
            urls = self.search_manager.get_urls_for_query(query, self.config.urls_per_query)
            if not urls:
                logger.warning(f"No URLs found for query: {query}")
                return

            # Process the URLs
            await self.process_urls(urls)

        except Exception as e:
            logger.error(f"Error processing search query: {str(e)}")

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[WebSearchResult]:
        """Fetch and process a single URL"""
        try:
            async with self.semaphore:
                for method in self.config.methods:
                    try:
                        logger.info(f"Trying method {method} for {url}")
                        
                        if method == "selenium":
                            title, content = self.content_extractor.extract_with_selenium(url)
                        elif method == "playwright":
                            title, content = await self.content_extractor.extract_with_playwright(url)
                        elif method == "beautifulsoup":
                            title, content = self.content_extractor.extract_with_beautifulsoup(url, self.headers)
                        elif method == "newspaper3k":
                            title, content = await self.content_extractor.extract_with_newspaper3k(url)
                        elif method == "requests_html":
                            title, content = await self.content_extractor.extract_with_requests_html(url)
                        elif method == "mechanicalsoup":
                            title, content = await self.content_extractor.extract_with_mechanicalsoup(url)
                        elif method == "httpx":
                            title, content = await self.content_extractor.extract_with_httpx(url)
                        
                        if content:
                            logger.info(f"Successfully extracted content from {url} using {method}")
                            summary = await self.generate_ai_summary(content, url) if self.config.use_ai_summary else {}
                            
                            return WebSearchResult(
                                url=url,
                                title=title or url,
                                content=content[:self.config.content_max_length],
                                timestamp=datetime.now(),
                                status="success",
                                summary=summary,
                                agent_id=str(uuid.uuid4()),
                                extraction_method=method
                            )
                    except Exception as e:
                        logger.error(f"{method} failed for {url}: {str(e)}")
                        continue

                logger.error(f"All extraction methods failed for {url}")
                return WebSearchResult(
                    url=url,
                    title="Error",
                    content="All extraction methods failed",
                    timestamp=datetime.now(),
                    status="error",
                    summary={},
                    agent_id=str(uuid.uuid4()),
                    extraction_method="all_failed"
                )

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    async def process_urls(self, urls: List[str]) -> None:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            self.results = await asyncio.gather(*tasks)

    def save_results(self, output_file: str):
        results_dict = []
        
        logger.info("\n=== ARTICLE SUMMARIES ===")
        
        for result in self.results:
            if result is None:
                continue
                
            try:
                result_data = result.model_dump(exclude_none=True)
                results_dict.append(result_data)
                
                if result.summary:
                    formatted_key_points = '\n'.join([f"- {point}" for point in result.summary.get('key_points', [])])
                    
                    logger.info(f"""
                        === ARTICLE DETAILS ===
                        URL: {result.url}
                        TITLE: {result.title}
                        EXTRACTION METHOD: {result.extraction_method}

                        SUMMARY:
                        {result.summary.get('summary', 'Not available')}

                        KEY POINTS:
                        {formatted_key_points}

                        MARKET IMPACT:
                        {result.summary.get('market_impact', 'Not available')}

                        TRADING IMPLICATIONS:
                        {result.summary.get('trading_implications', 'Not available')}
                        =============================
                                            """)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
    async def process_search_query(self, query: str) -> None:
        search_queries = await self.search_manager.generate_search_queries(query, self.prompt_manager)
        all_urls = []
        for search_query in search_queries:
            urls = self.search_manager.get_urls_for_query(search_query, self.config.urls_per_query)
            all_urls.extend(urls)
        await self.process_urls(all_urls)

    async def process_urls(self, urls: List[str]) -> None:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            self.results = await asyncio.gather(*tasks)

    async def run_search(self, input_data: Union[str, List[str]]) -> None:
        if self.config.search_mode == "query":
            await self.process_search_query(input_data)
        else:
            urls = input_data if isinstance(input_data, list) else [input_data]
            await self.process_urls(urls)

    def save_results(self, output_file: str):
        results_dict = []
        for result in self.results:
            if result is None:
                continue
            try:
                result_data = result.model_dump(exclude_none=True)
                results_dict.append(result_data)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        try:
            db_params = {
                'dbname': os.getenv('DB_NAME', 'market_simulation'),
                'user': os.getenv('DB_USER', 'db_user'),
                'password': os.getenv('DB_PASSWORD', 'db_pwd@123'),
                'host': os.getenv('DB_HOST', 'localhost'),
                                'port': os.getenv('DB_PORT', '5432')
            }
            
            inserter = SimulationDataInserter(db_params)
            
            if inserter.test_connection():
                logger.info("Database connection successful")
                inserter.insert_article_summaries(results_dict)
                logger.info(f"Successfully inserted {len(results_dict)} article summaries into database")
            else:
                raise Exception("Database connection test failed")
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            logger.info(f"Results saved to file: {output_file}")

def load_config(config_path: Path = Path("./market_agents/web_search_config.yaml")) -> WebSearchConfig:
    try:
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        config_path = script_dir / "web_search_config.yaml"  # Simplified path
        
        logger.info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
        # Load environment variables with defaults from yaml
        yaml_data['max_concurrent_requests'] = int(os.getenv('WEB_SEARCH_MAX_REQUESTS', yaml_data.get('max_concurrent_requests', 50)))
        yaml_data['rate_limit'] = float(os.getenv('WEB_SEARCH_RATE_LIMIT', yaml_data.get('rate_limit', 0.1)))
        yaml_data['content_max_length'] = int(os.getenv('WEB_SEARCH_CONTENT_LENGTH', yaml_data.get('content_max_length', 1000)))
        yaml_data['request_timeout'] = int(os.getenv('WEB_SEARCH_TIMEOUT', yaml_data.get('request_timeout', 30)))
        
        return WebSearchConfig(**yaml_data)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return WebSearchConfig()

def load_urls(input_file: str = "market_agents/urls.json") -> List[str]:
    try:
        file_path = Path(input_file).resolve()
        logger.info(f"Loading URLs from: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            urls = data.get('urls', [])
            logger.info(f"Successfully loaded {len(urls)} URLs")
            return urls
    except FileNotFoundError:
        logger.error(f"URLs file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {file_path}")
        raise

async def main():
    try:
        config = load_config()
        agent = WebSearchAgent(config)
        
        # Define your search query
        query = "Impact of AI on financial markets"
        logger.info(f"Starting search with query: {query}")
        
        await agent.run_search(query)
        
        output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agent.save_results(output_file)
        
        successful = sum(1 for r in agent.results if r and r.status == "success")
        failed = len(agent.results) - successful if agent.results else 0
        
        logger.info(f"""
Search completed:
- Total items processed: {len(agent.results) if agent.results else 0}
- Successful: {successful}
- Failed: {failed}
- Results saved to: {output_file}
        """)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())