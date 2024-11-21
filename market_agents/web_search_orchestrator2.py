import asyncio
import json
import logging
import os
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import time
from playwright.async_api import async_playwright
import aiohttp
import yaml
from bs4 import BeautifulSoup
from colorama import Fore, Style
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from scrapy import Selector
import requests
import scrapy
from newspaper import Article
from requests_html import AsyncHTMLSession
import httpx
from parsel import Selector as ParselSelector
import mechanicalsoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from market_agents.agents.personas.persona import Persona, generate_persona
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import (ParallelAIUtilities, RequestLimits)
from market_agents.logger_utils import *


logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(logging.NullHandler())

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('web_search.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)
logger.propagate = False

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
    """Configuration for web search"""
    max_concurrent_requests: int = 50
    rate_limit: float = 0.1
    content_max_length: int = 1000000
    request_timeout: int = 30
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
    methods: List[str] = [
        "selenium",
        "playwright", 
        "beautifulsoup",
        "newspaper3k",     # For news article extraction
        "scrapy",         
        "requests_html",  
        "mechanicalsoup", 
        "httpx"         
    ]
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
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            title = driver.title
            content = driver.page_source
            
            driver.quit()
            return title, content
        except Exception as e:
            logger.error(f"Selenium extraction failed: {str(e)}")
            driver.quit()
            return "", ""

    @staticmethod
    def extract_with_beautifulsoup(url: str, headers: Dict[str, str]) -> tuple[str, str]:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_parts = []
            
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                if element.name.startswith('h'):
                    content_parts.append(f"\n{'#' * int(element.name[1])} {element.get_text().strip()}\n")
                elif element.name == 'p':
                    content_parts.append(f"{element.get_text().strip()}\n\n")
                elif element.name == 'li':
                    content_parts.append(f"- {element.get_text().strip()}\n")
            
            content = "".join(content_parts)
            title = soup.title.string if soup.title else ""
            
            return title, content
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return "", ""
    @staticmethod
    async def extract_with_newspaper3k(url: str) -> tuple[str, str]:
        """Extract content using newspaper3k - specialized for news articles"""
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text

    @staticmethod
    async def extract_with_scrapy(url: str) -> tuple[str, str]:
        """Extract content using scrapy"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            selector = ParselSelector(response.text)
            
            # Adjust selectors based on your target websites
            title = selector.css('h1::text').get() or selector.css('title::text').get()
            content = ' '.join(selector.css('p::text').getall())
            return title, content

    @staticmethod
    async def extract_with_requests_html(url: str) -> tuple[str, str]:
        """Extract content using requests-html with JavaScript support"""
        session = AsyncHTMLSession()
        response = await session.get(url)
        await response.html.arender()  # Renders JavaScript
        
        title = response.html.find('title', first=True)
        content = ' '.join([p.text for p in response.html.find('p')])
        return title.text if title else url, content

    @staticmethod
    async def extract_with_mechanicalsoup(url: str) -> tuple[str, str]:
        """Extract content using MechanicalSoup"""
        browser = mechanicalsoup.StatefulBrowser()
        browser.set_user_agent('Mozilla/5.0')
        
        page = browser.get(url)
        title = page.soup.find('title')
        content = ' '.join([p.text for p in page.soup.find_all('p')])
        return title.text if title else url, content

    @staticmethod
    async def extract_with_httpx(url: str) -> tuple[str, str]:
        """Extract content using httpx with retry logic"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.find('title')
            content = ' '.join([p.text for p in soup.find_all('p')])
            return title.text if title else url, content

class WebSearchAgent:
    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.results: List[WebSearchResult] = []
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.content_extractor = ContentExtractor()
        
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=150000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None
        )
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> WebSearchResult:
        """Fetch and process a single URL using the default method"""
        try:
            content = ""
            title = ""
            method = self.config.default_method
            
            logger.info(f"Using method {method} for {url}")
            
            try:
                if method == "selenium":
                    title, content = self.content_extractor.extract_with_selenium(url)
                elif method == "playwright":
                    title, content = await self.content_extractor.extract_with_playwright(url)
                elif method == "beautifulsoup":
                    title, content = self.content_extractor.extract_with_beautifulsoup(url, self.headers)
                elif method == "newspaper3k":
                    title, content = await self.content_extractor.extract_with_newspaper3k(url)
                elif method == "scrapy":
                    title, content = await self.content_extractor.extract_with_scrapy(url)
                elif method == "requests_html":
                    title, content = await self.content_extractor.extract_with_requests_html(url)
                elif method == "mechanicalsoup":
                    title, content = await self.content_extractor.extract_with_mechanicalsoup(url)
                elif method == "httpx":
                    title, content = await self.content_extractor.extract_with_httpx(url)
                
                if content and not "errors.edgesuite.net" in content:
                    logger.info(f"Successfully extracted content using {method}")
                    summary = await self.generate_ai_summary(content, url)
                    
                    return WebSearchResult(
                        url=url,
                        title=title,
                        content=content[:self.config.content_max_length],
                        timestamp=datetime.now(),
                        status="success",
                        summary=summary,
                        agent_id=str(uuid.uuid4()),
                        extraction_method=method
                    )
                
            except Exception as e:
                logger.error(f"{method} failed for {url}: {str(e)}")
                return WebSearchResult(
                    url=url,
                    title="Error",
                    content=f"Failed to extract with {method}: {str(e)}",
                    timestamp=datetime.now(),
                    status="error: extraction failed",
                    summary={},
                    agent_id=str(uuid.uuid4()),
                    extraction_method=f"{method}_failed"
                )
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return WebSearchResult(
                url=url,
                title="",
                content="",
                timestamp=datetime.now(),
                status=f"error: {str(e)}",
                summary={},
                agent_id=str(uuid.uuid4()),
                extraction_method="error"
            )

    # Modify the save_results method to include extraction method in the log output
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
                    logger.info(f"""
    URL: {result.url}
    TITLE: {result.title}
    EXTRACTION METHOD: {result.extraction_method}
    SUMMARY:
    {json.dumps(result.summary, indent=2)}
    -------------------
                    """)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
    async def generate_ai_summary(self, content: str, url: str) -> dict:
        """Generate a comprehensive AI summary of the article content"""
        try:
            # 1. Generate detailed summary
            basic_prompt = f"""Analyze this article and provide a comprehensive summary:

            Article Content:
            {content}

            Please provide a detailed summary that captures all key information, main points, and important details from the article.
            """

            basic_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=basic_prompt,
                prompt=basic_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )
            
            # 2. Extract key points
            key_points_prompt = f"""Extract all important points from this article:

            Article Content:
            {content}

            Please list all key points, including:
            - Market movements and statistics
            - Important announcements
            - Policy changes
            - Market reactions
            - Notable quotes
            - Significant data points
            """

            key_points_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=key_points_prompt,
                prompt=key_points_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )

            market_impact_prompt = f"""You are a senior market analyst. Analyze the market impact of this article and provide specific insights:

            Article Content:
            {content}

            Focus your analysis on:
            1. IMMEDIATE MARKET REACTIONS:
            - Specific market movements mentioned
            - Price changes and percentage moves
            - Volume and trading activity

            2. SECTOR-SPECIFIC IMPACTS:
            - Which sectors are most affected?
            - Winners and losers from these developments
            - Potential ripple effects across industries

            3. MACROECONOMIC IMPLICATIONS:
            - Impact on interest rates and monetary policy
            - Effects on inflation expectations
            - Currency market implications
            - Global market considerations

            4. SENTIMENT ANALYSIS:
            - Changes in market sentiment
            - Institutional investor reactions
            - Retail investor implications

            Please provide a detailed, structured analysis with specific numbers and data points where available.
            Avoid generic statements and focus on actionable insights.
            """

            trading_prompt = f"""You are a professional trading strategist. Based on this article, provide specific trading recommendations:

            Article Content:
            {content}

            Please analyze and provide:

            1. SPECIFIC TRADING OPPORTUNITIES:
            - Individual stocks/sectors to watch
            - Entry and exit points if mentioned
            - Risk levels and potential targets
            - Time horizons for trades

            2. RISK ASSESSMENT:
            - Key risk factors to monitor
            - Potential hedging strategies
            - Stop-loss considerations
            - Volatility expectations

            3. PORTFOLIO ADJUSTMENTS:
            - Recommended sector rotations
            - Asset allocation changes
            - Position sizing suggestions
            - Diversification considerations

            4. TIMING CONSIDERATIONS:
            - Short-term trading opportunities (1-5 days)
            - Medium-term positioning (1-3 months)
            - Long-term strategic implications
            - Key dates and events to watch

            5. SPECIFIC ACTIONABLE RECOMMENDATIONS:
            - List specific trading ideas
            - Provide clear rationale for each
            - Include relevant price levels
            - Suggest risk management approaches

            Format your response with clear sections and bullet points where appropriate.
            Focus on practical, actionable insights backed by the article's content.
            """

            market_impact_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=market_impact_prompt,
                prompt=market_impact_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )

            trading_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=trading_prompt,
                prompt=trading_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )
            all_contexts = [basic_context, key_points_context, market_impact_context, trading_context]
            results = await self.ai_utils.run_parallel_ai_completion(all_contexts)

            # Process results with better error handling
            basic_summary = ""
            key_points = []
            market_impact = ""
            trading_implications = ""

            if results and len(results) == 4:
                if results[0] and hasattr(results[0], 'choices'):
                    basic_summary = results[0].choices[0].message.content.strip()
                
                if results[1] and hasattr(results[1], 'choices'):
                    key_points_text = results[1].choices[0].message.content.strip()
                    key_points = [point.strip() for point in key_points_text.split('\n') if point.strip()]
                
                if results[2] and hasattr(results[2], 'choices'):
                    market_impact = results[2].choices[0].message.content.strip()
                
                if results[3] and hasattr(results[3], 'choices'):
                    trading_implications = results[3].choices[0].message.content.strip()

            # Enhanced fallback mechanism for market impact and trading implications
            if not market_impact or market_impact == "Impact analysis unavailable":
                try:
                    # Extract market-related sentences from content
                    sentences = content.split('.')
                    market_related = [s for s in sentences if any(word in s.lower() for word in 
                        ['market', 'stock', 'index', 'trading', 'price', 'investor', 'rate', 'fed'])]
                    market_impact = "\n".join(market_related) if market_related else "Impact analysis unavailable"
                except:
                    market_impact = "Impact analysis unavailable"

            if not trading_implications or trading_implications == "Trading analysis unavailable":
                try:
                    # Extract trading-related sentences from content
                    sentences = content.split('.')
                    trading_related = [s for s in sentences if any(word in s.lower() for word in 
                        ['trade', 'buy', 'sell', 'position', 'strategy', 'opportunity', 'risk'])]
                    trading_implications = "\n".join(trading_related) if trading_related else "Trading analysis unavailable"
                except:
                    trading_implications = "Trading analysis unavailable"

            summary_dict = {
                "summary": basic_summary or content,
                "key_points": key_points or ["Key point extraction failed"],
                "market_impact": market_impact,
                "trading_implications": trading_implications
            }

            logger.info(f"""
    === Summary Generated for {url} ===
    Summary: 
    {summary_dict['summary']}

    Key Points:
    {json.dumps(summary_dict['key_points'], indent=2)}

    Market Impact:
    {summary_dict['market_impact']}

    Trading Implications:
    {summary_dict['trading_implications']}
    =====================================
            """)
            
            return summary_dict

        except Exception as e:
            logger.error(f"Error in AI summary generation for {url}: {str(e)}")
            # Enhanced fallback mechanism
            try:
                paragraphs = content.split('\n\n')
                market_sentences = [s for s in content.split('.') if any(word in s.lower() for word in 
                    ['market', 'stock', 'index', 'trading', 'price', 'investor', 'rate', 'fed'])]
                trading_sentences = [s for s in content.split('.') if any(word in s.lower() for word in 
                    ['trade', 'buy', 'sell', 'position', 'strategy', 'opportunity', 'risk'])]
                
                return {
                    "summary": paragraphs[0] if paragraphs else content,
                    "key_points": [p.strip() for p in paragraphs[1:5] if p.strip()] if len(paragraphs) > 1 
                        else ["No key points available"],
                    "market_impact": "\n".join(market_sentences) if market_sentences 
                        else "Impact analysis unavailable due to processing error",
                    "trading_implications": "\n".join(trading_sentences) if trading_sentences 
                        else "Trading analysis unavailable due to processing error"
                }
            except:
                return {
                    "summary": content if content else "Summary unavailable",
                    "key_points": ["Content extraction failed"],
                    "market_impact": "Impact analysis unavailable",
                    "trading_implications": "Trading analysis unavailable"
                }
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
            # Save complete result without any exclusions
            result_data = result.model_dump(exclude_none=True)
            results_dict.append(result_data)
            
            if result.summary:
                # Format the complete summary with proper indentation
                formatted_summary = json.dumps(result.summary, indent=2, ensure_ascii=False)
                
                # Format key points with full content
                key_points = result.summary.get('key_points', [])
                formatted_key_points = '\n'.join([f"- {point}" for point in key_points])
                
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
    
    # Save complete results with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)

def load_config(config_path: Path = Path("./market_agents/web_search_config.yaml")) -> WebSearchConfig:
    try:
        with open(config_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
        yaml_data['max_concurrent_requests'] = int(os.getenv('WEB_SEARCH_MAX_REQUESTS', yaml_data.get('max_concurrent_requests', 50)))
        yaml_data['rate_limit'] = float(os.getenv('WEB_SEARCH_RATE_LIMIT', yaml_data.get('rate_limit', 0.1)))
        yaml_data['content_max_length'] = int(os.getenv('WEB_SEARCH_CONTENT_LENGTH', yaml_data.get('content_max_length', 1000)))
        yaml_data['request_timeout'] = int(os.getenv('WEB_SEARCH_TIMEOUT', yaml_data.get('request_timeout', 30)))
        
        use_ai_summary_env = os.getenv('WEB_SEARCH_USE_AI_SUMMARY')
        if use_ai_summary_env is not None:
            yaml_data['use_ai_summary'] = use_ai_summary_env.lower() == 'true'
        
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
        urls = load_urls("market_agents/urls.json")
        logger.info(f"Loaded {len(urls)} URLs")
        
        agent = WebSearchAgent(config)
        start_time = time.time()
        
        await agent.process_urls(urls)
        
        output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agent.save_results(output_file)
        
        successful = sum(1 for r in agent.results if r.status == "success")
        failed = len(agent.results) - successful
        duration = time.time() - start_time
        
        log_section(logger, "SEARCH SUMMARY")
        logger.info(f"""
        Search completed:
        - Total URLs: {len(urls)}
        - Successful: {successful}
        - Failed: {failed}
        - Duration: {duration:.2f} seconds
        - Results saved to: {output_file}
        """)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())