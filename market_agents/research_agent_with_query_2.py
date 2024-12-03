import asyncio
import json
import logging
import os
import traceback
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
import argparse 
import re
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.logger_utils import *
from market_agents.insert_simulation_data import SimulationDataInserter
from typing import Any, Dict, List, Optional, Union, Tuple 

from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    client: str
    model: str
    max_tokens: int = 10000
    temperature: float = 0.7
    use_cache: bool = True
    system_prompt: str = Field(default="You are a helpful AI that generates relevant search queries.")
    prompt_template: str = Field(default="Generate 3-5 search queries related to this topic: {query}")

    def dict(self, *args, **kwargs):
        return {
            "client": self.client,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_cache": self.use_cache,
            "system_prompt": self.system_prompt,
            "prompt_template": self.prompt_template
        }



# Configure logging
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
    query: str = "default search query"
    max_concurrent_requests: int = 50
    rate_limit: float = 0.1
    content_max_length: int = 1000000
    request_timeout: int = 30
    urls_per_query: int = 5
    use_ai_summary: bool = True 
    methods: List[str] = ["selenium", "playwright", "beautifulsoup", "newspaper3k", "scrapy", 
                         "requests_html", "mechanicalsoup", "httpx"]
    default_method: str = "newspaper3k"
    headers: Dict[str, str] = {  # Add this field
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    llm_configs: Dict[str, Dict[str, Any]] = {  # Changed from List to Dict
        "search_query_generation": {
            "client": "openai",
            "model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
            "use_cache": True,
            "system_prompt": "You are an expert market research analyst specializing in technology stocks.",
            "prompt_template": "Generate 5 specific search queries related to: {query}"
        },
        "content_analysis": {
            "client": "openai",
            "model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.5,
            "use_cache": True,
            "system_prompt": "You are an expert financial analyst.",
            "prompt_template": "{prompt}"
        }
    }

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
class SearchManager:
    def __init__(self, ai_utils, config: WebSearchConfig, prompts: Dict):
        self.ai_utils = ai_utils
        self.config = config
        self.prompts = prompts
        self.last_request_time = 0
        self.request_delay = 5
        self.max_retries = 3
        self.headers = config.headers
        self.search_params = {
            'num': self.config.urls_per_query,
            'stop': self.config.urls_per_query,
            'pause': 2.0,
            'user_agent': self.headers['User-Agent']
        }
        self.query_url_mapping = {}


    async def generate_search_queries(self, base_query: str) -> List[str]:
        try:
            # Create LLMConfig instance from the dictionary
            llm_config_dict = self.config.llm_configs["search_query_generation"]
            llm_config = LLMConfig(**llm_config_dict)
            
            # Format the prompt using the template from config
            prompt = self.prompts["search_query_generation"].format(query=base_query)
            
            logger.info(f"Using prompt template:\n{prompt}")
            
            # Create prompt context using the imported LLMPromptContext
            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=llm_config_dict["system_prompt"],
                new_message=prompt,
                llm_config=llm_config.dict(),  # Convert LLMConfig to dict
                use_history=False
            )

            responses = await self.ai_utils.run_parallel_ai_completion([context])
            
            if responses and len(responses) > 0:
                # Get response content using str_content or json_object
                if responses[0].json_object:
                    response_text = json.dumps(responses[0].json_object.object)
                else:
                    response_text = responses[0].str_content

                if not response_text:
                    logger.error("No response content found")
                    return [base_query]

                logger.info(f"AI Response:\n{response_text}")
                
                # Extract queries from the response
                queries = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line and not any(line.startswith(x) for x in ['Query:', 'Please', 'Format', 'Make']):
                        cleaned_query = re.sub(r'^\d+\.\s*', '', line)
                        if cleaned_query:
                            queries.append(cleaned_query)
                
                # Add the original query if it's not already included
                all_queries = [base_query] + [q for q in queries if q != base_query]
                
                logger.info("\n=== Generated Search Queries ===")
                logger.info(f"Original Query: {base_query}")
                logger.info(f"Additional Queries Generated: {len(queries)}")
                logger.info("\nAll Queries:")
                for i, query in enumerate(all_queries, 1):
                    logger.info(f"{i}. {query}")
                
                return all_queries
                
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [base_query]

    
    def get_urls_for_query(self, query: str, num_results: int = 5) -> List[str]:
        """Get URLs from Google search with retry logic"""
        for attempt in range(self.max_retries):
            try:
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                urls = list(search(
                    query,
                    num=num_results,
                    stop=num_results,
                    pause=self.search_params['pause']
                ))
                
                self.last_request_time = time.time()
                
                if urls:
                    logger.info(f"\n=== URLs Found ===")
                    logger.info(f"Query: {query}")
                    for i, url in enumerate(urls, 1):
                        logger.info(f"URL {i}: {url}")
                    logger.info("================")
                    
                    # Store query-URL mapping
                    for url in urls:
                        self.query_url_mapping[url] = query
                    return urls
                    
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.request_delay * (attempt + 1)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    
        logger.error(f"All search attempts failed for query: {query}")
        return []


class ContentExtractor:
    def __init__(self, config):
        self.config = config
        self.timeout = 30 
    def extract_tables(soup: BeautifulSoup) -> List[Dict]:
        """Extract table data from HTML content"""
        tables = []
        try:
            for table in soup.find_all('table'):
                headers = []
                rows = []
                
                # Extract headers
                for th in table.find_all('th'):
                    headers.append(th.get_text(strip=True))
                
                # If no headers found, try first row
                first_row = table.find('tr')
                if not headers and first_row:
                    headers = [td.get_text(strip=True) for td in first_row.find_all('td')]
                
                # Extract rows
                for tr in table.find_all('tr')[1:]:
                    row = [td.get_text(strip=True) for td in tr.find_all('td')]
                    if row:
                        rows.append(row)
                
                if headers and rows:
                    tables.append({
                        'headers': headers,
                        'rows': rows
                    })
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
        return tables

    @staticmethod
    def extract_charts(soup: BeautifulSoup) -> List[Dict]:
        """Extract chart data and metadata"""
        charts = []
        try:
            chart_elements = soup.find_all(['div', 'figure'], class_=lambda x: x and any(
                term in str(x).lower() for term in ['chart', 'graph', 'plot', 'visualization']
            ))
            
            for element in chart_elements:
                chart_data = {
                    'type': 'unknown',
                    'title': '',
                    'description': ''
                }
                
                # Try to determine chart type and data
                if element.get('data-highcharts-chart'):
                    chart_data['type'] = 'highcharts'
                elif element.find(class_=lambda x: x and 'tradingview' in str(x).lower()):
                    chart_data['type'] = 'tradingview'
                
                title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'figcaption', 'title'])
                if title_elem:
                    chart_data['title'] = title_elem.get_text(strip=True)
                
                charts.append(chart_data)
        except Exception as e:
            logger.error(f"Error extracting charts: {str(e)}")
        return charts

    @staticmethod
    @staticmethod
    def clean_content(content: str) -> Dict[str, Any]:
        """Clean content and identify its type"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract tables and charts first
            tables = ContentExtractor.extract_tables(soup)
            charts = ContentExtractor.extract_charts(soup)
            
            # Clean text content
            for tag in soup.find_all(['script', 'style', 'meta', 'noscript']):
                tag.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            return {
                'text': text.strip(),
                'tables': tables,
                'charts': charts,
                'has_data': bool(tables or charts)
            }
        except Exception as e:
            logger.error(f"Error cleaning content: {str(e)}")
            return {
                'text': str(content),  # Convert content to string
                'tables': [],
                'charts': [],
                'has_data': False
            }
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
                return title, ContentExtractor.clean_content(content)
            except Exception as e:
                logger.error(f"Playwright extraction failed: {str(e)}")
                await browser.close()
                return "", ""

    @staticmethod
    async def extract_with_selenium(url: str, timeout: int = 60) -> tuple[str, Dict[str, Any]]:  # increased timeout
        """Modified to be async and return proper types"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")  # Add this
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            title = driver.title
            content = ContentExtractor.clean_content(driver.page_source)
            return title, content
        except Exception as e:
            logger.error(f"Selenium extraction failed: {str(e)}")
            return "", {"text": "", "tables": [], "charts": [], "has_data": False}
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    logger.error(f"Error closing Selenium driver: {str(e)}")

    @staticmethod
    async def extract_with_beautifulsoup(url: str, headers: Dict[str, str]) -> tuple[str, str]:
        try:
            response = httpx.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            content = ContentExtractor.clean_content(str(soup))
            return title, content
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return "", ""

    @staticmethod
    async def extract_with_scrapy(url: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                selector = ParselSelector(response.text)
                title = selector.css('h1::text').get() or selector.css('title::text').get()
                content = ContentExtractor.clean_content(response.text)
                return title, content
        except Exception as e:
            logger.error(f"Scrapy extraction failed: {str(e)}")
            return "", ""

    @staticmethod
    async def extract_with_requests_html(url: str) -> tuple[str, str]:
        try:
            session = AsyncHTMLSession()
            response = await session.get(url)
            await response.html.arender()
            title = response.html.find('title', first=True)
            content = ContentExtractor.clean_content(response.html.html)
            return title.text if title else url, content
        except Exception as e:
            logger.error(f"Requests-HTML extraction failed: {str(e)}")
            return "", ""

    @staticmethod
    async def extract_with_mechanicalsoup(url: str) -> tuple[str, str]:
        try:
            browser = mechanicalsoup.StatefulBrowser()
            browser.set_user_agent('Mozilla/5.0')
            page = browser.get(url)
            title = page.soup.find('title')
            content = ContentExtractor.clean_content(str(page.soup))
            return title.text if title else url, content
        except Exception as e:
            logger.error(f"MechanicalSoup extraction failed: {str(e)}")
            return "", ""

    @staticmethod
    async def extract_with_httpx(url: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('title')
                content = ContentExtractor.clean_content(response.text)
                return title.text if title else url, content
        except Exception as e:
            logger.error(f"HTTPX extraction failed: {str(e)}")
            return "", ""

class WebSearchAgent:
    def __init__(self, config: WebSearchConfig, prompts: Dict):
        self.config = config
        self.prompts = prompts
        self.results: List[WebSearchResult] = []
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.content_extractor = ContentExtractor(config)
        self.headers = config.headers
        
        # Initialize AI utilities
        oai_request_limits = RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=150000
        )
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None
        )
        
        # Initialize search manager
        self.search_manager = SearchManager(self.ai_utils, config, prompts)

    async def run(self):
        """Main method to run the search agent"""
        try:
            # Get the query from config
            base_query = self.config.query
            logger.info(f"Starting search with query: {base_query}")

            # Generate and process search queries
            await self.process_search_query(base_query)

            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/web_search")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"results_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump([result.dict() for result in self.results], f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            return self.results

        except Exception as e:
            logger.error(f"Error in search agent: {str(e)}")
            raise

    async def generate_ai_summary(self, content: Dict[str, Any], url: str) -> Dict[str, Any]:
        try:
            # Get the content analysis config
            llm_config_dict = self.config.llm_configs["content_analysis"]
            
            # Create LLMConfig instance
            llm_config = LLMConfig(**llm_config_dict)

            base_prompt = f"""Analyze the following content and provide:
            1. A concise summary (2-3 sentences)
            2. Key points (3-5 bullet points)
            3. Market impact assessment
            4. Trading implications
            """

            # Create different prompts based on content type
            if content['has_data']:
                prompt = f"""{base_prompt}
                Text Content: {content['text'][:4000]}
                Table Data: {json.dumps(content['tables'], indent=2) if content['tables'] else 'No tables'}
                Chart Information: {json.dumps(content['charts'], indent=2) if content['charts'] else 'No charts'}
                
                Please include data-specific analysis:
                5. Data metrics analysis
                6. Technical indicators (if present)
                7. Market data insights
                """
            else:
                prompt = f"""{base_prompt}
                Content: {content['text'][:4000]}
                """

            # Format the prompt using the template from config
            formatted_prompt = llm_config_dict["prompt_template"].format(prompt=prompt)

            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=llm_config_dict["system_prompt"],
                new_message=formatted_prompt,
                llm_config=llm_config.dict(),  # Convert LLMConfig to dict
                use_history=False
            )

            try:
                responses = await self.ai_utils.run_parallel_ai_completion([context])
                
                if not responses or len(responses) == 0:
                    logger.warning(f"No response received for {url}")
                    return self._get_default_summary(content['has_data'])

                response = responses[0]
                
                # Get response content using str_content or json_object
                response_text = None
                if response.json_object:
                    try:
                        response_text = json.dumps(response.json_object.object)
                    except (AttributeError, TypeError) as e:
                        logger.error(f"Error parsing JSON object for {url}: {str(e)}")
                
                if not response_text and response.str_content:
                    response_text = response.str_content

                if not response_text:
                    logger.warning(f"No valid response content for {url}")
                    return self._get_default_summary(content['has_data'])

                # Parse the response into sections
                sections = [s.strip() for s in response_text.split('\n\n') if s.strip()]
                
                summary_dict = {
                    "summary": sections[0] if sections else "Summary not available",
                    "key_points": [point.strip('- ') for point in (sections[1].split('\n') if len(sections) > 1 else [])],
                    "market_impact": sections[2] if len(sections) > 2 else "Market impact analysis not available",
                    "trading_implications": sections[3] if len(sections) > 3 else "Trading implications not available"
                }

                # Add data-specific analysis if content contains tables/charts
                if content['has_data'] and len(sections) > 4:
                    summary_dict.update({
                        "data_metrics": sections[4],
                        "technical_indicators": sections[5] if len(sections) > 5 else "Technical analysis not available",
                        "market_data_insights": sections[6] if len(sections) > 6 else "Market data insights not available"
                    })
                    
                    # Add structured data analysis
                    if content['tables']:
                        table_insights = []
                        for idx, table in enumerate(content['tables']):
                            table_insights.append({
                                "table_number": idx + 1,
                                "headers": table.get('headers', []),
                                "key_metrics": [row for row in table.get('rows', [])[:3]],
                                "analysis": f"Table {idx + 1} analysis from sections[4]" if len(sections) > 4 else "Analysis not available"
                            })
                        summary_dict["table_analysis"] = table_insights

                    if content['charts']:
                        chart_insights = []
                        for idx, chart in enumerate(content['charts']):
                            chart_insights.append({
                                "chart_number": idx + 1,
                                "type": chart.get('type', 'unknown'),
                                "title": chart.get('title', ''),
                                "analysis": f"Chart {idx + 1} analysis from sections[5]" if len(sections) > 5 else "Analysis not available"
                            })
                        summary_dict["chart_analysis"] = chart_insights

                return summary_dict

            except Exception as e:
                logger.error(f"Error processing AI response for {url}: {str(e)}")
                return self._get_default_summary(content['has_data'])

        except Exception as e:
            logger.error(f"Error in AI summary generation for {url}: {str(e)}")
            return self._get_default_summary(content['has_data'])

    def _get_default_summary(self, has_data: bool = False) -> Dict[str, Any]:
        """Get default summary structure based on content type"""
        summary = {
            "summary": "Summary generation failed",
            "key_points": ["Error in key points extraction"],
            "market_impact": "Impact analysis failed",
            "trading_implications": "Trading analysis failed"
        }
        
        if has_data:
            summary.update({
                "data_metrics": "Data metrics analysis failed",
                "technical_indicators": "Technical analysis failed",
                "market_data_insights": "Market data insights failed",
                "table_analysis": [],
                "chart_analysis": []
            })
        
        return summary
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[WebSearchResult]:
        """Fetch and process a single URL"""
        try:
            async with self.semaphore:
                original_query = self.search_manager.query_url_mapping.get(url, "Unknown query")
                logger.info(f"\n=== Processing URL ===\nURL: {url}\nOriginal Query: {original_query}")
                
                for method in self.config.methods:
                    try:
                        logger.info(f"Trying method {method} for {url}")
                        
                        content = None
                        title = None

                        if method == "selenium":
                            title, content = await self.content_extractor.extract_with_selenium(url)
                        elif method == "playwright":
                            title, content = await self.content_extractor.extract_with_playwright(url)
                        # ... other extraction methods ...

                        if content and isinstance(content, dict):  # Verify content is dictionary
                            logger.info(f"Successfully extracted content from {url} using {method}")
                            logger.info(f"Content type: {'Contains tables/charts' if content['has_data'] else 'Text only'}")
                            
                            summary = await self.generate_ai_summary(content, url) if self.config.use_ai_summary else {}
                            
                            return WebSearchResult(
                                url=url,
                                title=title or url,
                                content=content['text'][:self.config.content_max_length],
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
                return None

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    async def process_search_query(self, query: str) -> None:
        """Process a search query by generating multiple queries and fetching URLs"""
        try:
            # Generate multiple search queries using AI
            search_queries = await self.search_manager.generate_search_queries(query)
            
            logger.info(f"""
=== Search Process Starting ===
Original Query: {query}
Total Queries Generated: {len(search_queries)}
""")

            # Process each search query separately
            all_results = []
            
            for idx, search_query in enumerate(search_queries, 1):
                logger.info(f"""
=== Processing Query {idx}/{len(search_queries)} ===
Query: {search_query}
""")
                
                # Get URLs for this specific query
                urls = self.search_manager.get_urls_for_query(
                    search_query, 
                    num_results=self.config.urls_per_query
                )
                
                logger.info(f"""
URLs found for query "{search_query}":
{chr(10).join(f'- {url}' for url in urls)}
""")
                
                # Store the query-URL mapping
                for url in urls:
                    self.search_manager.query_url_mapping[url] = search_query
                
                # Process URLs for this query
                async with aiohttp.ClientSession() as session:
                    tasks = [self.fetch_url(session, url) for url in urls]
                    query_results = await asyncio.gather(*tasks)
                    valid_results = [r for r in query_results if r is not None]
                    all_results.extend(valid_results)
                    
                    logger.info(f"""
Query {idx} Results Summary:
- URLs processed: {len(urls)}
- Successful extractions: {len(valid_results)}
- Failed extractions: {len(urls) - len(valid_results)}
""")

            # Store all results
            self.results = all_results
            
            # Print final summary
            logger.info(f"""
=== Final Search Summary ===
Total Queries Processed: {len(search_queries)}
Total URLs Processed: {sum(len(self.search_manager.get_urls_for_query(q)) for q in search_queries)}
Total Successful Extractions: {len(self.results)}

Results by Query:
{chr(10).join(f'- "{q}": {len([r for r in self.results if self.search_manager.query_url_mapping.get(r.url) == q])} results' for q in search_queries)}
""")

        except Exception as e:
            logger.error(f"Error processing search query: {str(e)}")
            raise

    async def process_urls(self, urls: List[str]) -> None:
        """Process a list of URLs"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            self.results = [r for r in results if r is not None]

    def save_results(self, output_file: str):
        """Save results to file and attempt database insertion"""
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
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Try to save to database
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


def load_prompts(prompt_path: Path = Path("./market_agents/web_search_prompt.yaml")) -> dict:
    """Load prompts from yaml file"""
    try:
        with open(prompt_path, 'r') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        raise



def load_config(config_path: str = "market_agents/web_search_config.yaml", 
               prompt_path: str = "market_agents/web_search_prompt.yaml") -> tuple[WebSearchConfig, Dict]:
    try:
        # Load main config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Debug print
        print("Loaded config data:", json.dumps(config_data, indent=2))
            
        # Load prompts
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
            
        # Create WebSearchConfig instance
        config = WebSearchConfig(**config_data)
        
        return config, prompts
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
    
async def main():
    """Main execution function"""
    try:
        # Load both config and prompts
        config, prompts = load_config()
        
        # Initialize agent with both config and prompts
        agent = WebSearchAgent(config, prompts)
        
        logger.info(f"Starting search with query: {config.query}")
        await agent.process_search_query(config.query)
        
        output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agent.save_results(output_file)
        
        successful = sum(1 for r in agent.results if r and r.status == "success")
        failed = len(agent.results) - successful if agent.results else 0
        
        logger.info(f"""
Search completed:
- Query: {config.query}
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