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
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits, LLMOutput
from market_agents.inference.message_models import LLMPromptContext, LLMConfig 
from market_agents.logger_utils import *
from market_agents.insert_simulation_data import SimulationDataInserter
from typing import Any, Dict, List, Optional, Union, Tuple, Literal, Self
from pydantic import BaseModel, Field, model_validator
import numpy as np
from typing import List, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass




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



@dataclass
class ContentAnalysisConfig:
    """Configuration for content analysis"""
    client: str
    model: str
    max_tokens: int
    temperature: float
    response_format: str
    use_cache: bool
    system_prompt: str
    prompt_template: str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ContentAnalysisConfig':
        return cls(**config_dict)

class ContentProcessor:
    """Process and structure different types of content"""
    
    def __init__(self):
        self.max_content_length = 4000
        self.market_indicators = {
            'bullish': ['increase', 'rise', 'gain', 'up', 'higher', 'bull', 'growth'],
            'bearish': ['decrease', 'fall', 'loss', 'down', 'lower', 'bear', 'decline']
        }
    def analyze_market_data(self, tables: List[dict], charts: List[dict]) -> Dict[str, Any]:
        """Analyze market data from tables and charts"""
        analysis = {
            'price_movements': [],
            'volume_analysis': [],
            'trend_indicators': [],
            'key_levels': [],
            'market_sentiment': None,
            'risk_metrics': {},
            'statistical_analysis': {}
        }

        try:
            # Analyze tables
            if tables:
                table_analysis = self._analyze_tables(tables)
                analysis.update(table_analysis)

            # Analyze charts
            if charts:
                chart_analysis = self._analyze_charts(charts)
                analysis.update(chart_analysis)

            # Calculate overall market sentiment
            analysis['market_sentiment'] = self._calculate_sentiment(analysis)

            return analysis
        except Exception as e:
            logger.error(f"Error in market data analysis: {str(e)}")
            return analysis
    def _analyze_tables(self, tables: List[dict]) -> Dict[str, Any]:
        """Analyze numerical data from tables"""
        analysis = {
            'numerical_insights': [],
            'price_changes': [],
            'percentage_movements': [],
            'key_statistics': {}
        }

        try:
            for table in tables:
                # Convert table to DataFrame for analysis
                df = pd.DataFrame(table)
                
                # Look for price-related columns
                price_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['price', 'value', 'close', 'open']
                )]
                
                # Look for percentage columns
                pct_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['%', 'percent', 'change']
                )]

                if price_cols:
                    for col in price_cols:
                        try:
                            numeric_data = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_data.empty:
                                analysis['key_statistics'][col] = {
                                    'mean': numeric_data.mean(),
                                    'max': numeric_data.max(),
                                    'min': numeric_data.min(),
                                    'std': numeric_data.std()
                                }
                                
                                # Detect significant price movements
                                changes = numeric_data.pct_change()
                                significant_changes = changes[abs(changes) > 0.01]  # 1% threshold
                                if not significant_changes.empty:
                                    analysis['price_changes'].extend([
                                        {
                                            'column': col,
                                            'change': float(change),
                                            'index': idx
                                        } for idx, change in significant_changes.items()
                                    ])
                        except Exception as e:
                            logger.warning(f"Error analyzing price column {col}: {str(e)}")

                if pct_cols:
                    for col in pct_cols:
                        try:
                            pct_data = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100
                            if not pct_data.empty:
                                analysis['percentage_movements'].extend([
                                    {
                                        'column': col,
                                        'value': float(val),
                                        'index': idx
                                    } for idx, val in pct_data.items() if abs(val) > 0.01
                                ])
                        except Exception as e:
                            logger.warning(f"Error analyzing percentage column {col}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in table analysis: {str(e)}")

        return analysis
    def _analyze_charts(self, charts: List[dict]) -> Dict[str, Any]:
        """Analyze chart data for market trends"""
        analysis = {
            'trend_analysis': [],
            'support_resistance': [],
            'pattern_recognition': [],
            'technical_indicators': {}
        }

        try:
            for chart in charts:
                chart_type = chart.get('type', '').lower()
                chart_data = chart.get('data', {})

                if chart_type in ['line', 'candlestick', 'ohlc']:
                    # Analyze price action
                    if 'values' in chart_data:
                        values = pd.Series(chart_data['values'])
                        analysis['technical_indicators'].update({
                            'sma_20': float(values.rolling(20).mean().iloc[-1]) if len(values) >= 20 else None,
                            'sma_50': float(values.rolling(50).mean().iloc[-1]) if len(values) >= 50 else None,
                            'volatility': float(values.std()) if len(values) >= 2 else None
                        })

                        # Detect trends
                        trend = self._detect_trend(values)
                        if trend:
                            analysis['trend_analysis'].append(trend)

                        # Find support/resistance levels
                        levels = self._find_support_resistance(values)
                        if levels:
                            analysis['support_resistance'].extend(levels)

                elif chart_type in ['volume', 'bar']:
                    # Analyze volume patterns
                    if 'values' in chart_data:
                        volume_analysis = self._analyze_volume(pd.Series(chart_data['values']))
                        analysis['technical_indicators']['volume_analysis'] = volume_analysis

        except Exception as e:
            logger.error(f"Error in chart analysis: {str(e)}")

        return analysis
    def _detect_trend(self, data: pd.Series) -> Dict[str, Any]:
        """Detect price trends"""
        try:
            returns = data.pct_change()
            recent_trend = returns.tail(5).mean()
            
            return {
                'direction': 'bullish' if recent_trend > 0 else 'bearish',
                'strength': abs(recent_trend),
                'period': '5 periods',
                'confidence': min(abs(recent_trend) * 100, 100)  # Convert to percentage
            }
        except Exception:
            return None
    def _find_support_resistance(self, data: pd.Series) -> List[Dict[str, Any]]:
        """Identify support and resistance levels"""
        try:
            levels = []
            rolling_max = data.rolling(20).max()
            rolling_min = data.rolling(20).min()
            
            if not rolling_max.empty:
                levels.append({
                    'type': 'resistance',
                    'level': float(rolling_max.iloc[-1]),
                    'strength': 'primary'
                })
            
            if not rolling_min.empty:
                levels.append({
                    'type': 'support',
                    'level': float(rolling_min.iloc[-1]),
                    'strength': 'primary'
                })
                
            return levels
        except Exception:
            return []
    def _analyze_volume(self, volume_data: pd.Series) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            recent_volume = volume_data.tail(5)
            avg_volume = volume_data.mean()
            
            return {
                'average_volume': float(avg_volume),
                'volume_trend': 'increasing' if recent_volume.mean() > avg_volume else 'decreasing',
                'volume_strength': float(recent_volume.mean() / avg_volume)
            }
        except Exception:
            return {}
    def process_text(self, content: str) -> str:
        """Process and clean text content"""
        if not content:
            return ""
        
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
            
        return content

    def process_tables(self, html_content: str) -> list:
        """Extract and process tables from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = []
            
            for table in soup.find_all('table'):
                df = pd.read_html(str(table))[0]
                tables.append(df.to_dict('records'))
                
            return tables
        except Exception as e:
            logger.error(f"Error processing tables: {str(e)}")
            return []
    def _calculate_sentiment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0

            # Check price movements
            for change in analysis.get('price_changes', []):
                if change['change'] > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
                total_signals += 1

            # Check trends
            for trend in analysis.get('trend_analysis', []):
                if trend['direction'] == 'bullish':
                    bullish_signals += 2  # Weight trends more heavily
                else:
                    bearish_signals += 2
                total_signals += 2

            # Calculate sentiment score
            if total_signals > 0:
                sentiment_score = (bullish_signals - bearish_signals) / total_signals
                
                return {
                    'score': sentiment_score,
                    'direction': 'bullish' if sentiment_score > 0 else 'bearish',
                    'strength': abs(sentiment_score),
                    'confidence': min(abs(sentiment_score) * 100, 100)
                }
            
            return None
        except Exception:
            return None

    def process_charts(self, content: Dict[str, Any]) -> list:
        """Process chart data and descriptions"""
        charts = []
        if 'charts' in content:
            for chart in content['charts']:
                chart_data = {
                    'type': chart.get('type', 'unknown'),
                    'title': chart.get('title', ''),
                    'data': chart.get('data', {}),
                    'description': chart.get('description', '')
                }
                charts.append(chart_data)
        return charts

class ContentAnalyzer:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts
        self.content_processor = ContentProcessor()
        self.content_processor.max_content_length = config.content_max_length
    def _format_analysis_prompt(self, url: str, content_type: str, content_text: str, 
                            analysis_type: str, prompt_template: str, 
                            analysis_structure: Dict[str, Any],
                            processed_content: Dict[str, Any] = None) -> str:
        """Format the analysis prompt with all necessary components"""
        
        # Add tables and charts info if available
        data_sections = ""
        if processed_content:
            if processed_content.get('tables'):
                data_sections += "\nTABLES DATA:\n"
                for idx, table in enumerate(processed_content['tables'], 1):
                    data_sections += f"Table {idx}:\n{json.dumps(table, indent=2)}\n"
            
            if processed_content.get('charts'):
                data_sections += "\nCHARTS DATA:\n"
                for idx, chart in enumerate(processed_content['charts'], 1):
                    data_sections += f"Chart {idx}:\n{json.dumps(chart, indent=2)}\n"

        return f"""
        Analyze this {analysis_type} content and provide professional trading insights:

        URL: {url}
        CONTENT TYPE: {content_type}
        
        CONTENT:
        {content_text}
        {data_sections}

        {prompt_template}

        Please structure your analysis according to the following JSON format:
        {json.dumps(analysis_structure, indent=2)}

        Requirements:
        1. Provide specific, actionable trading insights
        2. Include quantitative metrics where available
        3. Highlight key risk factors and market implications
        4. Focus on time-sensitive opportunities
        5. Include specific price levels and market conditions
        6. Provide clear entry/exit strategies
        """


class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm", "litellm"]
    model: Optional[str] = None
    max_tokens: int = Field(default=400)
    temperature: float = 0
    response_format: Literal["json_beg", "text","json_object","structured_output","tool"] = "text"
    use_cache: bool = True

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:

        if self.response_format == "json_object" and self.client in ["vllm", "litellm","anthropic"]:
            raise ValueError(f"{self.client} does not support json_object response format")
        elif self.response_format == "structured_output" and self.client == "anthropic":
            raise ValueError(f"Anthropic does not support structured_output response format use json_beg or tool instead")
        return self
    


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
    urls_per_query: int = 2
    use_ai_summary: bool = True 
    methods: List[str] = ["selenium", "playwright", "beautifulsoup", "newspaper3k", "scrapy", 
                         "requests_html", "mechanicalsoup", "httpx"]
    default_method: str = "newspaper3k"
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    llm_configs: Dict[str, Dict[str, Any]]

    @model_validator(mode='after')
    def validate_llm_configs(self) -> 'WebSearchConfig':
        try:
            # Convert each llm_config dict to LLMConfig instance
            for key, config in self.llm_configs.items():
                # Extract non-LLMConfig fields
                system_prompt = config.pop('system_prompt', None)
                prompt_template = config.pop('prompt_template', None)
                
                # Validate remaining fields with LLMConfig
                llm_config = LLMConfig(**config)
                
                # Add back the additional fields
                config['system_prompt'] = system_prompt
                config['prompt_template'] = prompt_template
                
                # Update the config with validated values
                self.llm_configs[key] = config
                
        except Exception as e:
            raise ValueError(f"Invalid LLM config: {str(e)}")
        return self

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
            # Get config from llm_configs
            llm_config_dict = self.config.llm_configs["search_query_generation"].copy()
            
            # Parse the YAML content properly
            search_query_section = yaml.safe_load(self.prompts["search_query_generation"])
            
            # Get system prompt and template from parsed YAML
            system_prompt = search_query_section["system_prompt"]
            prompt_template = search_query_section["prompt_template"]
            
            # Remove non-LLMConfig fields before creating LLMConfig
            llm_config_dict.pop('system_prompt', None)
            llm_config_dict.pop('prompt_template', None)
            
            # Create and validate LLMConfig
            llm_config = LLMConfig(**llm_config_dict)
            
            # Format the prompt using the template
            prompt = prompt_template.format(query=base_query)
            
            # Create prompt context
            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=system_prompt,
                new_message=prompt,
                llm_config=llm_config.dict(),
                use_history=False
            )
            
            # Get current date information
            current_date = datetime.now()
            current_year = current_date.year
            current_month = current_date.strftime("%B")  # Full month name
            
            # Enhance base query with time context
            time_context = f"""
            Time Context:
            - Current Year: {current_year}
            - Current Month: {current_month}
            
            Please generate TWO search queries that:
            1. Include specific time frames (e.g., "2024 Q1", "December 2023", "last 30 days")
            2. Focus on recent developments and trends
            3. Include terms like "latest", "recent", "current", "upcoming"
            4. Consider both immediate news and short-term historical context
            
            Base Query: {base_query}
            """
            
            # Format the prompt using the template from config with enhanced context
            prompt = self.prompts["search_query_generation"].format(
                query=time_context
            )
            
            logger.info(f"Using prompt template with time context:\n{prompt}")
            

            responses = await self.ai_utils.run_parallel_ai_completion([context])
            
            if responses and len(responses) > 0:
                # Get response content
                if responses[0].json_object:
                    response_text = json.dumps(responses[0].json_object.object)
                else:
                    response_text = responses[0].str_content

                if not response_text:
                    logger.error("No response content found")
                    # Add time context to base query as fallback
                    time_modified_query = f"{base_query} {current_year} {current_month} latest"
                    return [time_modified_query]

                logger.info(f"AI Response:\n{response_text}")
                
                # Extract and process queries
                queries = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line and not any(line.startswith(x) for x in ['Query:', 'Please', 'Format', 'Make']):
                        cleaned_query = re.sub(r'^\d+\.\s*', '', line)
                        if cleaned_query:
                            # Ensure each query has time context if not already present
                            time_indicators = [str(current_year), current_month, "latest", "recent", "current", "last"]
                            if not any(indicator.lower() in cleaned_query.lower() for indicator in time_indicators):
                                cleaned_query = f"{cleaned_query} {current_year} latest"
                            queries.append(cleaned_query)
                
                # Modify base query to include time context if needed
                time_modified_base = base_query
                if not any(str(current_year) in q for q in [base_query] + queries):
                    time_modified_base = f"{base_query} {current_year} latest"
                
                # Combine queries and ensure uniqueness
                all_queries = [time_modified_base] + [q for q in queries if q != time_modified_base]
                
                # Log the generated queries
                logger.info("\n=== Generated Search Queries with Time Context ===")
                logger.info(f"Original Query: {base_query}")
                logger.info(f"Time-Modified Base Query: {time_modified_base}")
                logger.info(f"Additional Queries Generated: {len(queries)}")
                logger.info("\nAll Queries:")
                for i, query in enumerate(all_queries, 1):
                    logger.info(f"{i}. {query}")
                
                return all_queries
                
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return base query with time context as fallback
            time_modified_query = f"{base_query} {datetime.now().year} latest"
            return [time_modified_query]

    
    def get_urls_for_query(self, query: str, num_results: int = 2) -> List[str]:
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
        self.processed_urls = set()
        
        # Initialize AI utilities with configs from web_search_config.yaml
        oai_request_limits = RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=150000
        )
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None
        )
        
        # Load LLM configs from web_search_config.yaml
        self.llm_configs = config.llm_configs
        
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

    async def generate_ai_summary(self, url: str, content: Union[str, Dict[str, Any]], content_type: str) -> Dict[str, Any]:
        """Generate AI summary with market analysis"""
        try:
            # Initialize content processor
            content_processor = ContentProcessor()
            
            # Process content based on type
            if isinstance(content, dict):
                processed_content = {
                    'text': content_processor.process_text(content.get('text', '')),
                    'tables': content.get('tables', []),
                    'charts': content.get('charts', [])
                }
                
                # Determine content type and get appropriate template
                has_tables = bool(processed_content['tables'])
                has_charts = bool(processed_content['charts'])
                analysis_type = self._determine_analysis_type(has_tables, has_charts)
                prompt_template = self._get_analysis_template(analysis_type)
                
            else:
                processed_content = {
                    'text': content_processor.process_text(str(content)),
                    'tables': [],
                    'charts': []
                }
                analysis_type = "text_only"
                prompt_template = self._get_analysis_template(analysis_type)

            # Create analysis structure
            analysis_structure = self._create_analysis_structure(analysis_type)
            
            # Format the analysis prompt without processed_content parameter
            formatted_prompt = self._format_analysis_prompt(
                url=url,
                content_type=content_type,
                content_text=processed_content['text'],
                analysis_type=analysis_type,
                prompt_template=prompt_template,
                analysis_structure=analysis_structure
            )

            # Get LLM config and create context
            llm_config = LLMConfig(**self.config.llm_configs["content_analysis"])
            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=self.config.llm_configs["content_analysis"].get("system_prompt", ""),
                new_message=formatted_prompt,
                llm_config=llm_config.dict(),
                use_history=False
            )

            # Get AI response
            responses = await self.ai_utils.run_parallel_ai_completion([context])
            
            if responses and len(responses) > 0:
                response = responses[0]
                if response.json_object:
                    return response.json_object.object
                elif response.str_content:
                    return self._structure_text_response(response.str_content, analysis_type)
                
            return self._get_fallback_response(analysis_type)

        except Exception as e:
            logger.error(f"Error in generate_ai_summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_error_response(url, content_type, str(e))
    

    def _determine_analysis_type(self, has_tables: bool, has_charts: bool) -> str:
        """Determine the type of analysis based on content"""
        if has_tables and has_charts:
            return "mixed_content"
        elif has_tables:
            return "table_content"
        elif has_charts:
            return "chart_content"
        return "text_only"

    def _get_analysis_template(self, analysis_type: str) -> str:
        """Get the appropriate analysis template based on content type"""
        templates = {
            "mixed_content": self.prompts.get("mixed_content_analysis", self.prompts["text_only_analysis"]),
            "table_content": self.prompts.get("table_content_analysis", self.prompts["text_only_analysis"]),
            "chart_content": self.prompts.get("chart_content_analysis", self.prompts["text_only_analysis"]),
            "text_only": self.prompts["text_only_analysis"]
        }
        return templates.get(analysis_type, self.prompts["text_only_analysis"])


    def _format_analysis_prompt(self, url: str, content_type: str, content_text: str, 
                            analysis_type: str, prompt_template: str, 
                            analysis_structure: Dict[str, Any]) -> str:
        """Format the analysis prompt with all necessary components"""
        
        return f"""
        Analyze this {analysis_type} content and provide professional trading insights:

        URL: {url}
        CONTENT TYPE: {content_type}
        
        CONTENT:
        {content_text}

        {prompt_template}

        Please structure your analysis according to the following JSON format:
        {json.dumps(analysis_structure, indent=2)}

        Requirements:
        1. Provide specific, actionable trading insights
        2. Include quantitative metrics where available
        3. Highlight key risk factors and market implications
        4. Focus on time-sensitive opportunities
        5. Include specific price levels and market conditions
        6. Provide clear entry/exit strategies
        """
        
    def _get_default_analysis_structure(self) -> Dict[str, Any]:
        """Return the default analysis structure"""
        return {
            "OVERVIEW": {
                "main_points": [],
                "key_metrics": {},
                "critical_developments": []
            },
            "TECHNICAL_ANALYSIS": {
                "patterns": [],
                "indicators": {},
                "key_levels": [],
                "trend_analysis": []
            },
            "FUNDAMENTAL_FACTORS": {
                "market_drivers": [],
                "sector_impacts": [],
                "risk_factors": {}
            },
            "MARKET_IMPACT": {
                "immediate_reactions": {
                    "price_movements": [],
                    "volume_changes": {},
                    "sentiment_shifts": []
                },
                "sentiment": {}
            },
            "TRADING_IMPLICATIONS": {
                "opportunities": [],
                "entry_points": [],
                "risk_levels": [],
                "time_horizons": []
            }
        }

    def _create_analysis_structure(self, analysis_type: str) -> Dict[str, Any]:
        """Create appropriate analysis structure based on content type"""
        base_structure = {
            "OVERVIEW": {
                "main_points": [],
                "key_metrics": [],
                "critical_developments": []
            },
            "TECHNICAL_ANALYSIS": {
                "patterns": [],
                "indicators": [],
                "key_levels": [],
                "trend_analysis": ""
            },
            "FUNDAMENTAL_FACTORS": {
                "market_drivers": [],
                "sector_impacts": [],
                "risk_factors": []
            },
            "TRADING_IMPLICATIONS": {
                "opportunities": [],
                "entry_points": [],
                "risk_levels": [],
                "time_horizons": []
            },
            "MARKET_IMPACT": {
                "immediate_reactions": {
                    "price_movements": [],
                    "volume_changes": [],
                    "sentiment_shifts": []
                },
                "sector_specific": {
                    "affected_sectors": [],
                    "potential_winners": [],
                    "potential_losers": []
                },
                "macroeconomic": {
                    "market_trends": [],
                    "economic_indicators": [],
                    "global_factors": []
                },
                "sentiment": {
                    "retail_sentiment": "",
                    "institutional_outlook": "",
                    "market_psychology": ""
                }
            },
            "RECOMMENDATIONS": {
                "trading_ideas": [],
                "portfolio_adjustments": [],
                "risk_management": [],
                "timing_considerations": []
            }
        }

        # Add content-specific analysis sections
        if analysis_type in ["table_content", "mixed_content"]:
            base_structure["DATA_ANALYSIS"] = {
                "table_insights": [],
                "key_metrics": [],
                "trend_analysis": [],
                "comparative_analysis": []
            }

        if analysis_type in ["chart_content", "mixed_content"]:
            base_structure["CHART_ANALYSIS"] = {
                "pattern_recognition": [],
                "support_resistance": [],
                "trend_indicators": [],
                "volume_analysis": []
            }

        return base_structure

    def _format_analysis_prompt(self, url: str, content_type: str, content_text: str, 
                              analysis_type: str, prompt_template: str, 
                              analysis_structure: Dict[str, Any]) -> str:
        """Format the analysis prompt with all necessary components"""
        return f"""
        Analyze this {analysis_type} content and provide professional trading insights:

        URL: {url}
        CONTENT TYPE: {content_type}
        
        CONTENT:
        {content_text}

        {prompt_template}

        Please structure your analysis according to the following JSON format:
        {json.dumps(analysis_structure, indent=2)}

        Requirements:
        1. Provide specific, actionable trading insights
        2. Include quantitative metrics where available
        3. Highlight key risk factors and market implications
        4. Focus on time-sensitive opportunities
        5. Include specific price levels and market conditions
        6. Provide clear entry/exit strategies
        """

    def _create_prompt_context(self, formatted_prompt: str, llm_config: LLMConfig) -> LLMPromptContext:
        """Create the prompt context for AI processing"""
        context_id = str(uuid.uuid4())
        return LLMPromptContext(
            id=context_id,
            system_string=self.config.llm_configs["content_analysis"]["system_prompt"],
            new_message=formatted_prompt,
            llm_config=llm_config.dict(),
            use_history=False,
            source_id=context_id
        )

    async def _process_response_with_retries(self, context: LLMPromptContext, 
                                           analysis_type: str) -> Dict[str, Any]:
        """Process the AI response with retries and error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                responses = await self.ai_utils.run_parallel_ai_completion([context])
                
                if responses and len(responses) > 0:
                    response = responses[0]
                    
                    if response.json_object:
                        return response.json_object.object
                    
                    if response.str_content:
                        cleaned_content = self._clean_response_content(response.str_content)
                        try:
                            return json.loads(cleaned_content)
                        except json.JSONDecodeError:
                            return self._structure_text_response(cleaned_content, analysis_type)

            except Exception as e:
                logger.error(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

        return self._get_fallback_response(analysis_type)

    def _clean_response_content(self, content: str) -> str:
        """Clean and prepare response content for parsing"""
        content = content.strip()
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content

    def _structure_text_response(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Structure non-JSON response into proper format"""
        lines = content.split('\n')
        structured_response = self._create_analysis_structure(analysis_type)
        
        current_section = None
        current_subsection = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.upper().endswith(':'):
                current_section = line[:-1].upper()
                continue
                
            if current_section in structured_response:
                if isinstance(structured_response[current_section], list):
                    structured_response[current_section].append(line)
                elif isinstance(structured_response[current_section], dict):
                    if line.startswith('-'):
                        if current_subsection:
                            structured_response[current_section][current_subsection].append(line[1:].strip())
                    else:
                        current_subsection = line.lower().replace(' ', '_')
                        if current_subsection not in structured_response[current_section]:
                            structured_response[current_section][current_subsection] = []
        
        return structured_response

    def _get_fallback_response(self, analysis_type: str) -> Dict[str, Any]:
        """Get structured fallback response"""
        return {
            "error": "Analysis failed after multiple attempts",
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "structure": self._create_analysis_structure(analysis_type)
        }

    def _get_error_response(self, url: str, content_type: str, error: str) -> Dict[str, Any]:
        """Get structured error response"""
        return {
            "error": str(error),
            "url": url,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "summary": "Error occurred during analysis",
            "details": {
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        }
    def _get_default_summary(self, has_data: bool = False) -> Dict[str, Any]:
        """Get default summary structure with all analysis types"""
        return {
            "NEWS_ANALYSIS": {
                "key_developments": "Analysis failed - using default summary",
                "market_impact": "Analysis failed - using default summary",
                "sector_implications": "Analysis failed - using default summary"
            },
            "MARKET_SENTIMENT": {
                "investor_reaction": "Analysis failed - using default summary",
                "sentiment_indicators": "Analysis failed - using default summary",
                "confidence_levels": "Analysis failed - using default summary"
            },
            "TRADING_CONSIDERATIONS": {
                "opportunities": "Analysis failed - using default summary",
                "risks": "Analysis failed - using default summary",
                "timeline": "Analysis failed - using default summary"
            },
            "BASIC_SUMMARY": {
                "overview": "Analysis failed - using default summary",
                "key_points": [],
                "conclusion": "Analysis failed - using default summary"
            },
            "KEY_POINTS_ANALYSIS": {
                "market_movements": [],
                "announcements": [],
                "policy_changes": [],
                "market_reactions": [],
                "notable_quotes": [],
                "significant_data": []
            },
            "MARKET_IMPACT_ANALYSIS": {
                "immediate_reactions": {
                    "price_movements": [],
                    "volume_activity": [],
                    "market_changes": []
                },
                "sector_impacts": {
                    "affected_sectors": [],
                    "winners_losers": [],
                    "ripple_effects": []
                },
                "macro_implications": {
                    "interest_rates": "Analysis failed - using default summary",
                    "inflation": "Analysis failed - using default summary",
                    "currency": "Analysis failed - using default summary",
                    "global_markets": "Analysis failed - using default summary"
                },
                "sentiment": {
                    "market_sentiment": "Analysis failed - using default summary",
                    "institutional_reaction": "Analysis failed - using default summary",
                    "retail_reaction": "Analysis failed - using default summary"
                }
            },
            "TRADING_IMPLICATIONS_DETAILED": {
                "opportunities": {
                    "stocks_to_watch": [],
                    "entry_points": [],
                    "risk_levels": [],
                    "time_horizons": []
                },
                "risk_assessment": {
                    "key_risks": [],
                    "hedging_strategies": [],
                    "stop_loss_levels": [],
                    "volatility_outlook": "Analysis failed - using default summary"
                },
                "portfolio_adjustments": {
                    "sector_rotations": [],
                    "allocation_changes": [],
                    "position_sizing": "Analysis failed - using default summary",
                    "diversification": "Analysis failed - using default summary"
                },
                "timing": {
                    "short_term": "Analysis failed - using default summary",
                    "medium_term": "Analysis failed - using default summary",
                    "long_term": "Analysis failed - using default summary",
                    "key_dates": []
                },
                "actionable_recommendations": {
                    "trading_ideas": [],
                    "rationale": [],
                    "price_levels": [],
                    "risk_management": []
                },
                "quantitative_signals": {
                    "technical_levels": [],
                    "arbitrage_opportunities": [],
                    "relative_value": "Analysis failed - using default summary",
                    "risk_metrics": []
                }
            }
        }


    def _determine_content_type(self, content: Dict[str, Any]) -> str:
        """Determine the type of content for analysis"""
        has_tables = bool(content.get('tables'))
        has_charts = bool(content.get('charts'))
        has_text = bool(content.get('text'))

        if has_tables and has_charts:
            return 'mixed_content_analysis'
        elif has_tables:
            return 'table_content_analysis'
        elif has_charts:
            return 'chart_content_analysis'
        else:
            return 'text_only_analysis'
        
    def _parse_ai_response(self, response: LLMOutput, has_data: bool = False) -> Dict[str, Any]:
        """Parse AI response with improved error handling"""
        try:
            if not response or not response.str_content:
                return self._get_default_summary(has_data)

            # Clean up the response
            content = response.str_content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to structure the text response
                return self._structure_text_response(content, has_data)

        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return self._get_default_summary(has_data)
    def _get_default_summary(self, has_data: bool = False) -> Dict[str, Any]:
        """Get default summary structure with proper error handling"""
        summary = {
            "NEWS_ANALYSIS": {
                "key_developments": "Analysis failed - using default summary",
                "market_impact": "Analysis failed - using default summary",
                "sector_implications": "Analysis failed - using default summary"
            },
            "MARKET_SENTIMENT": {
                "investor_reaction": "Analysis failed - using default summary",
                "sentiment_indicators": "Analysis failed - using default summary",
                "confidence_levels": "Analysis failed - using default summary"
            },
            "TRADING_CONSIDERATIONS": {
                "opportunities": "Analysis failed - using default summary",
                "risks": "Analysis failed - using default summary",
                "timeline": "Analysis failed - using default summary"
            }
        }
        
        if has_data:
            summary.update({
                "DATA_ANALYSIS": {
                    "metrics": "Analysis failed - using default summary",
                    "trends": "Analysis failed - using default summary",
                    "insights": "Analysis failed - using default summary"
                }
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
                        logger.info(f"Trying method {method}")
                        
                        content = None
                        title = None

                        # Create a timeout context
                        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                        
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            if method == "selenium":
                                title, content = await self.content_extractor.extract_with_selenium(url)
                            elif method == "playwright":
                                title, content = await self.content_extractor.extract_with_playwright(url)
                            # ... other extraction methods ...

                            if content and isinstance(content, dict):
                                logger.info(f"Successfully extracted content using {method}")
                                
                                # Determine content type
                                content_type = "Contains tables/charts" if content.get('has_data') else "Text only"
                                logger.info(f"Content type: {content_type}")
                                
                                # Generate summary
                                summary = await self.generate_ai_summary(url, content, content_type) if self.config.use_ai_summary else {}
                                
                                if summary and not isinstance(summary, dict):
                                    summary = {"raw_summary": str(summary)}
                                
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