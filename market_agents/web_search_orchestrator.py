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

from market_agents.agents.personas.persona import Persona, generate_persona
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import (ParallelAIUtilities,
                                                    RequestLimits)
from market_agents.logger_utils import *

# Set up logging
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
    summary: Optional[dict] = {}  # Changed to Optional[dict]
    agent_id: str

class WebSearchConfig(BaseSettings):
    """Configuration for web search"""
    max_concurrent_requests: int = 50
    rate_limit: float = 0.1
    content_max_length: int = 1000
    request_timeout: int = 30
    llm_configs: List[LLMConfig] = [
        LLMConfig(
            client="openai",
            model=os.getenv("OPENAI_MODEL"),  # Use env variable directly
            max_tokens=500,
            temperature=0.7,
            use_cache=True
        ),
    ]  # Simplified to only use OpenAI
    use_ai_summary: bool = True
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

class WebSearchAgent:
    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.results: List[WebSearchResult] = []
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Initialize AI utilities for content processing - only OpenAI
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=150000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None  # Remove Anthropic
        )
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    async def fetch_url_with_playwright(self, url: str) -> tuple[str, str]:
        """Fetch URL using Playwright for JavaScript-heavy sites"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            try:
                logger.info(f"Navigating to {url} with Playwright")
                # Increase timeout and add bypass options
                await page.goto(url, wait_until='networkidle', timeout=60000)
                
                # Wait for content to load
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(2000)  # Additional 2s wait
                
                # Get the full page content
                content = await page.evaluate('''() => {
                    // Function to convert element to markdown-like format
                    function elementToMarkdown(element) {
                        let text = '';
                        
                        // Handle different element types
                        switch(element.tagName.toLowerCase()) {
                            case 'h1':
                            case 'h2':
                            case 'h3':
                            case 'h4':
                            case 'h5':
                            case 'h6':
                                text += '#'.repeat(parseInt(element.tagName[1])) + ' ' + element.innerText + '\\n\\n';
                                break;
                            case 'p':
                                text += element.innerText + '\\n\\n';
                                break;
                            case 'ul':
                            case 'ol':
                                Array.from(element.children).forEach(li => {
                                    text += '- ' + li.innerText + '\\n';
                                });
                                text += '\\n';
                                break;
                            case 'blockquote':
                                text += '> ' + element.innerText + '\\n\\n';
                                break;
                            default:
                                if (element.innerText && element.innerText.trim()) {
                                    text += element.innerText + '\\n\\n';
                                }
                        }
                        return text;
                    }
                    
                    // Get all content elements
                    const contentElements = document.querySelectorAll('body *');
                    let markdown = '';
                    
                    // Convert page to markdown
                    contentElements.forEach(element => {
                        // Skip hidden elements and common non-content elements
                        if (element.offsetParent !== null && 
                            !element.closest('nav, header, footer, .cookie-banner, #cookie-notice, .ad, .advertisement')) {
                            markdown += elementToMarkdown(element);
                        }
                    });
                    
                    // Clean up the markdown
                    return markdown
                        .replace(/\\n{3,}/g, '\\n\\n')  // Remove extra newlines
                        .replace(/\\s+/g, ' ')  // Normalize whitespace
                        .trim();
                }''')
                
                title = await page.title()
                
                if content:
                    logger.info(f"Successfully extracted content from {url}: {len(content)} characters")
                else:
                    # Fallback to basic HTML if markdown extraction fails
                    content = await page.content()
                    logger.warning(f"Falling back to HTML content for {url}")
                
                await browser.close()
                return title, content
                
            except Exception as e:
                logger.error(f"Playwright error for {url}: {str(e)}")
                try:
                    # Attempt to get whatever content is available
                    content = await page.content()
                    title = await page.title()
                    await browser.close()
                    return title, content
                except:
                    await browser.close()
                    return "", ""

    async def generate_ai_summary(self, content: str, url: str) -> dict:
        """Generate AI summary using parallel inference"""
        try:
            # First try to generate a basic summary without market analysis
            basic_prompt = f"""Provide a brief summary of this article in 2-3 sentences:

    Content:
    {content[:4000]}"""

            basic_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=basic_prompt,
                prompt=basic_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )
            
            # Get basic summary first
            basic_results = await self.ai_utils.run_parallel_ai_completion([basic_context])
            basic_summary = ""
            if basic_results and basic_results[0] and hasattr(basic_results[0], 'choices'):
                basic_summary = basic_results[0].choices[0].message.content.strip()
            
            # Extract key points from content
            key_points_prompt = f"""List 3-4 key points from this article:

    Content:
    {content[:4000]}"""

            points_context = LLMPromptContext(
                id=str(uuid.uuid4()),
                new_message=key_points_prompt,
                prompt=key_points_prompt,
                source_id=str(uuid.uuid4()),
                llm_config=self.config.llm_configs[0].dict()
            )
            
            # Get key points
            points_results = await self.ai_utils.run_parallel_ai_completion([points_context])
            key_points = []
            if points_results and points_results[0] and hasattr(points_results[0], 'choices'):
                points_text = points_results[0].choices[0].message.content
                key_points = [point.strip('- ').strip() for point in points_text.split('\n') if point.strip()]
            
            # Try to get market impact and trading implications
            try:
                analysis_prompt = f"""Analyze the market impact and trading implications of this article:

    Content:
    {content[:4000]}

    Provide:
    1. Market Impact (1-2 sentences)
    2. Trading Implications (1-2 sentences)"""

                analysis_context = LLMPromptContext(
                    id=str(uuid.uuid4()),
                    new_message=analysis_prompt,
                    prompt=analysis_prompt,
                    source_id=str(uuid.uuid4()),
                    llm_config=self.config.llm_configs[0].dict()
                )
                
                analysis_results = await self.ai_utils.run_parallel_ai_completion([analysis_context])
                market_impact = "Impact analysis unavailable"
                trading_implications = "Trading analysis unavailable"
                
                if analysis_results and analysis_results[0] and hasattr(analysis_results[0], 'choices'):
                    analysis_text = analysis_results[0].choices[0].message.content
                    if '1.' in analysis_text and '2.' in analysis_text:
                        parts = analysis_text.split('2.')
                        market_impact = parts[0].replace('1.', '').strip()
                        trading_implications = parts[1].strip()
            
            except Exception as e:
                logger.warning(f"Failed to generate market analysis for {url}: {str(e)}")
                market_impact = "Impact analysis unavailable"
                trading_implications = "Trading analysis unavailable"
            
            summary_dict = {
                "summary": basic_summary or f"Article about {content[:100]}...",
                "key_points": key_points or ["Key point extraction failed"],
                "market_impact": market_impact,
                "trading_implications": trading_implications
            }
            
            logger.info(f"""
    === Summary Generated for {url} ===
    Summary: {summary_dict['summary']}
    Key Points: {json.dumps(summary_dict['key_points'], indent=2)}
    Market Impact: {summary_dict['market_impact']}
    Trading Implications: {summary_dict['trading_implications']}
    =====================================
            """)
            
            return summary_dict
                
        except Exception as e:
            logger.error(f"Error in AI summary generation for {url}: {str(e)}")
            # Create a basic summary from the content even if AI fails
            try:
                first_paragraph = content.split('\n\n')[0]
                return {
                    "summary": first_paragraph[:200] + "...",
                    "key_points": [
                        content.split('\n\n')[1][:100] + "..." if len(content.split('\n\n')) > 1 else "No key points available",
                    ],
                    "market_impact": "Impact analysis unavailable",
                    "trading_implications": "Trading analysis unavailable"
                }
            except:
                return {
                    "summary": content[:200] + "..." if content else "Summary unavailable",
                    "key_points": ["Content extraction failed"],
                    "market_impact": "Impact analysis unavailable",
                    "trading_implications": "Trading analysis unavailable"
                }

    def build_agent_context(self, title: str, content: str, summary: Optional[str], persona: Optional[Persona]) -> str:
        """Build personalized context based on agent persona"""
        context_parts = []
        
        if title:
            context_parts.append(f"Title: {title}")
        
        if summary:
            context_parts.append(f"Analysis: {summary}")
        
        if persona:
            # Use risk_tolerance directly
            risk_level = "high-risk" if getattr(persona, 'risk_tolerance', 0.5) > 0.7 else \
                        "low-risk" if getattr(persona, 'risk_tolerance', 0.5) < 0.3 else \
                        "moderate-risk"
            
            experience_level = getattr(persona, 'investment_experience', 'intermediate')
            trading_style = ", ".join(getattr(persona, 'trader_type', ['general']))
            objectives = ", ".join(getattr(persona, 'objectives', []))
            
            context_parts.append(f"""
                Agent Profile:
                - Experience Level: {experience_level}
                - Risk Profile: {risk_level}
                - Trading Style: {trading_style}
                - Investment Goals: {objectives}
                """.strip())
        
        if content:
            # Add truncated content with focus on key information
            context_parts.append(f"Source Content: {content[:300]}...")
        
        return "\n\n".join(context_parts)
    async def fetch_url_regular(self, session: aiohttp.ClientSession, url: str, persona: Optional[Persona] = None) -> WebSearchResult:
        """Regular HTTP request fallback"""
        try:
            async with session.get(url, headers=self.headers, timeout=self.config.request_timeout) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    title = self.extract_title(soup)
                    content = self.extract_content(soup)
                    
                    return WebSearchResult(
                        url=url,
                        title=title,
                        content=content[:self.config.content_max_length],
                        timestamp=datetime.now(),
                        status="success",
                        persona=persona
                    )
                else:
                    return WebSearchResult(
                        url=url,
                        title="",
                        content="",
                        timestamp=datetime.now(),
                        status=f"error: HTTP {response.status}",
                        persona=persona
                    )
        except Exception as e:
            return WebSearchResult(
                url=url,
                title="",
                content="",
                timestamp=datetime.now(),
                status=f"error: {str(e)}",
                persona=persona
            )

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> WebSearchResult:
        """Fetch and process a single URL"""
        try:
            title, content = await self.fetch_url_with_playwright(url)
            
            # Handle edge case where content is an error URL
            if content and "errors.edgesuite.net" in content:
                # Try fetching again with different options
                await asyncio.sleep(2)  # Wait briefly
                title, content = await self.fetch_url_with_playwright(url)
            
            if not content or "errors.edgesuite.net" in content:
                return WebSearchResult(
                    url=url,
                    title="Access Denied",
                    content="Failed to access content - Access Denied",
                    timestamp=datetime.now(),
                    status="error: access denied",
                    summary={
                        "summary": "Unable to access article content due to access restrictions",
                        "key_points": ["Access to content was denied"],
                        "market_impact": "Impact analysis unavailable",
                        "trading_implications": "Trading analysis unavailable"
                    },
                    agent_id=str(uuid.uuid4())
                )
            
            # Generate AI summary from whatever content we have
            summary_dict = await self.generate_ai_summary(content, url)
            
            return WebSearchResult(
                url=url,
                title=title or "",
                content=content[:self.config.content_max_length],
                timestamp=datetime.now(),
                status="success",
                summary=summary_dict,
                agent_id=str(uuid.uuid4())
            )
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return WebSearchResult(
                url=url,
                title="Error",
                content=f"Failed to fetch content: {str(e)}",
                timestamp=datetime.now(),
                status=f"error: {str(e)}",
                summary={
                    "summary": f"Error fetching article: {str(e)}",
                    "key_points": ["Content fetch failed"],
                    "market_impact": "Impact analysis unavailable",
                    "trading_implications": "Trading analysis unavailable"
                },
                agent_id=str(uuid.uuid4())
            )

    def build_agent_context(self, title: str, content: str, summary: Optional[str], persona: Optional[Persona]) -> str:
        """Build personalized context based on agent persona"""
        context_parts = []
        
        if title:
            context_parts.append(f"Title: {title}")
        
        if summary:
            context_parts.append(f"Analysis: {summary}")
        
        if persona:
            # Add persona-specific context
            # Get risk_tolerance from risk_appetite instead
            risk_mapping = {
                "Conservative": "low-risk",
                "Moderate": "moderate-risk",
                "Aggressive": "high-risk"
            }
            risk_context = risk_mapping.get(persona.risk_appetite, "moderate-risk")
            experience_level = persona.investment_experience.lower()
            trading_style = ", ".join(persona.trader_type)
            
            context_parts.append(f"""
                Agent Profile:
                - Experience Level: {experience_level}
                - Risk Profile: {risk_context}
                - Trading Style: {trading_style}
                - Investment Goals: {', '.join(persona.objectives)}
                        """.strip())
        
        if content:
            # Add truncated content with focus on key information
            context_parts.append(f"Source Content: {content[:300]}...")
        
        return "\n\n".join(context_parts)
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title with fallbacks"""
        title = None
        
        # Try different title elements
        for selector in [
            'h1.article-title',
            'h1.entry-title',
            'h1.post-title',
            'article h1',
            'main h1',
            lambda s: s.title.string if s.title else None,
        ]:
            if callable(selector):
                title = selector(soup)
            else:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
            
            if title:
                break
                
        return title or "No title"
    def extract_content(self, soup: BeautifulSoup) -> str:
        """Enhanced content extraction"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        content = []
        
        # Priority elements for content
        priority_selectors = [
            'article',
            'main',
            '.post-content',
            '.article-content',
            '.entry-content',
            'div[itemprop="articleBody"]',
        ]
        
        # Try priority selectors first
        for selector in priority_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Minimum content length threshold
                    content.append(text)
                    break
        
        # Fallback to paragraph extraction if no main content found
        if not content:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Filter out short paragraphs
                    content.append(text)
        
        return ' '.join(content)
    def is_content_element(self, element) -> bool:
        """Determine if an element likely contains main content"""
        if element.name in ['article', 'main']:
            return True
        
        # Check for common content-related classes/ids
        element_str = str(element.get('class', '')) + str(element.get('id', ''))
        content_indicators = ['content', 'article', 'post', 'entry', 'main']
        return any(indicator in element_str.lower() for indicator in content_indicators)

    def build_context(self, title: str, content: str, summary: Optional[str] = None) -> str:
        """Build context string from content"""
        context_parts = []
        if title:
            context_parts.append(f"Title: {title}")
        if summary:
            context_parts.append(f"Summary: {summary}")
        if content:
            context_parts.append(f"Content: {content[:500]}...")  # First 500 chars
        return "\n\n".join(context_parts)

    async def process_urls(self, urls: List[str]) -> None:
        """Process a list of URLs concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]  # Removed persona parameter
            self.results = await asyncio.gather(*tasks)

    def save_results(self, output_file: str):
        """Save results and display summaries"""
        results_dict = []
        
        logger.info("\n=== ARTICLE SUMMARIES ===")
        
        for result in self.results:
            if result is None:
                continue
                
            try:
                result_data = result.model_dump(exclude_none=True)
                results_dict.append(result_data)
                
                # Display summary if available
                if result.summary:
                    logger.info(f"""
        URL: {result.url}
        TITLE: {result.title}
        SUMMARY:
        {json.dumps(result.summary, indent=2)}
        -------------------
                        """)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
def load_config(config_path: Path = Path("./market_agents/web_search_config.yaml")) -> WebSearchConfig:
    """Load configuration from YAML file with environment variable support"""
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    # Override with environment variables if they exist
    yaml_data['max_concurrent_requests'] = int(os.getenv('WEB_SEARCH_MAX_REQUESTS', yaml_data['max_concurrent_requests']))
    yaml_data['rate_limit'] = float(os.getenv('WEB_SEARCH_RATE_LIMIT', yaml_data['rate_limit']))
    yaml_data['content_max_length'] = int(os.getenv('WEB_SEARCH_CONTENT_LENGTH', yaml_data['content_max_length']))
    yaml_data['request_timeout'] = int(os.getenv('WEB_SEARCH_TIMEOUT', yaml_data['request_timeout']))
    
    # Fix boolean handling
    use_ai_summary_env = os.getenv('WEB_SEARCH_USE_AI_SUMMARY')
    if use_ai_summary_env is not None:
        yaml_data['use_ai_summary'] = use_ai_summary_env.lower() == 'true'
    
    return WebSearchConfig(**yaml_data)

def load_urls(input_file: str = "market_agents/urls.json") -> List[str]:
    """Load URLs from JSON file"""
    try:
        # Get the absolute path to the JSON file
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
        # Load configuration and URLs
        config = load_config()
        urls = load_urls("market_agents/urls.json")
        logger.info(f"Loaded {len(urls)} URLs")
        
        # Initialize and run web search
        agent = WebSearchAgent(config)
        start_time = time.time()
        
        await agent.process_urls(urls)
        
        # Save results
        output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agent.save_results(output_file)
        
        # Log summary
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