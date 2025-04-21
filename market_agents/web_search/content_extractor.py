import logging
import re
from typing import Dict, List, Any, Tuple
from bs4 import BeautifulSoup
import httpx
from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests_html import AsyncHTMLSession
import mechanicalsoup
from parsel import Selector as ParselSelector

logger = logging.getLogger(__name__)

class ContentExtractor:
    def __init__(self, config=None):
        self.config = config
        if hasattr(self.config, 'request_timeout'):
            self.timeout = self.config.request_timeout
        elif isinstance(self.config, dict):
            self.timeout = self.config.get('request_timeout', 30)
        else:
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