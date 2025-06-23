import os
import re
import requests
import argparse

from bs4 import BeautifulSoup
import concurrent.futures
import time
from dotenv import load_dotenv

class WebSearch:
    def __init__(self):
        load_dotenv()
    
    @staticmethod
    def google_search(query, num_results=10):
        url = 'https://www.google.com/search'
        params = {'q': query, 'num': num_results}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.3'}
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='tF2Cxc')
        urls = [result.find('a')['href'] for result in results]
        #return WebSearch._scrape_results_parallel(urls)
        return urls
    
    @staticmethod
    def _scrape_results_parallel(url_list):
        results = []
        # Fetch page content in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(WebSearch._get_page_content, url) for url in url_list]

        for future in concurrent.futures.as_completed(futures):
            content = future.result()
            if content:
                results.append(content)

        return results

    @staticmethod
    def _get_page_content(url):
        try:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/97.0.4692.71 Safari/537.36"
            )

            # Make request
            response = requests.get(url, headers={'User-Agent': user_agent})
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            text_content = ' '.join(paragraphs)
            
            # Remove extra whitespace using regex
            text_content = re.sub(r'\s+', ' ', text_content)
            
            # Extract tables
            table_data = []
            for table in soup.find_all('table'):
                table_rows = [[cell.get_text(strip=True) for cell in row.find_all('td')] for row in table.find_all('tr')]
                table_data.append(table_rows)
            
            if text_content:
                return {'url': url, 
                        'content': text_content,
                        'tables': table_data}
            else:
                None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from {url}: {e}")
            return None

        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

