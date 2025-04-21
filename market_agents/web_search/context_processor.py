import re
import logging
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Process and structure different types of content"""
    
    def __init__(self):
        self.max_content_length = 4000

    def process_text(self, content: str) -> str:
        """Process and clean text content"""
        if not content:
            return ""
        
        content = re.sub(r'\s+', ' ', content).strip()
        
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