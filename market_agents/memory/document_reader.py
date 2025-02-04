from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import re
from uuid import UUID
from PyPDF2 import PdfReader

class DocumentReader(ABC):
    """Base class for document readers that process different file formats."""
    
    @abstractmethod
    def read(self, file_path: Union[str, Path]) -> str:
        """
        Read and process content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Processed text content from the document
            
        Raises:
            ValueError: If file format is not supported
        """
        pass

class PDFReader(DocumentReader):
    """Reader for PDF documents."""
    
    def read(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError("File must be a PDF document")
            
        text_content = ""
        pdf_reader = PdfReader(str(file_path))
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            cleaned_text = re.sub(r'FTLN \d+', '', page_text)
            cleaned_text = re.sub(r'^\d+\s*', '', cleaned_text, flags=re.MULTILINE)
            text_content += cleaned_text
            
        return text_content

class TXTReader(DocumentReader):
    """Reader for plain text documents."""
    
    def read(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        if not file_path.suffix.lower() == '.txt':
            raise ValueError("File must be a TXT document")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

class DocumentReaderFactory:
    """Factory for creating appropriate document readers based on file type."""
    
    _readers = {
        '.pdf': PDFReader(),
        '.txt': TXTReader()
    }
    
    @classmethod
    def get_reader(cls, file_path: Union[str, Path]) -> DocumentReader:
        """
        Get appropriate reader for the given file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentReader: Appropriate reader instance for the file type
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        reader = cls._readers.get(file_path.suffix.lower())
        
        if reader is None:
            supported_formats = ", ".join(cls._readers.keys())
            raise ValueError(f"Unsupported file format. Supported formats are: {supported_formats}")
            
        return reader