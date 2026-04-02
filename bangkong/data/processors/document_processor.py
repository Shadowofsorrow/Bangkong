"""
Document processor for Bangkong LLM Training System
"""

import torch
from typing import List, Union
from .base_processor import DataProcessor
from ...config.schemas import BangkongConfig


class DocumentProcessor(DataProcessor):
    """Processor for document data (PDF, DOCX, etc.)."""
    
    def load(self, path: str) -> str:
        """
        Load document data from a file.
        
        Args:
            path: Path to the document file.
            
        Returns:
            Extracted text from the document.
        """
        try:
            if path.endswith('.pdf'):
                return self._load_pdf(path)
            elif path.endswith('.docx'):
                return self._load_docx(path)
            else:
                raise ValueError(f"Unsupported document format: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load document from {path}: {e}")
    
    def _load_pdf(self, path: str) -> str:
        """
        Load text from a PDF file.
        
        Args:
            path: Path to the PDF file.
            
        Returns:
            Extracted text from the PDF.
        """
        try:
            import PyPDF2
            
            text = ""
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Please install it with: pip install PyPDF2")
    
    def _load_docx(self, path: str) -> str:
        """
        Load text from a DOCX file.
        
        Args:
            path: Path to the DOCX file.
            
        Returns:
            Extracted text from the DOCX.
        """
        try:
            from docx import Document
            
            doc = Document(path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Please install it with: pip install python-docx")
    
    def preprocess(self, data: str) -> List[str]:
        """
        Preprocess document data.
        
        Args:
            data: Extracted text from the document.
            
        Returns:
            List of preprocessed text segments.
        """
        # Split text into paragraphs
        paragraphs = data.split('\n')
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Apply text cleaning
        cleaned_paragraphs = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for paragraph in paragraphs:
            if min_length <= len(paragraph) <= max_length:
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def validate(self, data: List[str]) -> bool:
        """
        Validate document data.
        
        Args:
            data: List of text segments to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
            
        for item in data:
            if not isinstance(item, str):
                return False
                
            # Check if text is not empty
            if not item.strip():
                return False
                
        return True