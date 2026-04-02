"""
Regional data processing for Bangkong LLM Training System
"""

import re
import json
from typing import List, Dict, Any
from ...config.schemas import BangkongConfig
from ...data.processors.text_processor import TextProcessor


class RegionalTextProcessor(TextProcessor):
    """Text processor that handles regional language specifics."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize regional text processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
        self.primary_language = config.model.primary_language
        self.supported_languages = config.model.supported_languages
        self.region = config.model.region
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess text data with regional language handling.
        
        Args:
            data: List of text strings.
            
        Returns:
            Preprocessed text data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for text in data:
            # Apply language-specific preprocessing
            processed_text = self._language_specific_preprocessing(text)
            
            # Filter by length
            if min_length <= len(processed_text) <= max_length:
                # Apply text cleaning
                processed_text = self._clean_text(processed_text)
                if processed_text:  # Only add non-empty texts
                    preprocessed.append(processed_text)
        
        return preprocessed
    
    def _language_specific_preprocessing(self, text: str) -> str:
        """
        Apply language-specific preprocessing.
        
        Args:
            text: Text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        # Handle Chinese text specifics
        if self.primary_language == "zh" or "zh" in self.supported_languages:
            # Remove excessive whitespace but preserve some for readability
            text = re.sub(r'\s+', ' ', text)
            
            # Handle Chinese punctuation normalization
            text = re.sub(r'[，。！？；：""''（）【】《》]', lambda m: {
                '，': ',',
                '。': '.',
                '！': '!',
                '？': '?',
                '；': ';',
                '：': ':',
                '""': '"',
                "''": '"',
                '（': '(',
                '）': ')',
                '【': '[',
                '】': ']',
                '《': '<',
                '》': '>'
            }.get(m.group(), m.group()), text)
        
        # Handle other languages as needed
        if self.primary_language == "ja" or "ja" in self.supported_languages:
            # Japanese-specific processing
            pass
        
        if self.primary_language == "ko" or "ko" in self.supported_languages:
            # Korean-specific processing
            pass
        
        return text


class MixedLanguageProcessor(TextProcessor):
    """Processor for mixed-language datasets."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize mixed language processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
        self.supported_languages = config.model.supported_languages
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess mixed-language text data.
        
        Args:
            data: List of text strings.
            
        Returns:
            Preprocessed text data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for text in data:
            # Detect languages in the text
            languages = self._detect_languages(text)
            
            # Process based on detected languages
            processed_text = self._process_mixed_language_text(text, languages)
            
            # Filter by length
            if min_length <= len(processed_text) <= max_length:
                # Apply text cleaning
                processed_text = self._clean_text(processed_text)
                if processed_text:  # Only add non-empty texts
                    preprocessed.append(processed_text)
        
        return preprocessed
    
    def _detect_languages(self, text: str) -> List[str]:
        """
        Detect languages in text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            List of detected languages.
        """
        languages = []
        
        # Simple language detection based on character sets
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            languages.append('zh')
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese characters
            languages.append('ja')
        if re.search(r'[\uac00-\ud7af]', text):  # Korean characters
            languages.append('ko')
        if re.search(r'[a-zA-Z]', text):  # Latin characters
            languages.append('en')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_languages = []
        for lang in languages:
            if lang not in seen:
                seen.add(lang)
                unique_languages.append(lang)
        
        return unique_languages if unique_languages else ['en']  # Default to English
    
    def _process_mixed_language_text(self, text: str, languages: List[str]) -> str:
        """
        Process mixed-language text.
        
        Args:
            text: Text to process.
            languages: Detected languages.
            
        Returns:
            Processed text.
        """
        # For mixed-language text, we might want to:
        # 1. Segment by language
        # 2. Apply language-specific processing
        # 3. Handle transliteration where appropriate
        
        # For now, we'll apply general cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text


def create_regional_data_processor(config: BangkongConfig) -> TextProcessor:
    """
    Create a regional data processor based on configuration.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Appropriate text processor for the region/language
    """
    if len(config.model.supported_languages) > 1:
        return MixedLanguageProcessor(config)
    elif config.model.region != "global":
        return RegionalTextProcessor(config)
    else:
        return TextProcessor(config)