"""
Vietnamese text normalization utilities.

Handles Unicode normalization and Vietnamese-specific preprocessing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


class VietnameseNormalizer:
    """
    Normalizer for Vietnamese text.
    
    Features:
    - Unicode normalization (NFKC)
    - Whitespace normalization
    - Optional word tokenization (using PyVi)
    """
    
    def __init__(self, use_tokenizer: bool = False):
        """
        Initialize normalizer.
        
        Args:
            use_tokenizer: Whether to use Vietnamese word tokenizer
        """
        self.use_tokenizer = use_tokenizer
        self._tokenizer = None
        
        if use_tokenizer:
            self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize Vietnamese tokenizer."""
        try:
            from pyvi import ViTokenizer
            self._tokenizer = ViTokenizer
        except ImportError:
            import warnings
            warnings.warn(
                "PyVi not installed. Install with: pip install pyvi"
            )
            self._tokenizer = None
    
    def normalize(self, text: str) -> str:
        """
        Normalize Vietnamese text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Vietnamese tokenization if enabled
        if self.use_tokenizer and self._tokenizer:
            text = self._tokenizer.tokenize(text)
        
        return text
    
    def normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            text: Answer text
            
        Returns:
            Normalized answer
        """
        text = self.normalize(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove trailing periods
        while text.endswith('.'):
            text = text[:-1].rstrip()
        
        return text
    
    def extract_characters(self, text: str) -> set:
        """
        Extract unique non-whitespace characters.
        
        Used for character-level F1 calculation.
        
        Args:
            text: Input text
            
        Returns:
            Set of characters
        """
        normalized = self.normalize_answer(text)
        if not normalized:
            return set()
        return {ch for ch in normalized if not ch.isspace()}


# Global normalizer instance
_default_normalizer: Optional[VietnameseNormalizer] = None


def get_normalizer(use_tokenizer: bool = False) -> VietnameseNormalizer:
    """
    Get or create default normalizer.
    
    Args:
        use_tokenizer: Whether to use Vietnamese tokenizer
        
    Returns:
        VietnameseNormalizer instance
    """
    global _default_normalizer
    
    if _default_normalizer is None or _default_normalizer.use_tokenizer != use_tokenizer:
        _default_normalizer = VietnameseNormalizer(use_tokenizer=use_tokenizer)
    
    return _default_normalizer


def normalize_text(text: str, use_tokenizer: bool = False) -> str:
    """
    Convenience function to normalize text.
    
    Args:
        text: Input text
        use_tokenizer: Whether to use Vietnamese tokenizer
        
    Returns:
        Normalized text
    """
    return get_normalizer(use_tokenizer).normalize(text)


def normalize_answer(text: str, use_tokenizer: bool = False) -> str:
    """
    Convenience function to normalize answer.
    
    Args:
        text: Answer text
        use_tokenizer: Whether to use Vietnamese tokenizer
        
    Returns:
        Normalized answer
    """
    return get_normalizer(use_tokenizer).normalize_answer(text)
