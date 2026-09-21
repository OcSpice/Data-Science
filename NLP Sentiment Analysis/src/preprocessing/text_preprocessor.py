"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module handles text preprocessing, cleaning, and PII anonymization.
"""

import re
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


class TextPreprocessor:
    """
    Handles text cleaning, normalization, and preprocessing for NLP tasks.
    
    Attributes:
        stop_words: Set of English stopwords
        lemmatizer: WordNet lemmatizer instance
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self):
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.name_pattern = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
    
    def _download_nltk_resources(self) -> None:
        """Download required NLTK resources if not already present."""
        resources = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
            try:
                nltk.data.find(f'taggers/{resource}')
            except LookupError:
                pass
    
    def clean_text(self, text: str) -> str:
        """
        Perform comprehensive text cleaning.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text with HTML, URLs, special characters removed
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        text = self.phone_pattern.sub('[PHONE]', text)
        text = self.special_char_pattern.sub(' ', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def anonymize_pii(self, text: str) -> str:
        """
        Anonymize potential PII (Personally Identifiable Information) in text.
        
        Args:
            text: Input text that may contain PII
            
        Returns:
            Text with PII replaced by placeholders
        """
        if not isinstance(text, str):
            return ""
        
        text = self.email_pattern.sub('[EMAIL_REDACTED]', text)
        text = self.phone_pattern.sub('[PHONE_REDACTED]', text)
        text = self.name_pattern.sub('[NAME_REDACTED]', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove English stopwords from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stop: bool = True, 
                   apply_lemmatization: bool = True) -> str:
        """
        Full preprocessing pipeline for a single text document.
        
        Args:
            text: Raw input text
            remove_stop: Whether to remove stopwords
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            Fully preprocessed text as a string
        """
        cleaned = self.clean_text(text)
        anonymized = self.anonymize_pii(cleaned)
        tokens = self.tokenize(anonymized)
        
        if remove_stop:
            tokens = self.remove_stopwords(tokens)
        
        if apply_lemmatization:
            tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], remove_stop: bool = True,
                         apply_lemmatization: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw input texts
            remove_stop: Whether to remove stopwords
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            List of fully preprocessed texts
        """
        return [self.preprocess(text, remove_stop, apply_lemmatization) 
                for text in texts]
