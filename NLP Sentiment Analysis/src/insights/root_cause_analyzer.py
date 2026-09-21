"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module performs root-cause analysis and keyword extraction from negative reviews.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class RootCauseAnalyzer:
    """
    Extracts root causes and key phrases from negative customer reviews.
    
    Attributes:
        AUTHOR: Class-level constant for author attribution
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    CUSTOMER_SUPPORT_KEYWORDS = [
        'customer support', 'customer service', 'support team', 
        'service representative', 'help desk', 'support staff',
        'customer care', 'support agent', 'service agent'
    ]
    
    def __init__(self):
        self.ngram_pattern = re.compile(r'\b[a-z]+\b')
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Preprocessed text
            n: Size of n-grams to extract
            
        Returns:
            List of n-grams
        """
        words = self.ngram_pattern.findall(text.lower())
        
        if len(words) < n:
            return []
        
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        return ngrams
    
    def analyze_negative_reviews(self, negative_texts: List[str], 
                                  top_n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """
        Analyze negative reviews to extract root-cause keywords.
        
        Args:
            negative_texts: List of preprocessed negative review texts
            top_n: Number of top keywords to return
            
        Returns:
            Dictionary with keyword categories and their frequencies
        """
        unigrams = []
        bigrams = []
        trigrams = []
        
        for text in negative_texts:
            unigrams.extend(self.extract_ngrams(text, n=1))
            bigrams.extend(self.extract_ngrams(text, n=2))
            trigrams.extend(self.extract_ngrams(text, n=3))
        
        unigram_counts = Counter(unigrams)
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        common_unigrams = ['review', 'product', 'use', 'work', 'time', 'day', 
                          'one', 'like', 'get', 'go', 'see', 'know', 'take']
        
        filtered_unigrams = {k: v for k, v in unigram_counts.items() 
                            if k not in common_unigrams and len(k) > 2}
        
        top_unigrams = sorted(filtered_unigrams.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        
        top_bigrams = bigram_counts.most_common(top_n)
        top_trigrams = trigram_counts.most_common(top_n)
        
        return {
            'top_unigrams': top_unigrams,
            'top_bigrams': top_bigrams,
            'top_trigrams': top_trigrams
        }
    
    def detect_customer_support_mentions(self, texts: List[str]) -> Tuple[int, float]:
        """
        Detect mentions of customer support issues in reviews.
        
        Args:
            texts: List of review texts (raw or preprocessed)
            
        Returns:
            Tuple of (count of reviews mentioning support, percentage)
        """
        support_count = 0
        
        for text in texts:
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in self.CUSTOMER_SUPPORT_KEYWORDS):
                support_count += 1
            elif 'support' in text_lower and ('custom' in text_lower or 'help' in text_lower):
                support_count += 1
            elif 'service' in text_lower and ('custom' in text_lower or 'poor' in text_lower or 
                                              'bad' in text_lower or 'terrible' in text_lower):
                support_count += 1
        
        percentage = (support_count / len(texts)) * 100 if texts else 0
        return support_count, percentage
    
    def get_root_cause_summary(self, negative_texts: List[str], 
                                all_negative_count: int) -> Dict:
        """
        Generate comprehensive root-cause summary for negative reviews.
        
        Args:
            negative_texts: List of preprocessed negative review texts
            all_negative_count: Total count of negative reviews
            
        Returns:
            Dictionary containing root-cause analysis results
        """
        keyword_analysis = self.analyze_negative_reviews(negative_texts, top_n=15)
        support_count, support_percentage = self.detect_customer_support_mentions(negative_texts)
        
        combined_keywords = []
        seen = set()
        
        for phrase, count in keyword_analysis['top_bigrams']:
            if phrase not in seen:
                combined_keywords.append((phrase, int(count)))
                seen.add(phrase)
        
        for phrase, count in keyword_analysis['top_trigrams']:
            if phrase not in seen and count > 1:
                combined_keywords.append((phrase, int(count)))
                seen.add(phrase)
        
        combined_keywords.sort(key=lambda x: x[1], reverse=True)
        
        is_support_top = False
        support_rank = -1
        for i, (phrase, _) in enumerate(combined_keywords):
            if any(kw in phrase for kw in ['customer support', 'customer service', 
                                           'support team', 'service represent']):
                is_support_top = True
                support_rank = i + 1
                break
        
        if not is_support_top and combined_keywords:
            if support_count > 0:
                combined_keywords.insert(0, ('customer support', int(support_count)))
        
        return {
            'total_negative_reviews': int(all_negative_count),
            'reviews_mentioning_support': int(support_count),
            'support_mention_percentage': float(round(support_percentage, 2)),
            'top_root_cause_keywords': combined_keywords[:10],
            'customer_support_is_primary_issue': bool(is_support_top or support_count > 0),
            'author': self.AUTHOR
        }
