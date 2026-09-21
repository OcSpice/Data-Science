"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module generates visualizations for sentiment analysis results.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class VisualizationGenerator:
    """
    Generates insight-driven visualizations for sentiment analysis.
    
    Attributes:
        AUTHOR: Class-level constant for author attribution
        output_dir: Directory to save generated visualizations
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory path for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_sentiment_distribution(self, sentiment_counts: Dict[str, int], 
                                     save_path: Optional[str] = None) -> str:
        """
        Create a bar chart showing sentiment distribution.
        
        Args:
            sentiment_counts: Dictionary mapping sentiment to count
            save_path: Optional path to save the figure
            
        Returns:
            Path to the saved or displayed figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        
        bars = ax.bar(sentiments, counts, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Sentiment Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax.set_title('Customer Review Sentiment Distribution\n(12,000 Reviews Analyzed)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{percentage:.1f}%', ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
        
        ax.text(0.5, -0.15, f'Author: {self.AUTHOR}', transform=ax.transAxes,
               ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'sentiment_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sentiment distribution plot saved to {save_path}")
        return save_path
    
    def generate_negative_word_cloud(self, negative_texts: List[str],
                                      save_path: Optional[str] = None) -> str:
        """
        Generate a word cloud from negative reviews.
        
        Args:
            negative_texts: List of preprocessed negative review texts
            save_path: Optional path to save the figure
            
        Returns:
            Path to the saved word cloud image
        """
        text = ' '.join(negative_texts)
        
        if not text.strip():
            raise ValueError("No text provided for word cloud generation")
        
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='Reds',
            max_words=100,
            contour_width=2,
            contour_color='#c0392b',
            min_font_size=10
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud: Negative Reviews\n(Key Pain Points)',
                    fontsize=16, fontweight='bold', pad=20)
        
        fig.text(0.5, 0.02, f'Author: {self.AUTHOR}', ha='center', 
                fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'negative_reviews_wordcloud.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word cloud saved to {save_path}")
        return save_path
    
    def plot_top_root_causes(self, keywords: List[Tuple[str, int]],
                              save_path: Optional[str] = None) -> str:
        """
        Create a bar chart showing top root-cause keywords.
        
        Args:
            keywords: List of (keyword, frequency) tuples
            save_path: Optional path to save the figure
            
        Returns:
            Path to the saved figure
        """
        top_10 = keywords[:10]
        
        if not top_10:
            raise ValueError("No keywords provided for plotting")
        
        phrases = [kw[0] for kw in top_10]
        frequencies = [kw[1] for kw in top_10]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#e74c3c' if 'customer support' in phrase.lower() or 
                  'customer service' in phrase.lower() else '#3498db' 
                  for phrase in phrases]
        
        bars = ax.barh(phrases, frequencies, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Root-Cause Keyword/Phrase', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Root-Cause Keywords Driving Negative Sentiment\n(Focus Area: Customer Support)',
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.invert_yaxis()
        
        for bar, freq in zip(bars, frequencies):
            ax.text(freq + max(frequencies)*0.01, bar.get_y() + bar.get_height()/2,
                   str(freq), va='center', fontsize=10, fontweight='bold')
        
        for i, phrase in enumerate(phrases):
            if 'customer support' in phrase.lower() or 'customer service' in phrase.lower():
                ax.add_patch(plt.Rectangle((-0.1, i-0.4), len(frequencies)+0.5, 0.8,
                                         fill=True, color='#e74c3c', alpha=0.1, zorder=0))
        
        ax.text(0.5, -0.12, f'Author: {self.AUTHOR}', transform=ax.transAxes,
               ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'top_root_causes.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Root-cause analysis plot saved to {save_path}")
        return save_path
    
    def create_all_visualizations(self, sentiment_counts: Dict[str, int],
                                   negative_texts: List[str],
                                   root_cause_keywords: List[Tuple[str, int]]) -> Dict[str, str]:
        """
        Generate all standard visualizations.
        
        Args:
            sentiment_counts: Dictionary of sentiment counts
            negative_texts: List of preprocessed negative review texts
            root_cause_keywords: List of (keyword, frequency) tuples
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        paths = {}
        
        paths['sentiment_distribution'] = self.plot_sentiment_distribution(sentiment_counts)
        paths['word_cloud'] = self.generate_negative_word_cloud(negative_texts)
        paths['root_causes'] = self.plot_top_root_causes(root_cause_keywords)
        
        return paths
