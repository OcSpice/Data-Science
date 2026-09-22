"""Visualization utilities for sentiment and customer-feedback theme analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from wordcloud import WordCloud


class VisualizationGenerator:
    """Generate portfolio-ready descriptive visualizations."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_sentiment_distribution(self, sentiment_counts: Dict[str, int], save_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        bars = ax.bar(sentiments, counts, edgecolor="black", linewidth=1.0)
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Customer Review Sentiment Distribution")
        total = sum(counts)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count}\n({count / total:.1%})", ha="center", va="bottom")
        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "sentiment_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Sentiment distribution plot saved to {save_path}")
        return save_path

    def generate_negative_word_cloud(self, negative_texts: List[str], save_path: Optional[str] = None) -> str:
        text = " ".join(negative_texts)
        if not text.strip():
            raise ValueError("No text provided for word cloud generation")
        wordcloud = WordCloud(width=800, height=600, background_color="white",
                              max_words=100, min_font_size=10).generate(text)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud: Negative Reviews")
        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "negative_reviews_wordcloud.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Word cloud saved to {save_path}")
        return save_path

    def plot_top_themes(self, themes: List[Tuple[str, int]], save_path: Optional[str] = None) -> str:
        top_10 = themes[:10]
        if not top_10:
            raise ValueError("No themes provided for plotting")
        labels = [item[0] for item in top_10]
        frequencies = [item[1] for item in top_10]
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(labels, frequencies, edgecolor="black", linewidth=1.0)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Customer Feedback Theme")
        ax.set_title("Top Customer Feedback Themes in Negative Reviews")
        ax.invert_yaxis()
        for bar, freq in zip(bars, frequencies):
            ax.text(freq + max(frequencies) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(freq), va="center")
        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "top_feedback_themes.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feedback theme plot saved to {save_path}")
        return save_path

    def create_all_visualizations(self, sentiment_counts: Dict[str, int],
                                  negative_texts: List[str],
                                  root_cause_keywords: List[Tuple[str, int]]) -> Dict[str, str]:
        """Backward-compatible wrapper; root_cause_keywords now means theme phrases."""
        return {
            "sentiment_distribution": self.plot_sentiment_distribution(sentiment_counts),
            "word_cloud": self.generate_negative_word_cloud(negative_texts),
            "feedback_themes": self.plot_top_themes(root_cause_keywords),
        }
