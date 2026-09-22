"""Descriptive customer-feedback theme analysis.

Theme frequency and sentiment association describe observed patterns; they do
not establish causal root causes.
"""

import re
from collections import Counter
from typing import Dict, List


class ThemeAnalyzer:
    THEMES = {
        "Customer Support": ["customer support", "customer service", "support team", "service representative", "help desk", "support staff", "customer care", "support agent"],
        "Billing": ["billing", "charged", "charge", "invoice", "payment", "subscription", "refund", "overcharged"],
        "Product Quality": ["quality", "defective", "broken", "broke", "damaged", "poor quality"],
        "Performance": ["slow", "performance", "crash", "crashed", "lag", "unresponsive", "speed"],
        "Integration": ["integration", "integrate", "api", "sync", "synchronization", "connect", "connection"],
        "Pricing": ["price", "pricing", "expensive", "cost", "fee", "fees", "cheap", "value"],
        "Usability": ["easy to use", "difficult", "confusing", "usability", "interface", "ui", "navigation", "user friendly"],
        "Marketing Expectations": ["advertised", "advertisement", "marketing", "promised", "expectation", "expected", "misleading", "like advertised"],
    }

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "had", "has", "have", "he", "her", "his", "i", "if", "in", "is", "it",
        "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the",
        "their", "there", "they", "this", "to", "too", "was", "we", "were", "what",
        "when", "which", "who", "with", "you", "your",
        "review", "product", "use", "work", "time", "day", "one", "like", "get",
        "go", "see", "know", "take",
    }

    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        words = re.findall(r"\b[a-z]+\b", str(text).lower())
        if len(words) < n:
            return []
        return [
            " ".join(words[i:i+n])
            for i in range(len(words)-n+1)
            if not any(word in self.STOPWORDS for word in words[i:i+n])
        ]

    def analyze_negative_reviews(self, negative_texts: List[str], top_n: int = 20) -> Dict:
        unigrams, bigrams, trigrams = Counter(), Counter(), Counter()
        for text in negative_texts:
            tokens = re.findall(r"\b[a-z]+\b", str(text).lower())
            unigrams.update(token for token in tokens if token not in self.STOPWORDS)
            bigrams.update(self.extract_ngrams(text, 2))
            trigrams.update(self.extract_ngrams(text, 3))

        top_unigrams = [
            x for x in unigrams.most_common(top_n * 2)
            if len(x[0]) > 2
        ][:top_n]

        return {
            "top_unigrams": top_unigrams,
            "top_bigrams": bigrams.most_common(top_n),
            "top_trigrams": trigrams.most_common(top_n),
        }

    def analyze_themes(self, reviews: List[str], sentiments: List[str], categories: List[str]) -> Dict:
        total = len(reviews)
        negative_total = sum(s == "Negative" for s in sentiments)
        rows = []
        for theme, keywords in self.THEMES.items():
            review_hits = negative_hits = 0
            category_counts = Counter()
            for review, sentiment, category in zip(reviews, sentiments, categories):
                text = str(review).lower()
                if any(keyword in text for keyword in keywords):
                    review_hits += 1
                    negative_hits += int(sentiment == "Negative")
                    category_counts[str(category)] += 1
            rows.append({
                "theme": theme,
                "review_count": review_hits,
                "review_pct": round(review_hits/total*100, 2) if total else 0.0,
                "negative_review_count": negative_hits,
                "negative_pct_within_theme": round(negative_hits/review_hits*100, 2) if review_hits else 0.0,
                "negative_share": round(negative_hits/negative_total*100, 2) if negative_total else 0.0,
                "top_categories": dict(category_counts.most_common(5)),
            })

        rows.sort(key=lambda x: x["negative_share"], reverse=True)
        negative_texts = [str(r) for r, s in zip(reviews, sentiments) if s == "Negative"]
        keyword_analysis = self.analyze_negative_reviews(negative_texts, top_n=15)

        return {
            "total_reviews": total,
            "negative_reviews": negative_total,
            "themes": rows,
            "top_negative_phrases": keyword_analysis["top_bigrams"][:10],
            "interpretation_note": "Theme frequency and sentiment association are descriptive. They do not establish causal root causes.",
        }
