"""Dataset loading, schema validation and descriptive quality checks."""

from pathlib import Path
from typing import Optional
import re
import pandas as pd


class DataLoader:
    EXPECTED_COLUMNS = ["ReviewID", "Product", "Category", "Source", "Country", "Rating", "Sentiment", "Review", "WordCount", "CharCount", "Topic"]
    VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        self.data = pd.read_csv(self.file_path)
        self._validate_schema()
        return self.data

    def _validate_schema(self) -> None:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        missing = set(self.EXPECTED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        invalid = set(self.data["Sentiment"].dropna().unique()) - self.VALID_SENTIMENTS
        if invalid:
            raise ValueError(f"Invalid sentiment values found: {invalid}")
        if self.data["Review"].isna().any():
            self.data["Review"] = self.data["Review"].fillna("")
        if self.data["Sentiment"].isna().any():
            raise ValueError("Sentiment contains missing values.")

    def get_summary(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        return {
            "total_records": int(len(self.data)),
            "sentiment_distribution": {str(k): int(v) for k, v in self.data["Sentiment"].value_counts().to_dict().items()},
            "average_rating": float(self.data["Rating"].mean()),
            "categories": self.data["Category"].dropna().unique().tolist(),
        }

    @staticmethod
    def _normalize_template(text: str) -> str:
        """Normalize superficial variation to expose repeated review templates."""
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", " <num> ", text)
        text = re.sub(r"[^a-z0-9<>\s]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def get_duplicate_summary(self, text_column: str = "Review", label_column: str = "Sentiment") -> dict:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        texts = self.data[text_column].fillna("").astype(str).str.strip()
        duplicate_mask = texts.duplicated(keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        unique_duplicate_texts = int(texts[duplicate_mask].nunique())
        conflicting = 0
        for _, group in self.data.loc[duplicate_mask].groupby(text_column, dropna=False):
            if group[label_column].nunique() > 1:
                conflicting += 1

        normalized = texts.map(self._normalize_template)
        template_counts = normalized.value_counts()
        repeated_template_rows = int(normalized[normalized.map(template_counts) > 1].size)
        repeated_template_count = int((template_counts > 1).sum())

        return {
            "exact_duplicate_rows": duplicate_rows,
            "unique_duplicate_texts": unique_duplicate_texts,
            "duplicate_row_pct": round(duplicate_rows / len(self.data) * 100, 2),
            "duplicate_texts_with_conflicting_labels": int(conflicting),
            "normalized_repeated_template_rows": repeated_template_rows,
            "normalized_repeated_template_count": repeated_template_count,
            "normalized_repeated_template_row_pct": round(repeated_template_rows / len(self.data) * 100, 2),
        }

    def get_category_sentiment_summary(self) -> list:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        table = pd.crosstab(self.data["Category"], self.data["Sentiment"], normalize="index").mul(100).round(2).reset_index()
        return table.to_dict(orient="records")
