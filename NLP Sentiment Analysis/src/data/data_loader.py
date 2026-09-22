"""Data loading and leakage-safe preparation for the Women's Clothing Reviews dataset."""

from pathlib import Path
import re
from typing import Dict, Optional

import pandas as pd


class DataLoader:
    """Load, validate, deduplicate, and label the project dataset."""

    REQUIRED_COLUMNS = {
        "Clothing ID", "Age", "Title", "Review Text", "Rating",
        "Recommended IND", "Positive Feedback Count", "Division Name",
        "Department Name", "Class Name",
    }

    SENTIMENT_MAP = {
        1: "Negative", 2: "Negative",
        3: "Neutral",
        4: "Positive", 5: "Positive",
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None
        self.audit: Dict = {}

    @staticmethod
    def normalize_review(text: str) -> str:
        """Normalize text only for duplicate detection; preserve original text separately."""
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        df = pd.read_csv(self.file_path)
        self._validate_schema(df)

        raw_rows = len(df)
        missing_reviews = int(df["Review Text"].isna().sum())

        df = df.loc[df["Review Text"].notna()].copy()
        df["Review Text"] = df["Review Text"].astype(str).str.strip()
        df = df.loc[df["Review Text"].ne("")].copy()

        df["normalized_review"] = df["Review Text"].map(self.normalize_review)
        before_dedup = len(df)
        df = df.drop_duplicates(subset="normalized_review", keep="first").copy()
        duplicate_rows_removed = before_dedup - len(df)

        df["Sentiment"] = df["Rating"].map(self.SENTIMENT_MAP)
        if df["Sentiment"].isna().any():
            raise ValueError("Unexpected rating values found while creating sentiment labels.")

        # Keep the raw review and a modeling text column. Title is supplementary context,
        # while normalized_review is retained only for audit/deduplication checks.
        df["Text"] = df["Review Text"]
        title_mask = df["Title"].notna() & df["Title"].astype(str).str.strip().ne("")
        df.loc[title_mask, "Text"] = (
            df.loc[title_mask, "Title"].astype(str).str.strip()
            + ". "
            + df.loc[title_mask, "Review Text"].astype(str).str.strip()
        )

        self.data = df.reset_index(drop=True)
        self.audit = {
            "raw_rows": raw_rows,
            "missing_review_text": missing_reviews,
            "rows_after_missing_text_removal": before_dedup,
            "duplicate_review_rows_removed": duplicate_rows_removed,
            "final_rows": len(self.data),
            "sentiment_distribution": self.data["Sentiment"].value_counts().to_dict(),
            "rating_distribution": self.data["Rating"].value_counts().sort_index().to_dict(),
        }
        return self.data

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        ratings = set(pd.to_numeric(df["Rating"], errors="coerce").dropna().astype(int).unique())
        invalid = ratings - set(self.SENTIMENT_MAP)
        if invalid:
            raise ValueError(f"Unsupported rating values: {sorted(invalid)}")

    def get_summary(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        summary = dict(self.audit)
        summary["average_rating"] = float(self.data["Rating"].mean())
        summary["unique_clothing_ids"] = int(self.data["Clothing ID"].nunique())
        return summary
