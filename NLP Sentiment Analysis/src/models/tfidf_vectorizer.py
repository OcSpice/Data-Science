"""TF-IDF feature extraction for leakage-safe NLP modeling."""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


class CustomTFIDFVectorizer:
    """Thin wrapper around sklearn TF-IDF that preserves sparse matrices."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first.")
        return self.vectorizer.get_feature_names_out().tolist()

