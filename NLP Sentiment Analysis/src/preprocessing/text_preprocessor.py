"""Text cleaning utilities for customer-review NLP."""

import re
from typing import List


class TextPreprocessor:
    """Apply deterministic, model-friendly normalization to review text."""

    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"

    _html = re.compile(r"<[^>]+>")
    _url = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _phone = re.compile(r"\b(?:\+?\d[\d .()\-]{7,}\d)\b")
    _non_alpha = re.compile(r"[^a-zA-Z\s']")
    _whitespace = re.compile(r"\s+")

    def anonymize_pii(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self._email.sub(" EMAIL_REDACTED ", text)
        text = self._phone.sub(" PHONE_REDACTED ", text)
        return text

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self._html.sub(" ", text)
        text = self._url.sub(" ", text)
        text = self.anonymize_pii(text)
        text = text.lower()
        text = self._non_alpha.sub(" ", text)
        return self._whitespace.sub(" ", text).strip()

    def preprocess(self, text: str) -> str:
        return self.clean_text(text)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]
