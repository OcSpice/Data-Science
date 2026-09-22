"""End-to-end NLP sentiment analysis pipeline for the Women's Clothing Reviews dataset."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier, LABELS


class SentimentAnalysisPipeline:
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    RANDOM_STATE = 42
    MODEL_TYPES = ["majority", "naive_bayes", "logistic_regression", "linear_svm"]

    def __init__(self, data_path: str, output_dir: str = "reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = CustomTFIDFVectorizer()
        self.models = {}
        self.results = {}

    def run(self) -> dict:
        print("=" * 68)
        print("NLP SENTIMENT ANALYSIS PIPELINE")
        print(f"Author: {self.AUTHOR}")
        print("=" * 68)

        print("\n[1/5] Loading and auditing data...")
        data = self.data_loader.load()
        audit = self.data_loader.get_summary()
        print(f"Raw rows: {audit['raw_rows']:,}")
        print(f"Final modeling rows: {audit['final_rows']:,}")

        texts = data["Text"].fillna("").tolist()
        labels = data["Sentiment"].tolist()

        print("\n[2/5] Preprocessing review text...")
        processed = self.preprocessor.preprocess_batch(texts)

        print("\n[3/5] Stratified train/test split...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed,
            labels,
            test_size=0.20,
            random_state=self.RANDOM_STATE,
            stratify=labels,
        )
        print(f"Training rows: {len(X_train_text):,}")
        print(f"Testing rows: {len(X_test_text):,}")

        print("\n[4/5] Fitting TF-IDF on training data only...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        print(f"TF-IDF features: {X_train.shape[1]:,}")
        print(f"Sparse matrix shape: {X_train.shape}")

        print("\n[5/5] Training and evaluating models...")
        y_test_array = np.asarray(y_test)
        for model_type in self.MODEL_TYPES:
            classifier = SentimentClassifier(
                model_type=model_type, random_state=self.RANDOM_STATE
            )
            classifier.fit(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test_array)
            self.models[model_type] = classifier
            self.results[model_type] = metrics
            print(
                f"{model_type:20s} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

        output = {
            "dataset_audit": audit,
            "split": {
                "random_state": self.RANDOM_STATE,
                "test_size": 0.20,
                "train_rows": len(X_train_text),
                "test_rows": len(X_test_text),
            },
            "tfidf": {
                "features": X_train.shape[1],
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
            },
            "labels": LABELS,
            "models": self.results,
        }

        report_path = self.output_dir / "model_evaluation.json"
        report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nSaved evaluation report: {report_path}")

        return output

    def predict_sentiment(self, new_reviews: list, model_type: str = "linear_svm") -> list:
        if model_type not in self.models:
            raise ValueError("Run the pipeline before prediction.")
        processed = self.preprocessor.preprocess_batch(new_reviews)
        X_new = self.vectorizer.transform(processed)
        return self.models[model_type].predict(X_new).tolist()


def main():
    current_dir = Path(__file__).parent.parent
    data_path = current_dir / "data" / "Womens Clothing E-Commerce Reviews.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    pipeline = SentimentAnalysisPipeline(
        data_path=str(data_path),
        output_dir=str(current_dir / "reports"),
    )
    results = pipeline.run()

    print("\nMacro-F1 comparison:")
    for name, metrics in results["models"].items():
        print(f"- {name}: {metrics['macro_f1']:.4f}")

    return results


if __name__ == "__main__":
    main()
