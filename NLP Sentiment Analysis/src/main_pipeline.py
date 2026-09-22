"""
Leakage-safe NLP sentiment analysis and customer feedback analytics pipeline.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier
from insights.theme_analyzer import ThemeAnalyzer
from visualization.plot_generator import VisualizationGenerator
from insights.report_generator import ReportGenerator


class SentimentAnalysisPipeline:
    """Orchestrates leakage-safe modeling and descriptive customer-feedback analysis."""

    def __init__(self, data_path: str, output_dir: str = "reports", random_state: int = 42):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.data_loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = CustomTFIDFVectorizer(max_features=3000, ngram_range=(1, 2))
        self.analyzer = ThemeAnalyzer()
        self.visualization_gen = VisualizationGenerator(str(self.output_dir))
        self.report_generator = ReportGenerator(str(self.output_dir))

    def _prepare_model_data(self):
        labels = self.data["Sentiment"].to_numpy()
        texts = self.preprocessed_texts
        train_texts, test_texts, y_train_raw, y_test_raw = train_test_split(
            texts, labels, test_size=0.20, random_state=self.random_state, stratify=labels
        )
        # Leakage control: vocabulary and IDF are fitted on training text only.
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        label_helper = SentimentClassifier()
        return X_train, X_test, label_helper.prepare_labels(y_train_raw), label_helper.prepare_labels(y_test_raw)

    def _model_comparison(self, X_train, X_test, y_train, y_test) -> Dict:
        comparison = {}
        majority_class = int(pd.Series(y_train).mode().iloc[0])
        majority_pred = [majority_class] * len(y_test)
        comparison["majority_baseline"] = {
            "model": "majority_baseline",
            "accuracy": float((pd.Series(y_test).to_numpy() == majority_class).mean()),
            "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
            "f1_weighted": 0.0, "roc_auc_macro": None, "pr_auc_macro": None,
            "train_size": int(len(y_train)), "test_size": int(len(y_test)),
            "confusion_matrix": [],
            "classification_report": {},
        }
        for model_type in ["naive_bayes", "logistic_regression", "svm"]:
            model = SentimentClassifier(model_type=model_type, random_state=self.random_state)
            comparison[model_type] = model.fit_and_evaluate(X_train, X_test, y_train, y_test)
        return comparison

    def run(self, generate_reports: bool = True) -> Dict:
        self.data = self.data_loader.load()
        sentiment_summary = self.data_loader.get_summary()
        self.preprocessed_texts = self.preprocessor.preprocess_batch(
            self.data["Review"].fillna("").astype(str).tolist()
        )

        duplicate_summary = self.data_loader.get_duplicate_summary()
        category_sentiment = self.data_loader.get_category_sentiment_summary()

        X_train, X_test, y_train, y_test = self._prepare_model_data()
        model_comparison = self._model_comparison(X_train, X_test, y_train, y_test)

        theme_analysis = self.analyzer.analyze_themes(
            self.data["Review"].fillna("").astype(str).tolist(),
            self.data["Sentiment"].tolist(),
            self.data["Category"].tolist(),
        )
        negative_texts = [
            text for text, sentiment in zip(self.preprocessed_texts, self.data["Sentiment"])
            if sentiment == "Negative"
        ]

        results = {
            "sentiment_summary": sentiment_summary,
            "data_quality": {"duplicate_summary": duplicate_summary, "category_sentiment": category_sentiment},
            "model_comparison": model_comparison,
            "theme_analysis": theme_analysis,
            "evaluation_design": {
                "split_strategy": "Stratified 80/20 train/test split",
                "tfidf_fit_scope": "Training set only",
                "random_state": self.random_state,
                "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro", "pr_auc_macro"],
            },
        }

        if generate_reports:
            viz_paths = self.visualization_gen.create_all_visualizations(
                sentiment_counts=sentiment_summary["sentiment_distribution"],
                negative_texts=negative_texts,
                root_cause_keywords=theme_analysis["top_negative_phrases"],
            )
            results["visualization_paths"] = viz_paths
            results["json_report_path"] = self.report_generator.generate_json_report(
                sentiment_metrics=sentiment_summary,
                model_metrics=model_comparison,
                theme_analysis=theme_analysis,
                data_quality=results["data_quality"],
                evaluation_design=results["evaluation_design"],
                visualization_paths=viz_paths,
            )
            results["text_summary_path"] = self.report_generator.generate_text_summary(
                sentiment_metrics=sentiment_summary,
                model_metrics=model_comparison,
                theme_analysis=theme_analysis,
                data_quality=results["data_quality"],
            )
        return results


def main():
    current_dir = Path(__file__).parent.parent
    pipeline = SentimentAnalysisPipeline(
        data_path=str(current_dir / "Customer_Reviews_Dataset.csv"),
        output_dir=str(current_dir / "reports"),
    )
    results = pipeline.run(generate_reports=True)
    print("\nModel comparison:")
    for name, metrics in results["model_comparison"].items():
        print(f"- {name}: Accuracy={metrics['accuracy']:.3f}, Macro-F1={metrics['f1_macro']:.3f}")
    print(f"\nSentiment distribution: {results['sentiment_summary']['sentiment_distribution']}")
    print(f"Exact duplicate diagnostics: {results['data_quality']['duplicate_summary']}")
    print("Top negative-review phrases:")
    for phrase, count in results["theme_analysis"]["top_negative_phrases"][:10]:
        print(f"- {phrase}: {count}")
    return results


if __name__ == "__main__":
    main()
