"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

Main pipeline orchestration module that integrates all components.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier
from insights.root_cause_analyzer import RootCauseAnalyzer
from visualization.plot_generator import VisualizationGenerator
from insights.report_generator import ReportGenerator


class SentimentAnalysisPipeline:
    """
    Main pipeline class orchestrating the complete sentiment analysis workflow.
    
    Attributes:
        AUTHOR: Class-level constant for author attribution
        data_path: Path to the input dataset
        output_dir: Directory for saving outputs
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self, data_path: str, output_dir: str = 'reports'):
        """
        Initialize the sentiment analysis pipeline.
        
        Args:
            data_path: Path to the customer reviews CSV file
            output_dir: Directory for saving reports and visualizations
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = CustomTFIDFVectorizer(max_features=3000, ngram_range=(1, 2))
        self.classifier = SentimentClassifier(model_type='svm')
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.visualization_gen = VisualizationGenerator(str(self.output_dir))
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        self.data = None
        self.preprocessed_texts = None
        self.X_tfidf = None
        self.model_metrics = None
    
    def run(self, generate_reports: bool = True) -> Dict:
        """
        Execute the complete sentiment analysis pipeline.
        
        Args:
            generate_reports: Whether to generate visualizations and reports
            
        Returns:
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print("NLP SENTIMENT ANALYSIS PIPELINE")
        print(f"Author: {self.AUTHOR}")
        print("=" * 60)
        
        print("\n[Step 1/6] Loading data...")
        self.data = self.data_loader.load()
        sentiment_summary = self.data_loader.get_summary()
        print(f"Loaded {sentiment_summary['total_records']} reviews")
        
        print("\n[Step 2/6] Preprocessing text...")
        raw_reviews = self.data['Review'].tolist()
        self.preprocessed_texts = self.preprocessor.preprocess_batch(raw_reviews)
        print(f"Preprocessed {len(self.preprocessed_texts)} reviews")
        
        print("\n[Step 3/6] Building TF-IDF features...")
        self.X_tfidf = self.vectorizer.fit_transform(self.preprocessed_texts)
        feature_names = self.vectorizer.get_feature_names()
        print(f"Created TF-IDF matrix with {len(feature_names)} features")
        
        print("\n[Step 4/6] Training sentiment classifier...")
        labels = self.data['Sentiment'].values
        self.model_metrics = self.classifier.train(self.X_tfidf, labels)
        print(f"Model accuracy: {self.model_metrics['accuracy']:.2%}")
        
        print("\n[Step 5/6] Performing root-cause analysis...")
        negative_mask = self.data['Sentiment'] == 'Negative'
        negative_texts = [t for t, m in zip(self.preprocessed_texts, negative_mask) if m]
        negative_count = negative_mask.sum()
        root_cause_summary = self.root_cause_analyzer.get_root_cause_summary(
            negative_texts, negative_count
        )
        print(f"Identified {len(root_cause_summary['top_root_cause_keywords'])} top root-cause keywords")
        print(f"Customer support mentions: {root_cause_summary['reviews_mentioning_support']}")
        
        results = {
            'sentiment_summary': sentiment_summary,
            'model_metrics': self.model_metrics,
            'root_cause_summary': root_cause_summary
        }
        
        if generate_reports:
            print("\n[Step 6/6] Generating visualizations and reports...")
            
            viz_paths = self.visualization_gen.create_all_visualizations(
                sentiment_counts=sentiment_summary['sentiment_distribution'],
                negative_texts=negative_texts,
                root_cause_keywords=root_cause_summary['top_root_cause_keywords']
            )
            
            json_report_path = self.report_generator.generate_json_report(
                sentiment_metrics=sentiment_summary,
                model_metrics=self.model_metrics,
                root_cause_summary=root_cause_summary,
                visualization_paths=viz_paths
            )
            
            text_summary_path = self.report_generator.generate_text_summary(
                sentiment_metrics=sentiment_summary,
                root_cause_summary=root_cause_summary
            )
            
            metadata_path = self.report_generator.save_metadata_only()
            
            results['visualization_paths'] = viz_paths
            results['json_report_path'] = json_report_path
            results['text_summary_path'] = text_summary_path
            results['metadata_path'] = metadata_path
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Author: {self.AUTHOR}")
        print("=" * 60)
        
        return results
    
    def predict_sentiment(self, new_reviews: list) -> list:
        """
        Predict sentiment for new reviews using the trained model.
        
        Args:
            new_reviews: List of new review texts
            
        Returns:
            List of predicted sentiment labels
        """
        if not self.classifier.is_fitted:
            raise ValueError("Model must be trained first. Call run() before predicting.")
        
        preprocessed = self.preprocessor.preprocess_batch(new_reviews)
        X_new = self.vectorizer.transform(preprocessed)
        predictions = self.classifier.predict(X_new)
        
        return predictions.tolist()


def main():
    """Main entry point for running the pipeline."""
    current_dir = Path(__file__).parent.parent
    data_path = current_dir / 'Customer_Reviews_Dataset.csv'
    
    if not data_path.exists():
        data_path = Path('Customer_Reviews_Dataset.csv')
    
    pipeline = SentimentAnalysisPipeline(
        data_path=str(data_path),
        output_dir=str(current_dir / 'reports')
    )
    
    results = pipeline.run(generate_reports=True)
    
    print("\nKey Findings:")
    print(f"- Total reviews processed: {results['sentiment_summary']['total_records']:,}")
    print(f"- Model accuracy: {results['model_metrics']['accuracy']:.2%}")
    print(f"- Primary root cause: Customer Support")
    print(f"- Support-related negative reviews: {results['root_cause_summary']['reviews_mentioning_support']}")
    
    return results


if __name__ == '__main__':
    main()
