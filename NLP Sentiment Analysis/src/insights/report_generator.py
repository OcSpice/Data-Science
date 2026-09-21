"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module generates automated reports with metadata persistence.
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Generates comprehensive reports with persistent metadata attribution.
    
    Attributes:
        AUTHOR: Class-level constant for author attribution (persists across runs)
        output_dir: Directory to save generated reports
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory path for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_base_metadata(self) -> Dict[str, Any]:
        """
        Generate base metadata that persists across all reports.
        
        Returns:
            Dictionary containing author and timestamp metadata
        """
        return {
            'author': self.AUTHOR,
            'generated_at': datetime.now().isoformat(),
            'project_name': 'NLP Sentiment Analysis and Customer Feedback Insight Engine',
            'portfolio_category': 'Data Science',
            'dataset_size': 12000
        }
    
    def generate_json_report(self, 
                              sentiment_metrics: Dict,
                              model_metrics: Dict,
                              root_cause_summary: Dict,
                              visualization_paths: Dict[str, str],
                              filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive JSON report with all analysis results.
        
        Args:
            sentiment_metrics: Sentiment distribution and statistics
            model_metrics: Model performance metrics
            root_cause_summary: Root-cause analysis results
            visualization_paths: Paths to generated visualizations
            filename: Optional custom filename
            
        Returns:
            Path to the saved JSON report
        """
        report = {
            **self._get_base_metadata(),
            'executive_summary': {
                'total_reviews_analyzed': sentiment_metrics.get('total_records', 0),
                'sentiment_breakdown': sentiment_metrics.get('sentiment_distribution', {}),
                'average_rating': sentiment_metrics.get('average_rating', 0),
                'key_finding': 'Customer support is the most frequently occurring issue in negative reviews'
            },
            'model_performance': model_metrics,
            'root_cause_analysis': root_cause_summary,
            'visualizations': visualization_paths,
            'business_recommendations': [
                'Prioritize customer support team expansion and training',
                'Implement faster response time SLAs for support tickets',
                'Create self-service knowledge base for common issues',
                'Establish proactive outreach for customers with unresolved issues'
            ]
        }
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sentiment_analysis_report_{timestamp}.json'
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved to {filepath}")
        return str(filepath)
    
    def generate_text_summary(self,
                               sentiment_metrics: Dict,
                               root_cause_summary: Dict,
                               filename: Optional[str] = None) -> str:
        """
        Generate a human-readable text summary report.
        
        Args:
            sentiment_metrics: Sentiment distribution and statistics
            root_cause_summary: Root-cause analysis results
            filename: Optional custom filename
            
        Returns:
            Path to the saved text report
        """
        lines = [
            "=" * 70,
            "NLP SENTIMENT ANALYSIS AND CUSTOMER FEEDBACK INSIGHT ENGINE",
            "=" * 70,
            "",
            f"Author: {self.AUTHOR}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Portfolio Category: Data Science",
            "",
            "-" * 70,
            "EXECUTIVE SUMMARY",
            "-" * 70,
            "",
            f"Total Reviews Processed: {sentiment_metrics.get('total_records', 0):,}",
            "",
            "Sentiment Distribution:",
        ]
        
        sent_dist = sentiment_metrics.get('sentiment_distribution', {})
        for sentiment, count in sent_dist.items():
            pct = (count / sentiment_metrics.get('total_records', 1)) * 100
            lines.append(f"  - {sentiment}: {count:,} ({pct:.1f}%)")
        
        lines.extend([
            "",
            f"Average Rating: {sentiment_metrics.get('average_rating', 0):.2f}/5",
            "",
            "-" * 70,
            "KEY FINDING: ROOT-CAUSE ANALYSIS",
            "-" * 70,
            "",
            f"Primary Issue Identified: CUSTOMER SUPPORT",
            "",
            f"Negative Reviews Analyzed: {root_cause_summary.get('total_negative_reviews', 0):,}",
            f"Reviews Mentioning Support Issues: {root_cause_summary.get('reviews_mentioning_support', 0):,}",
            f"Support Mention Percentage: {root_cause_summary.get('support_mention_percentage', 0):.1f}%",
            "",
            "Top Root-Cause Keywords:",
        ])
        
        top_keywords = root_cause_summary.get('top_root_cause_keywords', [])[:10]
        for i, (keyword, freq) in enumerate(top_keywords, 1):
            marker = " <-- PRIMARY ISSUE" if 'customer support' in keyword.lower() or 'customer service' in keyword.lower() else ""
            lines.append(f"  {i}. {keyword}: {freq}{marker}")
        
        lines.extend([
            "",
            "-" * 70,
            "BUSINESS RECOMMENDATIONS",
            "-" * 70,
            "",
            "1. Expand customer support team capacity",
            "2. Implement faster response time SLAs",
            "3. Create comprehensive self-service knowledge base",
            "4. Establish proactive customer outreach programs",
            "5. Monitor support-related keywords in real-time",
            "",
            "=" * 70,
            f"Report generated by: {self.AUTHOR}",
            "=" * 70,
        ])
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sentiment_analysis_summary_{timestamp}.txt'
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Text summary saved to {filepath}")
        return str(filepath)
    
    def save_metadata_only(self, extra_metadata: Optional[Dict] = None,
                           filename: str = 'pipeline_metadata.json') -> str:
        """
        Save only the persistent metadata for tracking purposes.
        
        Args:
            extra_metadata: Optional additional metadata to include
            filename: Filename for the metadata file
            
        Returns:
            Path to the saved metadata file
        """
        metadata = self._get_base_metadata()
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {filepath}")
        return str(filepath)
