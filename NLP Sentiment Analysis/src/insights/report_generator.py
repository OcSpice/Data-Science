"""Generate reproducible, methodology-aware sentiment analysis reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _metadata(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(),
            "project_name": "NLP Sentiment Analysis & Customer Feedback Analytics",
            "portfolio_category": "Data Science",
        }

    def generate_json_report(self, sentiment_metrics: Dict, model_metrics: Dict,
                             theme_analysis: Dict, data_quality: Dict,
                             evaluation_design: Dict, visualization_paths: Dict[str, str],
                             filename: Optional[str] = None) -> str:
        report = {
            **self._metadata(),
            "executive_summary": {
                "total_reviews_analyzed": sentiment_metrics.get("total_records", 0),
                "sentiment_breakdown": sentiment_metrics.get("sentiment_distribution", {}),
                "average_rating": sentiment_metrics.get("average_rating", 0),
            },
            "evaluation_design": evaluation_design,
            "model_comparison": model_metrics,
            "data_quality": data_quality,
            "theme_analysis": theme_analysis,
            "visualizations": visualization_paths,
            "interpretation_guardrails": [
                "TF-IDF vocabulary and IDF statistics are fitted on training text only.",
                "Macro-F1 is emphasized for multiclass comparison.",
                "Theme frequency is descriptive and does not establish causality.",
                "Results are specific to this dataset and should not be generalized without validation.",
            ],
        }
        filename = filename or f"sentiment_analysis_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(filepath)

    def generate_text_summary(self, sentiment_metrics: Dict, model_metrics: Dict,
                              theme_analysis: Dict, data_quality: Dict,
                              filename: Optional[str] = None) -> str:
        lines = [
            "NLP SENTIMENT ANALYSIS & CUSTOMER FEEDBACK ANALYTICS",
            "=" * 65,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Total reviews: {sentiment_metrics.get('total_records', 0):,}",
            f"Average rating: {sentiment_metrics.get('average_rating', 0):.2f}/5",
            "", "SENTIMENT DISTRIBUTION", "-" * 65,
        ]
        total = sentiment_metrics.get("total_records", 1)
        for label, count in sentiment_metrics.get("sentiment_distribution", {}).items():
            lines.append(f"- {label}: {count:,} ({count/total:.1%})")
        lines.extend(["", "MODEL COMPARISON", "-" * 65])
        for name, metrics in model_metrics.items():
            lines.append(f"- {name}: Accuracy={metrics.get('accuracy',0):.3f}; Macro-F1={metrics.get('f1_macro',0):.3f}; Weighted-F1={metrics.get('f1_weighted',0):.3f}")
        lines.extend(["", "DATA QUALITY", "-" * 65, json.dumps(data_quality.get("duplicate_summary", {}), indent=2)])
        lines.extend(["", "CUSTOMER FEEDBACK THEMES", "-" * 65])
        for row in theme_analysis.get("themes", [])[:8]:
            lines.append(f"- {row['theme']}: {row['review_count']} reviews; {row['negative_pct_within_theme']:.1f}% negative within theme")
        lines.extend(["", "INTERPRETATION NOTE", "-" * 65, theme_analysis.get("interpretation_note", "")])
        filename = filename or f"sentiment_analysis_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
        filepath = self.output_dir / filename
        filepath.write_text("\n".join(lines), encoding="utf-8")
        return str(filepath)
