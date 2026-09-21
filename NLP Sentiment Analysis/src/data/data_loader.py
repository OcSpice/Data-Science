"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA

This module handles data loading, schema validation, and text quality checks.
"""

import pandas as pd
import re
from typing import Optional, Tuple, List
from pathlib import Path


class DataLoader:
    """
    Handles loading and validation of customer review datasets.
    
    Attributes:
        file_path: Path to the CSV file
        expected_columns: List of required column names
    """
    
    EXPECTED_COLUMNS = [
        'ReviewID', 'Product', 'Category', 'Source', 'Country',
        'Rating', 'Sentiment', 'Review', 'WordCount', 'CharCount', 'Topic'
    ]
    
    VALID_SENTIMENTS = {'Positive', 'Negative', 'Neutral'}
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If schema validation fails
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        self._validate_schema()
        return self.data
    
    def _validate_schema(self) -> None:
        """
        Validate that the loaded data has expected columns and types.
        
        Raises:
            ValueError: If schema validation fails
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        invalid_sentiments = set(self.data['Sentiment'].unique()) - self.VALID_SENTIMENTS
        if invalid_sentiments:
            raise ValueError(f"Invalid sentiment values found: {invalid_sentiments}")
        
        print(f"Schema validation passed. {len(self.data)} records loaded.")
    
    def get_summary(self) -> dict:
        """
        Generate summary statistics for the loaded dataset.
        
        Returns:
            dict: Summary statistics including record count, sentiment distribution
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        return {
            'total_records': len(self.data),
            'sentiment_distribution': self.data['Sentiment'].value_counts().to_dict(),
            'average_rating': float(self.data['Rating'].mean()),
            'categories': self.data['Category'].unique().tolist()
        }
