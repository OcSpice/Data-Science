"""
Data Loader Module

Handles loading of customer churn dataset from CSV files.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """
    Responsible for loading the customer churn dataset.
    
    Attributes:
        data_path (Path): Path to the dataset file.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader with the path to the dataset.
        
        Args:
            data_path: Path to the CSV file containing customer churn data.
        """
        self.data_path = Path(data_path)
        self._data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            pd.DataFrame: The loaded dataset.
            
        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the file is empty or has no columns.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self._data = pd.read_csv(self.data_path)
        
        if self._data.empty:
            raise ValueError("Dataset is empty")
        
        if len(self._data.columns) == 0:
            raise ValueError("Dataset has no columns")
        
        return self._data
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Return the loaded data if available."""
        return self._data
