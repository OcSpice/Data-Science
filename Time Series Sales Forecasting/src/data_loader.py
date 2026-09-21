"""
Data Loading Module for Time Series Sales Forecasting
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module handles loading, merging, and preprocessing of multiple CSV data sources
into a unified time series format ready for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and merging of retail sales forecasting data from multiple CSV files.
    
    The class intelligently merges train_*.csv files, calendar.csv, and sell_prices.csv
    to create a unified dataset with all features properly aligned.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the DataLoader with the path to the data directory.
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.train_data: Optional[pd.DataFrame] = None
        self.calendar_data: Optional[pd.DataFrame] = None
        self.sell_prices_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        
    def load_train_files(self) -> pd.DataFrame:
        """
        Load and concatenate all train_*.csv files.
        
        Returns:
            Concatenated DataFrame with all training data
        """
        train_files = sorted(self.data_dir.glob("train_*.csv"))
        
        if not train_files:
            raise FileNotFoundError(f"No train files found in {self.data_dir}")
        
        logger.info(f"Loading {len(train_files)} training files")
        
        dfs = []
        for file in train_files:
            logger.info(f"Loading {file.name}")
            df = pd.read_csv(file)
            dfs.append(df)
        
        self.train_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(self.train_data)} records from training files")
        
        return self.train_data
    
    def load_calendar(self) -> pd.DataFrame:
        """
        Load calendar.csv containing date-level features.
        
        Returns:
            DataFrame with calendar features
        """
        calendar_path = self.data_dir / "calendar.csv"
        
        if not calendar_path.exists():
            raise FileNotFoundError(f"calendar.csv not found in {self.data_dir}")
        
        logger.info("Loading calendar.csv")
        self.calendar_data = pd.read_csv(calendar_path)
        logger.info(f"Loaded {len(self.calendar_data)} calendar records")
        
        return self.calendar_data
    
    def load_sell_prices(self) -> pd.DataFrame:
        """
        Load sell_prices.csv containing item and store-level pricing information.
        
        Returns:
            DataFrame with pricing information
        """
        prices_path = self.data_dir / "sell_prices.csv"
        
        if not prices_path.exists():
            raise FileNotFoundError(f"sell_prices.csv not found in {self.data_dir}")
        
        logger.info("Loading sell_prices.csv")
        self.sell_prices_data = pd.read_csv(prices_path)
        logger.info(f"Loaded {len(self.sell_prices_data)} price records")
        
        return self.sell_prices_data
    
    def load_validation(self) -> pd.DataFrame:
        """
        Load validation.csv holdout set for model evaluation.
        
        Returns:
            DataFrame with validation data
        """
        val_path = self.data_dir / "validation.csv"
        
        if not val_path.exists():
            raise FileNotFoundError(f"validation.csv not found in {self.data_dir}")
        
        logger.info("Loading validation.csv")
        self.validation_data = pd.read_csv(val_path)
        logger.info(f"Loaded {len(self.validation_data)} validation records")
        
        return self.validation_data
    
    def load_test(self) -> pd.DataFrame:
        """
        Load test.csv for final predictions.
        
        Returns:
            DataFrame with test data
        """
        test_path = self.data_dir / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(f"test.csv not found in {self.data_dir}")
        
        logger.info("Loading test.csv")
        self.test_data = pd.read_csv(test_path)
        logger.info(f"Loaded {len(self.test_data)} test records")
        
        return self.test_data
    
    def merge_data(self, include_validation: bool = True) -> pd.DataFrame:
        """
        Merge all data sources into a unified time series format.
        
        This method performs intelligent merging of:
        1. Training data (all train_*.csv files)
        2. Calendar features (holidays, events, SNAP days)
        3. Sell prices (item-store level pricing)
        
        Args:
            include_validation: If True, include validation data in merged dataset
            
        Returns:
            Unified DataFrame with all features
        """
        if self.train_data is None:
            self.load_train_files()
        
        if self.calendar_data is None:
            self.load_calendar()
            
        if self.sell_prices_data is None:
            self.load_sell_prices()
        
        # Start with training data
        merged = self.train_data.copy()
        
        # Add validation data if requested
        if include_validation and self.validation_data is not None:
            val_df = self.validation_data.copy()
            merged = pd.concat([merged, val_df], ignore_index=True)
            logger.info(f"Included {len(val_df)} validation records")
        
        # Ensure date column is datetime
        if 'date' in merged.columns:
            merged['date'] = pd.to_datetime(merged['date'])
        
        # Merge with calendar data on date
        calendar_cols = ['date', 'd', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
                         'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                         'snap_CA', 'snap_TX', 'snap_WI', 'temperature', 'fuel_price',
                         'cpi', 'unemployment']
        
        available_cal_cols = [c for c in calendar_cols if c in self.calendar_data.columns]
        merged = merged.merge(
            self.calendar_data[available_cal_cols],
            on=['date'],
            how='left',
            suffixes=('', '_cal')
        )
        
        # Merge with sell prices on store_id, item_id, and wm_yr_wk
        price_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        available_price_cols = [c for c in price_cols if c in self.sell_prices_data.columns]
        
        merged = merged.merge(
            self.sell_prices_data[available_price_cols],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left',
            suffixes=('', '_price')
        )
        
        # Handle missing sell_price by forward filling within each item-store group
        merged = merged.sort_values(['item_id', 'store_id', 'date'])
        merged['sell_price'] = merged.groupby(['item_id', 'store_id'])['sell_price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        # Fill any remaining NaN values
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if merged[col].isnull().any():
                merged[col] = merged[col].fillna(0)
        
        self.merged_data = merged
        logger.info(f"Merged dataset shape: {merged.shape}")
        
        return merged
    
    def create_time_series_by_item_store(self, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Structure data into individual time series for each item-store combination.
        
        Args:
            data: DataFrame to process. If None, uses merged_data.
            
        Returns:
            Dictionary mapping (item_id, store_id) tuples to time series DataFrames
        """
        if data is None:
            if self.merged_data is None:
                raise ValueError("No data loaded. Call merge_data first.")
            data = self.merged_data
        
        time_series_dict = {}
        
        for (item_id, store_id), group in data.groupby(['item_id', 'store_id']):
            ts = group.sort_values('date').reset_index(drop=True)
            time_series_dict[(item_id, store_id)] = ts
        
        logger.info(f"Created {len(time_series_dict)} individual time series")
        
        return time_series_dict
    
    def get_aggregated_sales(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get total daily sales across all items and stores.
        
        Args:
            data: DataFrame to aggregate. If None, uses merged_data.
            
        Returns:
            DataFrame with date and total sales
        """
        if data is None:
            if self.merged_data is None:
                raise ValueError("No data loaded. Call merge_data first.")
            data = self.merged_data
        
        aggregated = data.groupby('date')['sales'].sum().reset_index()
        aggregated.columns = ['date', 'total_sales']
        
        return aggregated
    
    def handle_missing_values(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset using appropriate strategies.
        
        Strategies:
        - Numeric columns: Forward fill then backward fill within groups
        - Categorical columns: Fill with mode or 'Unknown'
        
        Args:
            data: DataFrame to process. If None, uses merged_data.
            
        Returns:
            DataFrame with missing values handled
        """
        if data is None:
            if self.merged_data is None:
                raise ValueError("No data loaded. Call merge_data first.")
            data = self.merged_data.copy()
        
        # Handle numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                # Try group-based filling first
                if 'item_id' in data.columns and 'store_id' in data.columns:
                    data[col] = data.groupby(['item_id', 'store_id'])[col].transform(
                        lambda x: x.fillna(method='ffill').fillna(method='bfill')
                    )
                # Global filling for remaining NaNs
                data[col] = data[col].fillna(data[col].median())
        
        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0] if len(data[c]) > 0 else 'Unknown'
                data[col] = data[col].fillna(mode_val)
        
        logger.info(f"Missing value handling complete. Remaining NaNs: {data.isnull().sum().sum()}")
        
        return data


def load_all_data(data_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load and merge all data sources.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Tuple of (merged DataFrame, dictionary of time series by item-store)
    """
    loader = DataLoader(data_dir)
    loader.load_train_files()
    loader.load_calendar()
    loader.load_sell_prices()
    loader.load_validation()
    loader.load_test()
    
    merged = loader.merge_data()
    time_series_dict = loader.create_time_series_by_item_store()
    
    return merged, time_series_dict


if __name__ == "__main__":
    # Example usage
    data_directory = "../Datasets"
    loader = DataLoader(data_directory)
    
    merged_df = loader.merge_data()
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Columns: {merged_df.columns.tolist()}")
