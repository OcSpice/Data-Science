"""
Feature Engineering Module for Time Series Sales Forecasting
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module creates robust features from the merged dataset including:
- Temporal features (day of week, month, quarter, etc.)
- Lag features for time series modeling
- Rolling window statistics
- Holiday and event indicators
- Price-based features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates comprehensive features for time series forecasting models.
    
    Implements feature engineering strategies specifically designed for
    retail sales forecasting with multiple hierarchical levels.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.feature_names: List[str] = []
        self.categorical_encodings: Dict[str, Dict] = {}
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Start of month/quarter/year indicators
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        temporal_features = [
            'day_of_week', 'day_of_month', 'day_of_year', 'month', 'quarter',
            'year', 'week_of_year', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start',
            'is_quarter_end'
        ]
        
        self.feature_names.extend([f for f in temporal_features if f not in self.feature_names])
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        lags: Optional[List[int]] = None,
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for the target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            lags: List of lag periods to create
            group_cols: Columns to group by before creating lags
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if lags is None:
            lags = [1, 2, 3, 7, 14, 28]
        
        if group_cols is None:
            group_cols = ['item_id', 'store_id']
        
        # Sort by date within groups
        if 'date' in df.columns:
            df = df.sort_values(group_cols + ['date'])
        
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            if group_cols:
                df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
            else:
                df[col_name] = df[target_col].shift(lag)
            
            self.feature_names.append(col_name)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        windows: Optional[List[int]] = None,
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            windows: List of window sizes
            group_cols: Columns to group by before creating rolling features
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if windows is None:
            windows = [7, 14, 28]
        
        if group_cols is None:
            group_cols = ['item_id', 'store_id']
        
        for window in windows:
            # Rolling mean
            mean_col = f'{target_col}_rolling_mean_{window}'
            if group_cols:
                df[mean_col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
            else:
                df[mean_col] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            self.feature_names.append(mean_col)
            
            # Rolling standard deviation
            std_col = f'{target_col}_rolling_std_{window}'
            if group_cols:
                df[std_col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
            else:
                df[std_col] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
            self.feature_names.append(std_col)
            
            # Rolling median
            median_col = f'{target_col}_rolling_median_{window}'
            if group_cols:
                df[median_col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).median()
                )
            else:
                df[median_col] = df[target_col].shift(1).rolling(window=window, min_periods=1).median()
            self.feature_names.append(median_col)
        
        return df
    
    def create_expanding_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create expanding window statistics (cumulative features).
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            group_cols: Columns to group by
            
        Returns:
            DataFrame with expanding features
        """
        df = df.copy()
        
        if group_cols is None:
            group_cols = ['item_id', 'store_id']
        
        # Expanding mean
        exp_mean_col = f'{target_col}_expanding_mean'
        if group_cols:
            df[exp_mean_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
        else:
            df[exp_mean_col] = df[target_col].shift(1).expanding(min_periods=1).mean()
        self.feature_names.append(exp_mean_col)
        
        # Expanding standard deviation
        exp_std_col = f'{target_col}_expanding_std'
        if group_cols:
            df[exp_std_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).std()
            )
        else:
            df[exp_std_col] = df[target_col].shift(1).expanding(min_periods=1).std()
        self.feature_names.append(exp_std_col)
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create holiday and event indicator features.
        
        Args:
            df: Input DataFrame with event columns from calendar
            
        Returns:
            DataFrame with holiday features
        """
        df = df.copy()
        
        # Event indicators
        df['is_event'] = (~df.get('event_name_1', pd.Series([None])).isna()).astype(int)
        df['is_national_event'] = (df.get('event_type_1', pd.Series([None])) == 'National').astype(int)
        df['is_religious_event'] = (df.get('event_type_1', pd.Series([None])) == 'Religious').astype(int)
        
        # SNAP (food stamp) days by state
        if 'snap_CA' in df.columns:
            df['is_snap_day'] = ((df['snap_CA'] == 1) | 
                                  (df.get('snap_TX', 0) == 1) | 
                                  (df.get('snap_WI', 0) == 1)).astype(int)
        else:
            df['is_snap_day'] = 0
        
        # Create specific holiday dummies
        holiday_events = ['NewYear', 'Thanksgiving', 'Christmas', 'Easter', 'MemorialDay',
                          'IndependenceDay', 'LaborDay', 'ColumbusDay', 'VeteransDay',
                          'MartinLutherKingDay', 'PresidentsDay', 'Halloween']
        
        for event in holiday_events:
            col_name = f'is_{event.lower()}'
            if 'event_name_1' in df.columns:
                df[col_name] = (df['event_name_1'] == event).astype(int)
            else:
                df[col_name] = 0
            self.feature_names.append(col_name)
        
        holiday_features = ['is_event', 'is_national_event', 'is_religious_event', 'is_snap_day']
        self.feature_names.extend(holiday_features)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: Input DataFrame with sell_price column
            
        Returns:
            DataFrame with price features
        """
        df = df.copy()
        
        if 'sell_price' not in df.columns:
            logger.warning("sell_price column not found. Skipping price features.")
            return df
        
        # Price changes
        df['price_change'] = df.groupby(['item_id', 'store_id'])['sell_price'].diff()
        df['price_change_pct'] = df.groupby(['item_id', 'store_id'])['sell_price'].pct_change()
        
        # Price relative to group mean
        df['price_relative_to_mean'] = df['sell_price'] / df.groupby('item_id')['sell_price'].transform('mean')
        
        # Price momentum (rolling mean of price changes)
        df['price_momentum'] = df.groupby(['item_id', 'store_id'])['price_change'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        price_features = ['price_change', 'price_change_pct', 'price_relative_to_mean', 'price_momentum']
        self.feature_names.extend(price_features)
        
        return df
    
    def create_sales_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional sales-derived features.
        
        Args:
            df: Input DataFrame with sales data
            
        Returns:
            DataFrame with sales features
        """
        df = df.copy()
        
        if 'sales' not in df.columns:
            return df
        
        # Sales volatility
        df['sales_volatility'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        # Sales momentum
        df['sales_momentum'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.diff(7)
        )
        
        # Year-over-year change (if data spans multiple years)
        df['sales_yoy'] = df.groupby(['item_id', 'store_id', 'month', 'day_of_month'])['sales'].transform(
            lambda x: x.pct_change()
        )
        
        sales_features = ['sales_volatility', 'sales_momentum', 'sales_yoy']
        self.feature_names.extend(sales_features)
        
        return df
    
    def encode_categoricals(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        encoding_type: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode
            encoding_type: Type of encoding ('label' or 'frequency')
            
        Returns:
            DataFrame with encoded categoricals
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id', 'weekday']
        
        available_cats = [c for c in categorical_cols if c in df.columns]
        
        for col in available_cats:
            if encoding_type == 'label':
                # Label encoding
                unique_vals = df[col].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_vals)}
                df[f'{col}_encoded'] = df[col].map(encoding_map)
                self.categorical_encodings[col] = encoding_map
                self.feature_names.append(f'{col}_encoded')
                
            elif encoding_type == 'frequency':
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_freq'] = df[col].map(freq_map)
                self.categorical_encodings[col] = freq_map
                self.feature_names.append(f'{col}_freq')
        
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_expanding: bool = False,
        include_price: bool = True
    ) -> pd.DataFrame:
        """
        Create all features in one call.
        
        Args:
            df: Input DataFrame
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            include_expanding: Whether to include expanding features
            include_price: Whether to include price features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Reset feature names
        self.feature_names = []
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Holiday features
        df = self.create_holiday_features(df)
        
        # Lag features
        if include_lags:
            df = self.create_lag_features(df)
        
        # Rolling features
        if include_rolling:
            df = self.create_rolling_features(df)
        
        # Expanding features
        if include_expanding:
            df = self.create_expanding_features(df)
        
        # Price features
        if include_price:
            df = self.create_price_features(df)
        
        # Additional sales features
        df = self.create_sales_features(df)
        
        # Encode categoricals
        df = self.encode_categoricals(df)
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of all created feature columns.
        
        Returns:
            List of feature column names
        """
        return list(set(self.feature_names))
    
    def select_features_for_model(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features suitable for modeling.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (feature DataFrame, list of feature names)
        """
        if exclude_cols is None:
            exclude_cols = ['id', 'item_id', 'store_id', 'dept_id', 'cat_id',
                           'state_id', 'date', 'sales', 'd', 'wm_yr_wk',
                           'weekday', 'event_name_1', 'event_type_1',
                           'event_name_2', 'event_type_2']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Remove any columns with NaN values
        feature_cols = [c for c in feature_cols if df[c].isnull().sum() == 0]
        
        return df[feature_cols], feature_cols


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    loader = DataLoader("../Datasets")
    merged_data = loader.merge_data()
    
    engineer = FeatureEngineer()
    featured_data = engineer.create_all_features(merged_data)
    
    print(f"Features created: {engineer.get_feature_columns()}")
    print(f"Data shape: {featured_data.shape}")
