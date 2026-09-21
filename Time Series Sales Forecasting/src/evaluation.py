"""
Evaluation Module for Time Series Sales Forecasting
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module provides comprehensive evaluation metrics and baseline model
comparisons to demonstrate the superiority of the custom Holt-Winters
implementation over naive forecasting approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Calculate standard forecast accuracy metrics.
    
    Implements MAPE, RMSE, MAE, MASE, and other key performance indicators
    for time series forecast evaluation.
    """
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Formula: MAPE = (1/n) * Σ|((actual - predicted) / actual)| * 100
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            MAPE as a percentage
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return 0.0
        
        percentage_errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
        return np.mean(percentage_errors) * 100
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Formula: RMSE = sqrt((1/n) * Σ(actual - predicted)²)
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            RMSE value
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Formula: MAE = (1/n) * Σ|actual - predicted|
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            MAE value
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def mase(
        actual: np.ndarray,
        predicted: np.ndarray,
        training_actual: np.ndarray,
        season_length: int = 1
    ) -> float:
        """
        Calculate Mean Absolute Scaled Error (MASE).
        
        MASE scales the MAE by the in-sample mean absolute error of a naive
        seasonal forecast, making it comparable across different series.
        
        Formula: MASE = MAE / (mean(|training[t] - training[t-s]|))
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            training_actual: Training data for scaling
            season_length: Seasonal period for naive comparison
            
        Returns:
            MASE value
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        training_actual = np.asarray(training_actual, dtype=np.float64)
        
        # Calculate MAE of predictions
        mae_pred = np.mean(np.abs(actual - predicted))
        
        # Calculate MAE of naive seasonal forecast on training data
        if len(training_actual) > season_length:
            naive_errors = np.abs(training_actual[season_length:] - training_actual[:-season_length])
            mae_naive = np.mean(naive_errors)
        else:
            mae_naive = np.mean(np.abs(np.diff(training_actual)))
        
        if mae_naive == 0:
            return 0.0
        
        return mae_pred / mae_naive
    
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
        
        Formula: SMAPE = (1/n) * Σ|predicted - actual| / ((|actual| + |predicted|) / 2) * 100
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            SMAPE as a percentage
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def calculate_all(
        actual: np.ndarray,
        predicted: np.ndarray,
        training_actual: Optional[np.ndarray] = None,
        season_length: int = 7
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics at once.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            training_actual: Optional training data for MASE calculation
            season_length: Seasonal period for MASE
            
        Returns:
            Dictionary with all metric names and values
        """
        metrics = {
            'MAPE': EvaluationMetrics.mape(actual, predicted),
            'RMSE': EvaluationMetrics.rmse(actual, predicted),
            'MAE': EvaluationMetrics.mae(actual, predicted),
            'SMAPE': EvaluationMetrics.smape(actual, predicted)
        }
        
        if training_actual is not None:
            metrics['MASE'] = EvaluationMetrics.mase(
                actual, predicted, training_actual, season_length
            )
        
        return metrics


class BaselineModels:
    """
    Implement baseline forecasting models for comparison.
    
    These naive models serve as benchmarks to demonstrate the value
    added by sophisticated forecasting methods like Holt-Winters.
    """
    
    @staticmethod
    def naive_forecast(
        actual: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Naive forecast: Last observation carried forward.
        
        Predicts that all future values will equal the last observed value.
        
        Args:
            actual: Array of historical actual values
            steps: Number of periods to forecast
            
        Returns:
            Array of naive forecasts
        """
        last_value = actual[-1]
        return np.full(steps, last_value)
    
    @staticmethod
    def seasonal_naive_forecast(
        actual: np.ndarray,
        steps: int,
        season_length: int = 7
    ) -> np.ndarray:
        """
        Seasonal naive forecast: Use value from same season in previous cycle.
        
        For each forecast horizon h, predicts the value observed at
        time t-season_length+h (or wraps around if needed).
        
        Args:
            actual: Array of historical actual values
            steps: Number of periods to forecast
            season_length: Length of seasonal period
            
        Returns:
            Array of seasonal naive forecasts
        """
        n = len(actual)
        forecasts = np.zeros(steps)
        
        for h in range(steps):
            # Index of the value to use from the last complete season
            ref_idx = n - season_length + h
            if ref_idx < 0:
                # Not enough history, use most recent available
                ref_idx = n - 1 - ((n - 1 - h) % season_length)
            
            forecasts[h] = actual[min(ref_idx, n - 1)]
        
        return forecasts
    
    @staticmethod
    def mean_forecast(
        actual: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Mean forecast: Predict the historical mean.
        
        Args:
            actual: Array of historical actual values
            steps: Number of periods to forecast
            
        Returns:
            Array of mean forecasts
        """
        mean_value = np.mean(actual)
        return np.full(steps, mean_value)
    
    @staticmethod
    def drift_forecast(
        actual: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Drift forecast: Linear extrapolation based on first and last observations.
        
        Extends the line connecting the first and last observations.
        
        Args:
            actual: Array of historical actual values
            steps: Number of periods to forecast
            
        Returns:
            Array of drift forecasts
        """
        n = len(actual)
        if n < 2:
            return np.full(steps, actual[-1] if n == 1 else 0)
        
        # Calculate slope
        slope = (actual[-1] - actual[0]) / (n - 1)
        
        # Generate forecasts
        forecasts = np.zeros(steps)
        for h in range(steps):
            forecasts[h] = actual[-1] + slope * (h + 1)
        
        return forecasts
    
    @staticmethod
    def rolling_mean_forecast(
        actual: np.ndarray,
        steps: int,
        window: int = 7
    ) -> np.ndarray:
        """
        Rolling mean forecast: Use mean of last k observations.
        
        Args:
            actual: Array of historical actual values
            steps: Number of periods to forecast
            window: Size of rolling window
            
        Returns:
            Array of rolling mean forecasts
        """
        window = min(window, len(actual))
        mean_value = np.mean(actual[-window:])
        return np.full(steps, mean_value)


class ModelComparator:
    """
    Compare multiple forecasting models against baselines.
    
    This class orchestrates the comparison between custom implementations
    and naive baselines, calculating improvement ratios and generating
    comparison reports.
    """
    
    def __init__(self, actual: np.ndarray, training_actual: Optional[np.ndarray] = None):
        """
        Initialize the comparator.
        
        Args:
            actual: Actual values for the forecast period
            training_actual: Training data (for MASE calculation)
        """
        self.actual = np.asarray(actual, dtype=np.float64)
        self.training_actual = (np.asarray(training_actual, dtype=np.float64) 
                               if training_actual is not None else actual)
        self.results: Dict[str, Dict[str, float]] = {}
    
    def evaluate_model(
        self,
        model_name: str,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a model's predictions and store results.
        
        Args:
            model_name: Name identifier for the model
            predictions: Array of predicted values
            
        Returns:
            Dictionary of metrics for this model
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        
        # Ensure lengths match
        min_len = min(len(self.actual), len(predictions))
        actual_trimmed = self.actual[:min_len]
        predictions_trimmed = predictions[:min_len]
        
        metrics = EvaluationMetrics.calculate_all(
            actual_trimmed,
            predictions_trimmed,
            self.training_actual,
            season_length=7
        )
        
        self.results[model_name] = metrics
        logger.info(f"{model_name} - MAPE: {metrics['MAPE']:.2f}%, "
                   f"RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")
        
        return metrics
    
    def compare_baselines(self) -> pd.DataFrame:
        """
        Run all baseline models and compare their performance.
        
        Returns:
            DataFrame with comparison results
        """
        steps = len(self.actual)
        
        # Naive forecast
        naive_pred = BaselineModels.naive_forecast(self.training_actual, steps)
        self.evaluate_model('Naive', naive_pred)
        
        # Seasonal naive forecast
        seasonal_naive_pred = BaselineModels.seasonal_naive_forecast(
            self.training_actual, steps, season_length=7
        )
        self.evaluate_model('Seasonal_Naive', seasonal_naive_pred)
        
        # Mean forecast
        mean_pred = BaselineModels.mean_forecast(self.training_actual, steps)
        self.evaluate_model('Mean', mean_pred)
        
        # Drift forecast
        drift_pred = BaselineModels.drift_forecast(self.training_actual, steps)
        self.evaluate_model('Drift', drift_pred)
        
        # Rolling mean forecast
        rolling_pred = BaselineModels.rolling_mean_forecast(self.training_actual, steps, window=7)
        self.evaluate_model('Rolling_Mean', rolling_pred)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get results as a formatted comparison table.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        df.index.name = 'Model'
        
        return df
    
    def calculate_improvement(
        self,
        model_name: str,
        baseline_name: str = 'Naive',
        metric: str = 'MAPE'
    ) -> float:
        """
        Calculate how much better a model is compared to a baseline.
        
        Args:
            model_name: Name of the model to evaluate
            baseline_name: Name of the baseline model
            metric: Metric to use for comparison
            
        Returns:
            Improvement ratio (baseline_error / model_error)
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not evaluated")
        if baseline_name not in self.results:
            raise ValueError(f"Baseline '{baseline_name}' not evaluated")
        
        baseline_error = self.results[baseline_name][metric]
        model_error = self.results[model_name][metric]
        
        if model_error == 0:
            return float('inf')
        
        return baseline_error / model_error
    
    def generate_report(self) -> str:
        """
        Generate a text report summarizing the comparison.
        
        Returns:
            Formatted text report
        """
        if not self.results:
            return "No models evaluated yet."
        
        report_lines = [
            "=" * 60,
            "FORECAST MODEL COMPARISON REPORT",
            "=" * 60,
            ""
        ]
        
        # Comparison table
        table = self.get_comparison_table()
        report_lines.append("Model Performance Metrics:")
        report_lines.append(table.to_string())
        report_lines.append("")
        
        # Find best model
        best_mape = min(self.results.items(), key=lambda x: x[1]['MAPE'])
        report_lines.append(f"Best Model by MAPE: {best_mape[0]} ({best_mape[1]['MAPE']:.2f}%)")
        report_lines.append("")
        
        # Improvement over baselines
        report_lines.append("Improvement Over Baselines:")
        for model_name in self.results:
            if model_name not in ['Naive', 'Seasonal_Naive', 'Mean', 'Drift', 'Rolling_Mean']:
                naive_improvement = self.calculate_improvement(model_name, 'Naive', 'MAPE')
                seasonal_improvement = self.calculate_improvement(model_name, 'Seasonal_Naive', 'MAPE')
                report_lines.append(f"  {model_name}:")
                report_lines.append(f"    - {naive_improvement:.2f}x better than Naive")
                report_lines.append(f"    - {seasonal_improvement:.2f}x better than Seasonal Naive")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def create_evaluation_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline_predictions: Optional[Dict[str, np.ndarray]] = None,
    training_data: Optional[np.ndarray] = None
) -> Dict:
    """
    Create a comprehensive evaluation summary.
    
    Args:
        actual: Actual values
        predicted: Model predictions
        baseline_predictions: Dictionary of baseline model predictions
        training_data: Training data for MASE calculation
        
    Returns:
        Dictionary with full evaluation summary
    """
    comparator = ModelComparator(actual, training_data)
    
    # Evaluate main model
    main_metrics = comparator.evaluate_model('HoltWinters', predicted)
    
    # Evaluate baselines if provided
    if baseline_predictions:
        for name, preds in baseline_predictions.items():
            comparator.evaluate_model(name, preds)
    
    # Calculate improvements
    improvements = {}
    for baseline in ['Naive', 'Seasonal_Naive']:
        if baseline in comparator.results:
            improvements[f'improvement_over_{baseline}'] = comparator.calculate_improvement(
                'HoltWinters', baseline, 'MAPE'
            )
    
    summary = {
        'metrics': main_metrics,
        'comparison': comparator.get_comparison_table().to_dict(),
        'improvements': improvements,
        'report': comparator.generate_report()
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_train = 300
    n_test = 30
    t = np.arange(n_train + n_test)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 5, n_train + n_test)
    y = trend + seasonal + noise
    
    train_data = y[:n_train]
    test_actual = y[n_train:]
    
    # Simulate model predictions (with some error)
    test_predicted = test_actual + np.random.normal(0, 3, n_test)
    
    # Create comparator and evaluate
    comparator = ModelComparator(test_actual, train_data)
    comparator.evaluate_model('Custom_Model', test_predicted)
    comparator.compare_baselines()
    
    print(comparator.generate_report())
