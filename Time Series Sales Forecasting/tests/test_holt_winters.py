"""
Unit Tests for Time Series Sales Forecasting Pipeline
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module contains pytest tests to validate the core mathematical functions
of the from-scratch Holt-Winters implementation and evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from holt_winters import HoltWintersModel
from evaluation import EvaluationMetrics, BaselineModels


class TestHoltWintersInitialization:
    """Test the initialization of Holt-Winters components."""
    
    def test_initial_level_calculation(self):
        """Test that initial level is calculated correctly as mean of first season."""
        np.random.seed(42)
        y = np.array([10, 12, 11, 13, 12, 14, 13] * 3)  # 3 weeks of data
        
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        level, trend, seasonal = model._initialize_components(y)
        
        expected_level = np.mean(y[:7])
        assert np.isclose(level, expected_level), f"Level {level} != expected {expected_level}"
    
    def test_initial_seasonal_sum_zero_additive(self):
        """Test that additive seasonal components sum to zero."""
        np.random.seed(42)
        y = np.array([10, 12, 11, 13, 12, 14, 13] * 4)  # 4 weeks of data
        
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        level, trend, seasonal = model._initialize_components(y)
        
        # Seasonal components should sum to approximately zero (normalized)
        assert np.isclose(np.sum(seasonal), 0, atol=1e-6), \
            f"Additive seasonal sum {np.sum(seasonal)} != 0"
    
    def test_initial_seasonal_mean_one_multiplicative(self):
        """Test that multiplicative seasonal components average to one."""
        np.random.seed(42)
        y = np.abs(np.array([10, 12, 11, 13, 12, 14, 13] * 4))  # Positive values
        
        model = HoltWintersModel(season_length=7, seasonal_type='mul')
        level, trend, seasonal = model._initialize_components(y)
        
        # Seasonal components should average to approximately one (normalized)
        assert np.isclose(np.mean(seasonal), 1, atol=0.1), \
            f"Multiplicative seasonal mean {np.mean(seasonal)} != 1"
    
    def test_trend_initialization(self):
        """Test that initial trend captures direction of change."""
        # Create series with clear upward trend
        y = np.array([10, 11, 12, 13, 14, 15, 16,
                      20, 21, 22, 23, 24, 25, 26])  # Two seasons with increase
        
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        level, trend, seasonal = model._initialize_components(y)
        
        # Trend should be positive (upward)
        assert trend > 0, f"Trend {trend} should be positive for increasing series"


class TestHoltWintersFitting:
    """Test the fitting process of Holt-Winters model."""
    
    def test_fit_produces_fitted_values(self):
        """Test that fit produces fitted values of correct length."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2, n)
        
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        model.fit(y)
        
        assert len(model.fitted_values) == n, \
            f"Fitted values length {len(model.fitted_values)} != input length {n}"
    
    def test_fit_reduces_error_over_time(self):
        """Test that fitted values track actual values reasonably well."""
        np.random.seed(42)
        n = 200
        t = np.arange(n)
        # Strong signal with clear pattern
        y = 100 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 7)
        
        model = HoltWintersModel(season_length=7, seasonal_type='add', 
                                  alpha=0.3, beta=0.1, gamma=0.2)
        model.fit(y)
        
        # Calculate MAPE on fitted values
        mape = EvaluationMetrics.mape(y, model.fitted_values)
        
        # Should have reasonable fit (MAPE < 10% for this clean signal)
        assert mape < 10, f"MAPE {mape}% too high for clean signal"
    
    def test_residuals_sum_near_zero(self):
        """Test that residuals sum to approximately zero after fitting."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 1, n)
        
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        model.fit(y)
        
        # Residuals should sum to approximately zero
        residual_sum = np.sum(model.residuals)
        assert np.abs(residual_sum) < n * 0.5, \
            f"Residual sum {residual_sum} too large"
    
    def test_model_state_after_fit(self):
        """Test that model state is properly set after fitting."""
        np.random.seed(42)
        y = np.random.randn(100) + 50
        
        model = HoltWintersModel(season_length=7)
        model.fit(y)
        
        assert model._is_fitted, "Model should be marked as fitted"
        assert model.level is not None, "Level should be set"
        assert model.trend is not None, "Trend should be set"
        assert model.seasonal is not None, "Seasonal should be set"
        assert len(model.seasonal) == 7, "Seasonal should have 7 components"


class TestHoltWintersPrediction:
    """Test the prediction functionality of Holt-Winters model."""
    
    def test_predict_requires_fit(self):
        """Test that predict raises error if model not fitted."""
        model = HoltWintersModel(season_length=7)
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(10)
    
    def test_predict_returns_correct_length(self):
        """Test that predict returns array of correct length."""
        np.random.seed(42)
        y = np.random.randn(100) + 50
        
        model = HoltWintersModel(season_length=7)
        model.fit(y)
        
        steps = 14
        forecasts = model.predict(steps)
        
        assert len(forecasts) == steps, \
            f"Forecast length {len(forecasts)} != requested steps {steps}"
    
    def test_forecast_incorporates_seasonality(self):
        """Test that forecasts reflect seasonal patterns."""
        # Create series with strong weekly seasonality
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        seasonal_pattern = np.array([5, 10, 15, 20, 25, 20, 10])  # Weekly pattern
        y = 100 + np.tile(seasonal_pattern, n // 7 + 1)[:n]
        
        model = HoltWintersModel(season_length=7, seasonal_type='add',
                                  alpha=0.3, beta=0.05, gamma=0.3)
        model.fit(y)
        
        # Generate 7-day forecast
        forecasts = model.predict(7)
        
        # Forecasts should show variation (not all same value)
        assert np.std(forecasts) > 0.1, "Forecasts should reflect seasonality"


class TestEvaluationMetrics:
    """Test the evaluation metrics calculations."""
    
    def test_mape_calculation(self):
        """Test MAPE calculation accuracy."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([105, 195, 310, 390, 505])
        
        mape = EvaluationMetrics.mape(actual, predicted)
        
        # Manual calculation
        errors = np.abs((actual - predicted) / actual)
        expected_mape = np.mean(errors) * 100
        
        assert np.isclose(mape, expected_mape), f"MAPE {mape} != expected {expected_mape}"
    
    def test_mape_zero_actual_handling(self):
        """Test MAPE handles zero actual values gracefully."""
        actual = np.array([100, 0, 300, 0, 500])
        predicted = np.array([105, 5, 310, 2, 505])
        
        mape = EvaluationMetrics.mape(actual, predicted)
        
        # Should only consider non-zero actual values
        assert np.isfinite(mape), "MAPE should be finite with zeros in actual"
    
    def test_rmse_calculation(self):
        """Test RMSE calculation accuracy."""
        actual = np.array([10, 20, 30, 40, 50])
        predicted = np.array([12, 18, 32, 38, 52])
        
        rmse = EvaluationMetrics.rmse(actual, predicted)
        
        # Manual calculation
        errors = actual - predicted
        expected_rmse = np.sqrt(np.mean(errors ** 2))
        
        assert np.isclose(rmse, expected_rmse), f"RMSE {rmse} != expected {expected_rmse}"
    
    def test_mae_calculation(self):
        """Test MAE calculation accuracy."""
        actual = np.array([10, 20, 30, 40, 50])
        predicted = np.array([12, 18, 32, 38, 52])
        
        mae = EvaluationMetrics.mae(actual, predicted)
        
        # Manual calculation
        expected_mae = np.mean(np.abs(actual - predicted))
        
        assert np.isclose(mae, expected_mae), f"MAE {mae} != expected {expected_mae}"
    
    def test_smape_calculation(self):
        """Test SMAPE calculation accuracy."""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        
        smape = EvaluationMetrics.smape(actual, predicted)
        
        assert np.isfinite(smape), "SMAPE should be finite"
        assert smape >= 0, "SMAPE should be non-negative"


class TestBaselineModels:
    """Test baseline forecasting models."""
    
    def test_naive_forecast_last_value(self):
        """Test that naive forecast uses last observed value."""
        actual = np.array([10, 20, 30, 40, 50])
        steps = 5
        
        forecasts = BaselineModels.naive_forecast(actual, steps)
        
        expected = np.full(steps, 50)  # Last value repeated
        assert np.array_equal(forecasts, expected), \
            f"Naive forecasts {forecasts} != expected {expected}"
    
    def test_seasonal_naive_forecast(self):
        """Test seasonal naive forecast uses values from previous season."""
        actual = np.array([1, 2, 3, 4, 5, 6, 7,   # Week 1
                           11, 12, 13, 14, 15, 16, 17])  # Week 2
        steps = 7
        season_length = 7
        
        forecasts = BaselineModels.seasonal_naive_forecast(actual, steps, season_length)
        
        # Should use week 2 values as reference for week 3 forecast
        expected = actual[7:14]  # Last complete season
        assert np.array_equal(forecasts, expected), \
            f"Seasonal naive forecasts {forecasts} != expected {expected}"
    
    def test_mean_forecast(self):
        """Test mean forecast uses historical average."""
        actual = np.array([10, 20, 30, 40, 50])
        steps = 3
        
        forecasts = BaselineModels.mean_forecast(actual, steps)
        
        expected_mean = np.mean(actual)
        expected = np.full(steps, expected_mean)
        
        assert np.allclose(forecasts, expected), \
            f"Mean forecasts {forecasts} != expected {expected}"
    
    def test_drift_forecast_slope(self):
        """Test drift forecast extrapolates linear trend."""
        actual = np.array([10, 20, 30, 40, 50])  # Linear increase of 10 per step
        steps = 3
        
        forecasts = BaselineModels.drift_forecast(actual, steps)
        
        # Should continue the trend: 60, 70, 80
        expected = np.array([60, 70, 80])
        
        assert np.allclose(forecasts, expected), \
            f"Drift forecasts {forecasts} != expected {expected}"


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_fit_predict_cycle(self):
        """Test complete fit and predict cycle."""
        np.random.seed(42)
        
        # Generate synthetic data with known properties
        n_train = 200
        n_test = 20
        t = np.arange(n_train + n_test)
        
        trend = 100 + 0.2 * t
        seasonal = 15 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, n_train + n_test)
        y = trend + seasonal + noise
        
        train_data = y[:n_train]
        test_data = y[n_train:]
        
        # Fit model
        model = HoltWintersModel(season_length=7, seasonal_type='add')
        model.fit(train_data)
        
        # Predict
        forecasts = model.predict(n_test)
        
        # Evaluate
        mape = EvaluationMetrics.mape(test_data, forecasts)
        
        # Should achieve reasonable accuracy
        assert mape < 15, f"MAPE {mape}% too high for synthetic data"
    
    def test_parameter_optimization_improves_fit(self):
        """Test that parameter optimization reduces error."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = 100 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2, n)
        
        # Fit with default parameters
        model_default = HoltWintersModel(season_length=7)
        model_default.fit(y)
        sse_default = np.sum(model_default.residuals ** 2)
        
        # Fit with optimized parameters
        model_optimized = HoltWintersModel(season_length=7)
        model_optimized.optimize_parameters(y)
        sse_optimized = np.sum(model_optimized.residuals ** 2)
        
        # Optimized should have lower or equal SSE
        assert sse_optimized <= sse_default * 1.1, \
            f"Optimized SSE {sse_optimized} not better than default {sse_default}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestChronologicalPipelineBoundaries:
    """Tests for the documented leakage-safe train/validation/test design."""

    def test_documented_boundaries_do_not_overlap(self):
        from pipeline import (
            TRAIN_END,
            VALIDATION_START,
            VALIDATION_END,
            TEST_START,
            TEST_END,
        )

        assert TRAIN_END < VALIDATION_START
        assert VALIDATION_END < TEST_START
        assert TRAIN_END == pd.Timestamp("2025-11-05")
        assert VALIDATION_START == pd.Timestamp("2025-11-06")
        assert VALIDATION_END == pd.Timestamp("2025-12-03")
        assert TEST_START == pd.Timestamp("2025-12-04")
        assert TEST_END == pd.Timestamp("2025-12-31")

    def test_split_train_validation_uses_documented_dates(self):
        from pipeline import split_train_validation

        dates = pd.date_range("2025-11-01", "2025-12-05", freq="D")
        frame = pd.DataFrame({"date": dates, "sales": np.ones(len(dates))})

        train, validation = split_train_validation(frame)

        assert train["date"].max() == pd.Timestamp("2025-11-05")
        assert validation["date"].min() == pd.Timestamp("2025-11-06")
        assert validation["date"].max() == pd.Timestamp("2025-12-03")

    def test_parameter_selection_returns_validation_optimum(self):
        from pipeline import select_parameters

        t = np.arange(120)
        train = 100 + 0.2 * t + 10 * np.sin(2 * np.pi * t / 7)
        validation_t = np.arange(14)
        validation = 124 + 0.2 * validation_t + 10 * np.sin(
            2 * np.pi * (120 + validation_t) / 7
        )

        params, table = select_parameters(train, validation)

        assert set(params) == {"alpha", "beta", "gamma"}
        assert len(table) == 5 * 5 * 4
        assert table["MAPE"].is_monotonic_increasing
        assert table.iloc[0]["MAPE"] <= table.iloc[-1]["MAPE"]


    def test_forecast_preserves_seasonal_phase_after_non_multiple_training_length(self):
        """Forecast step 1 must use the seasonal position after the final observation."""
        seasonal_pattern = np.array([0, 10, 20, 30, 40, 30, 10], dtype=float)
        y = 100 + np.tile(seasonal_pattern, 15)[:103]

        model = HoltWintersModel(
            season_length=7,
            seasonal_type="add",
            alpha=0.3,
            beta=0.05,
            gamma=0.3,
        )
        model.fit(y)
        # Replace learned components with deterministic markers so the test
        # verifies the seasonal index calculation rather than model fit quality.
        model.level = 0.0
        model.trend = 0.0
        model.seasonal = np.arange(7, dtype=float)
        forecasts = model.predict(7)

        # 103 observations leaves the next observation at seasonal index 5.
        assert np.array_equal(forecasts, np.array([5, 6, 0, 1, 2, 3, 4], dtype=float))
