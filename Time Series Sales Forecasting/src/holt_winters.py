"""
From-Scratch Triple Exponential Smoothing (Holt-Winters) Implementation
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module implements the Holt-Winters Triple Exponential Smoothing algorithm
completely from scratch using only NumPy and Pandas. No statsmodels or other
time series libraries are used for the core forecasting logic.

The implementation includes:
- Additive and Multiplicative seasonal variants
- Level, Trend, and Seasonal component initialization
- Iterative parameter updates following the mathematical formulation
- Forecasting with confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HoltWintersParams:
    """Container for Holt-Winters model parameters."""
    alpha: float  # Level smoothing parameter
    beta: float   # Trend smoothing parameter
    gamma: float  # Seasonal smoothing parameter
    level: float  # Initial level
    trend: float  # Initial trend
    seasonal: np.ndarray  # Initial seasonal components
    season_length: int  # Length of seasonal period


class HoltWintersModel:
    """
    Triple Exponential Smoothing (Holt-Winters) model implemented from scratch.
    
    This class implements both additive and multiplicative seasonal variants
    of the Holt-Winters method for time series forecasting.
    
    Mathematical Formulation (Additive Seasonality):
    -----------------------------------------------
    Level:    L(t) = α * (y(t) / S(t-s)) + (1-α) * (L(t-1) + T(t-1))
    Trend:    T(t) = β * (L(t) - L(t-1)) + (1-β) * T(t-1)
    Seasonal: S(t) = γ * (y(t) / L(t)) + (1-γ) * S(t-s)
    Forecast: F(t+h) = L(t) + h*T(t) + S(t-s+h)
    
    For multiplicative seasonality, the formulas adjust accordingly.
    
    Parameters:
    -----------
    season_length : int
        Number of periods in each seasonal cycle (e.g., 7 for weekly, 12 for monthly)
    seasonal_type : str
        'add' for additive seasonality, 'mul' for multiplicative seasonality
    alpha : float, optional
        Level smoothing parameter (0 < α < 1). Default is 0.2
    beta : float, optional
        Trend smoothing parameter (0 < β < 1). Default is 0.1
    gamma : float, optional
        Seasonal smoothing parameter (0 < γ < 1). Default is 0.1
    """
    
    def __init__(
        self,
        season_length: int = 7,
        seasonal_type: str = 'add',
        alpha: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.1
    ):
        """
        Initialize the Holt-Winters model.
        
        Args:
            season_length: Length of the seasonal period
            seasonal_type: Type of seasonality ('add' or 'mul')
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
            gamma: Seasonal smoothing parameter
        """
        if seasonal_type not in ['add', 'mul']:
            raise ValueError("seasonal_type must be 'add' or 'mul'")
        
        self.season_length = season_length
        self.seasonal_type = seasonal_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Model state (will be set during fitting)
        self.level: Optional[float] = None
        self.trend: Optional[float] = None
        self.seasonal: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.params: Optional[HoltWintersParams] = None
        self._is_fitted = False
        self.n_observations: Optional[int] = None
    
    def _initialize_components(self, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Initialize level, trend, and seasonal components.
        
        Uses a simple but effective initialization strategy:
        - Level: Average of first season
        - Trend: Average change per period over first few seasons
        - Seasonal: Deviation from level for each period in first season(s)
        
        Args:
            y: Input time series
            
        Returns:
            Tuple of (initial_level, initial_trend, initial_seasonal)
        """
        n = len(y)
        m = self.season_length
        
        # Ensure we have enough data
        if n < 2 * m:
            logger.warning(f"Time series length ({n}) is less than 2 seasonal periods ({2*m}). "
                          f"Using simplified initialization.")
            # Simplified initialization for short series
            level = np.mean(y[:m]) if n >= m else np.mean(y)
            trend = 0.0
            seasonal = np.zeros(m)
            if n >= m:
                seasonal = y[:m] - level
            return level, trend, seasonal
        
        # Initialize level as average of first season
        level = np.mean(y[:m])
        
        # Initialize trend as average change between first two seasons
        if n >= 2 * m:
            second_season_avg = np.mean(y[m:2*m])
            trend = (second_season_avg - level) / m
        else:
            trend = 0.0
        
        # Initialize seasonal components
        # Calculate deseasonalized data and extract seasonal pattern
        seasonal = np.zeros(m)
        
        # Number of complete seasons available
        n_seasons = n // m
        
        if self.seasonal_type == 'add':
            # Additive: seasonal = observation - moving average
            # Use centered moving average for initialization
            ma = np.convolve(y, np.ones(m)/m, mode='same')
            
            # Calculate seasonal indices for each position in the season
            for i in range(m):
                indices = list(range(i, min(n_seasons * m, n), m))
                if len(indices) > 0:
                    seasonal[i] = np.mean(y[indices] - ma[indices])
            
            # Normalize seasonal components to sum to zero
            seasonal = seasonal - np.mean(seasonal)
            
        else:  # multiplicative
            # Multiplicative: seasonal = observation / moving average
            ma = np.convolve(y, np.ones(m)/m, mode='same')
            ma[ma == 0] = 1e-10  # Avoid division by zero
            
            for i in range(m):
                indices = list(range(i, min(n_seasons * m, n), m))
                if len(indices) > 0 and np.all(ma[indices] != 0):
                    seasonal[i] = np.mean(y[indices] / ma[indices])
            
            # Normalize seasonal components to average to one
            seasonal_mean = np.mean(seasonal)
            if seasonal_mean > 0:
                seasonal = seasonal / seasonal_mean
            else:
                seasonal = np.ones(m)
        
        return level, trend, seasonal
    
    def fit(self, y: np.ndarray) -> 'HoltWintersModel':
        """
        Fit the Holt-Winters model to the time series.
        
        Implements the iterative updating equations for level, trend, and
        seasonal components using the specified smoothing parameters.
        
        Args:
            y: Time series data as numpy array
            
        Returns:
            Self for method chaining
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        m = self.season_length
        
        if n < m:
            raise ValueError(f"Time series must have at least {m} observations "
                           f"(season_length={m}), but got {n}")
        
        # Handle NaN values
        if np.any(np.isnan(y)):
            logger.warning("NaN values detected in time series. Forward filling.")
            y = pd.Series(y).ffill().bfill().values
        
        # Initialize components
        self.level, self.trend, self.seasonal = self._initialize_components(y)
        
        # Store initial values
        initial_level = self.level
        initial_trend = self.trend
        initial_seasonal = self.seasonal.copy()
        
        # Arrays to store fitted values and component history
        self.fitted_values = np.zeros(n)
        level_history = np.zeros(n)
        trend_history = np.zeros(n)
        seasonal_history = np.zeros((n, m))
        
        # Iterative updating
        for t in range(n):
            # Get the appropriate seasonal index
            s_idx = t % m
            
            if t == 0:
                # First observation: use initialized values
                if self.seasonal_type == 'add':
                    self.fitted_values[t] = self.level + self.trend + self.seasonal[s_idx]
                else:
                    self.fitted_values[t] = (self.level + self.trend) * self.seasonal[s_idx]
            else:
                # Previous values
                prev_level = level_history[t-1]
                prev_trend = trend_history[t-1]
                prev_seasonal = seasonal_history[t-1, s_idx] if t > m else initial_seasonal[s_idx]
                
                # Update level
                if self.seasonal_type == 'add':
                    self.level = (self.alpha * (y[t] - prev_seasonal) + 
                                 (1 - self.alpha) * (prev_level + prev_trend))
                else:
                    seasonal_divisor = prev_seasonal if prev_seasonal != 0 else 1e-10
                    self.level = (self.alpha * (y[t] / seasonal_divisor) + 
                                 (1 - self.alpha) * (prev_level + prev_trend))
                
                # Update trend
                self.trend = (self.beta * (self.level - prev_level) + 
                             (1 - self.beta) * prev_trend)
                
                # Update seasonal component
                if self.seasonal_type == 'add':
                    self.seasonal[s_idx] = (self.gamma * (y[t] - self.level) + 
                                           (1 - self.gamma) * prev_seasonal)
                else:
                    level_divisor = self.level if self.level != 0 else 1e-10
                    self.seasonal[s_idx] = (self.gamma * (y[t] / level_divisor) + 
                                           (1 - self.gamma) * prev_seasonal)
                
                # Compute fitted value
                if self.seasonal_type == 'add':
                    self.fitted_values[t] = self.level + self.trend + self.seasonal[s_idx]
                else:
                    self.fitted_values[t] = (self.level + self.trend) * self.seasonal[s_idx]
            
            # Store component values
            level_history[t] = self.level
            trend_history[t] = self.trend
            seasonal_history[t] = self.seasonal.copy()
        
        # Calculate residuals
        self.residuals = y - self.fitted_values
        
        # Store final parameters
        self.params = HoltWintersParams(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            level=float(self.level),
            trend=float(self.trend),
            seasonal=self.seasonal.copy(),
            season_length=m
        )
        
        self.n_observations = n
        self._is_fitted = True
        logger.info(f"Holt-Winters model fitted with alpha={self.alpha:.4f}, "
                   f"beta={self.beta:.4f}, gamma={self.gamma:.4f}")
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for future periods.
        
        Args:
            steps: Number of periods to forecast ahead
            
        Returns:
            Array of forecasted values
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        forecasts = np.zeros(steps)
        m = self.season_length
        
        for h in range(1, steps + 1):
            # Get seasonal index for forecast horizon
            # Continue the learned seasonal phase from the end of training.
            # Using (h - 1) % m would reset the season at forecast step 1 and
            # can shift the weekly pattern when the training length is not a
            # multiple of the seasonal period.
            s_idx = (self.n_observations + h - 1) % m
            
            if self.seasonal_type == 'add':
                # Additive: F(t+h) = L(t) + h*T(t) + S(t-s+h)
                forecasts[h-1] = self.level + h * self.trend + self.seasonal[s_idx]
            else:
                # Multiplicative: F(t+h) = (L(t) + h*T(t)) * S(t-s+h)
                forecasts[h-1] = (self.level + h * self.trend) * self.seasonal[s_idx]
        
        return forecasts
    
    def forecast(self, y: np.ndarray, steps: int) -> np.ndarray:
        """
        Fit the model and generate forecasts in one step.
        
        Convenience method that combines fit() and predict().
        
        Args:
            y: Training time series
            steps: Number of periods to forecast
            
        Returns:
            Array of forecasted values
        """
        self.fit(y)
        return self.predict(steps)
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get the decomposed time series components.
        
        Returns:
            Dictionary with 'level', 'trend', 'seasonal', and 'fitted' arrays
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first.")
        
        return {
            'level': np.full(len(self.fitted_values), self.level),
            'trend': np.full(len(self.fitted_values), self.trend),
            'seasonal': self.seasonal,
            'fitted': self.fitted_values,
            'residuals': self.residuals
        }
    
    def optimize_parameters(
        self,
        y: np.ndarray,
        param_grid: Optional[Dict[str, List[float]]] = None
    ) -> 'HoltWintersModel':
        """
        Optimize smoothing parameters using grid search to minimize SSE.
        
        Args:
            y: Time series data
            param_grid: Dictionary of parameter values to try
            
        Returns:
            Self with optimized parameters
        """
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 0.2, 0.3, 0.5, 0.7],
                'beta': [0.05, 0.1, 0.2, 0.3],
                'gamma': [0.05, 0.1, 0.2, 0.3]
            }
        
        best_sse = float('inf')
        best_params = {}
        
        for alpha in param_grid.get('alpha', [0.2]):
            for beta in param_grid.get('beta', [0.1]):
                for gamma in param_grid.get('gamma', [0.1]):
                    try:
                        # Create temporary model with these parameters
                        temp_model = HoltWintersModel(
                            season_length=self.season_length,
                            seasonal_type=self.seasonal_type,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma
                        )
                        temp_model.fit(y)
                        
                        # Calculate sum of squared errors
                        sse = np.sum(temp_model.residuals ** 2)
                        
                        if sse < best_sse:
                            best_sse = sse
                            best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
                    
                    except Exception:
                        continue
        
        if best_params:
            self.alpha = best_params['alpha']
            self.beta = best_params['beta']
            self.gamma = best_params['gamma']
            logger.info(f"Optimized parameters: {best_params}, SSE: {best_sse:.4f}")
        
        # Refit with optimized parameters
        self.fit(y)
        
        return self
    
    def get_confidence_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for forecasts.
        
        Uses the variance of residuals to estimate prediction uncertainty.
        
        Args:
            steps: Number of forecast periods
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first.")
        
        forecasts = self.predict(steps)
        
        # Estimate residual standard deviation
        residual_std = np.std(self.residuals)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Prediction intervals widen with forecast horizon
        # Using sqrt(h) scaling for uncertainty growth
        horizons = np.arange(1, steps + 1)
        interval_width = z_score * residual_std * np.sqrt(horizons)
        
        lower_bound = forecasts - interval_width
        upper_bound = forecasts + interval_width
        
        return lower_bound, upper_bound


class AggregatedHoltWinters:
    """
    Holt-Winters model for aggregated time series forecasting.
    
    This class applies the from-scratch Holt-Winters implementation
    to aggregated sales data (total sales across all items/stores).
    """
    
    def __init__(
        self,
        season_length: int = 7,
        seasonal_type: str = 'add',
        alpha: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.1
    ):
        """Initialize the aggregated forecaster."""
        self.model = HoltWintersModel(
            season_length=season_length,
            seasonal_type=seasonal_type,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        self._is_fitted = False
    
    def fit(self, dates: np.ndarray, sales: np.ndarray) -> 'AggregatedHoltWinters':
        """
        Fit the model to aggregated sales data.
        
        Args:
            dates: Array of dates
            sales: Array of total sales values
            
        Returns:
            Self for method chaining
        """
        self.model.fit(sales)
        self.dates = pd.to_datetime(dates)
        self._is_fitted = True
        return self
    
    def predict(self, steps: int, start_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generate forecasts with dates.
        
        Args:
            steps: Number of periods to forecast
            start_date: Starting date for forecasts (default: day after last training date)
            
        Returns:
            DataFrame with date and forecast columns
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first.")
        
        forecasts = self.model.predict(steps)
        
        if start_date is None:
            start_date = self.dates[-1] + pd.Timedelta(days=1)
        
        forecast_dates = pd.date_range(start=start_date, periods=steps, freq='D')
        
        result = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts
        })
        
        return result
    
    def evaluate(self, actual: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast accuracy against actual values.
        
        Args:
            actual: Array of actual values
            
        Returns:
            Dictionary with MAPE, MAE, RMSE metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first.")
        
        if len(actual) != len(self.model.fitted_values):
            raise ValueError("Actual values length must match fitted values length")
        
        predicted = self.model.fitted_values
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse
        }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic time series with trend and seasonality
    n = 365
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 5, n)
    y = trend + seasonal + noise
    
    # Fit model
    hw = HoltWintersModel(season_length=7, seasonal_type='add')
    hw.fit(y)
    
    # Generate forecasts
    forecasts = hw.predict(30)
    
    print(f"Fitted values shape: {hw.fitted_values.shape}")
    print(f"Forecasts: {forecasts[:10]}")
    print(f"Final level: {hw.level:.4f}")
    print(f"Final trend: {hw.trend:.4f}")
