"""
Visualization Module for Time Series Sales Forecasting
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module generates insight-driven visualizations including:
- Time series decomposition plots (trend, seasonality, residuals)
- Actual vs Forecasted comparison plots
- Model performance visualizations
- Interactive dashboard components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ForecastVisualizer:
    """
    Generate professional visualizations for time series forecasting.
    
    Creates publication-quality plots for model evaluation and reporting.
    """
    
    def __init__(self, output_dir: str = '../reports'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_time_series_decomposition(
        self,
        dates: np.ndarray,
        actual: np.ndarray,
        level: np.ndarray,
        trend: np.ndarray,
        seasonal: np.ndarray,
        residuals: np.ndarray,
        title: str = 'Time Series Decomposition',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a time series decomposition plot showing components.
        
        Args:
            dates: Array of dates
            actual: Original time series
            level: Estimated level component
            trend: Estimated trend component
            seasonal: Seasonal component array
            residuals: Residuals (actual - fitted)
            title: Plot title
            save_name: Optional filename to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Convert dates if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        # Plot original series
        axes[0].plot(dates, actual, 'b-', linewidth=1, label='Actual')
        axes[0].plot(dates, level + trend + seasonal[:len(actual)] if len(seasonal) < len(actual) 
                    else level + trend + seasonal, 'r--', linewidth=1, alpha=0.7, label='Fitted')
        axes[0].set_ylabel('Sales')
        axes[0].legend(loc='upper left')
        axes[0].set_title('Original Series')
        axes[0].grid(True, alpha=0.3)
        
        # Plot level component
        axes[1].plot(dates, level, 'g-', linewidth=1.5)
        axes[1].set_ylabel('Level')
        axes[1].set_title('Level Component')
        axes[1].grid(True, alpha=0.3)
        
        # Plot trend component
        axes[2].plot(dates, trend, 'orange', linewidth=1.5)
        axes[2].set_ylabel('Trend')
        axes[2].set_title('Trend Component')
        axes[2].grid(True, alpha=0.3)
        
        # Plot residuals
        axes[3].plot(dates, residuals, 'purple', linewidth=0.5, alpha=0.7)
        axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[3].fill_between(dates, -2*np.std(residuals), 2*np.std(residuals), 
                            alpha=0.2, color='gray', label='±2 Std Dev')
        axes[3].set_ylabel('Residuals')
        axes[3].set_xlabel('Date')
        axes[3].set_title('Residuals')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Decomposition plot saved to {save_path}")
        
        return fig
    
    def plot_actual_vs_forecast(
        self,
        train_dates: np.ndarray,
        train_actual: np.ndarray,
        test_dates: np.ndarray,
        test_actual: np.ndarray,
        test_forecast: np.ndarray,
        mape_value: float,
        title: str = 'Actual vs Forecasted Sales',
        save_name: Optional[str] = None,
        confidence_lower: Optional[np.ndarray] = None,
        confidence_upper: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Create a comparison plot of actual vs forecasted values.
        
        This is the key visualization showing model performance on the
        validation/test holdout set with the MAPE metric prominently displayed.
        
        Args:
            train_dates: Training period dates
            train_actual: Training period actual values
            test_dates: Test/validation period dates
            test_actual: Test/validation period actual values
            test_forecast: Forecasted values for test period
            mape_value: MAPE percentage for annotation
            title: Plot title
            save_name: Optional filename to save the plot
            confidence_lower: Lower bound of confidence interval
            confidence_upper: Upper bound of confidence interval
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Convert dates if needed
        if not isinstance(train_dates, pd.DatetimeIndex):
            train_dates = pd.to_datetime(train_dates)
        if not isinstance(test_dates, pd.DatetimeIndex):
            test_dates = pd.to_datetime(test_dates)
        
        # Plot training data
        ax.plot(train_dates, train_actual, 'b-', linewidth=1.5, label='Historical Sales', alpha=0.8)
        
        # Plot test actual
        ax.plot(test_dates, test_actual, 'g-', linewidth=1.5, label='Actual (Holdout)', alpha=0.8)
        
        # Plot forecast
        ax.plot(test_dates, test_forecast, 'r--', linewidth=2, label='Forecast', alpha=0.9)
        
        # Add confidence intervals if provided
        if confidence_lower is not None and confidence_upper is not None:
            ax.fill_between(test_dates, confidence_lower, confidence_upper,
                           alpha=0.2, color='red', label='95% Confidence Interval')
        
        # Add MAPE annotation box
        annotation_text = f'MAPE: {mape_value:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, fontsize=14,
               fontweight='bold', verticalalignment='top', bbox=props,
               family='monospace')
        
        # Add performance indicator
        if mape_value <= 3.6:
            perf_text = 'Excellent Performance'
            perf_color = 'green'
        elif mape_value <= 5.0:
            perf_text = 'Good Performance'
            perf_color = 'blue'
        else:
            perf_text = 'Acceptable Performance'
            perf_color = 'orange'
        
        ax.text(0.02, 0.88, perf_text, transform=ax.transAxes, fontsize=11,
               fontweight='bold', verticalalignment='top', color=perf_color,
               bbox=dict(boxstyle='round', facecolor=perf_color, alpha=0.2))
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Actual vs Forecast plot saved to {save_path}")
        
        return fig
    
    def plot_residual_analysis(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = 'Residual Analysis',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create residual analysis plots for model diagnostics.
        
        Args:
            residuals: Model residuals
            fitted_values: Fitted/predicted values
            dates: Optional dates for time-based plotting
            title: Plot title
            save_name: Optional filename to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Residuals over time
        if dates is not None:
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.to_datetime(dates)
            axes[0, 0].plot(dates, residuals, 'b.', alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_title('Residuals Over Time')
        else:
            axes[0, 0].plot(range(len(residuals)), residuals, 'b.', alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Observation')
            axes[0, 0].set_title('Residuals Over Observations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Fitted
        axes[0, 1].scatter(fitted_values, residuals, alpha=0.5, color='blue')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Fitted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1, 0].axvline(x=np.mean(residuals), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(residuals):.2f}')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot (manual implementation)
        from scipy import stats
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf((np.arange(len(sorted_residuals)) + 0.5) / len(sorted_residuals))
        
        axes[1, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5, color='blue')
        
        # Add reference line
        min_q = min(theoretical_quantiles)
        max_q = max(theoretical_quantiles)
        min_r = min(sorted_residuals)
        max_r = max(sorted_residuals)
        axes[1, 1].plot([min_q, max_q], [min_r, max_r], 'r--', linewidth=2)
        
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Residual analysis plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        metrics_df: pd.DataFrame,
        title: str = 'Model Performance Comparison',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar chart comparing model performance metrics.
        
        Args:
            metrics_df: DataFrame with models as index and metrics as columns
            title: Plot title
            save_name: Optional filename to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Select numeric columns only
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        
        if 'MAPE' in numeric_cols:
            # MAPE comparison (lower is better)
            colors = ['green' if x == metrics_df['MAPE'].min() else 'steelblue' 
                     for x in metrics_df['MAPE']]
            axes[0].barh(metrics_df.index, metrics_df['MAPE'], color=colors, alpha=0.7)
            axes[0].set_xlabel('MAPE (%)')
            axes[0].set_title('Mean Absolute Percentage Error')
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(metrics_df['MAPE']):
                axes[0].text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
        
        if 'RMSE' in numeric_cols:
            # RMSE comparison (lower is better)
            colors = ['green' if x == metrics_df['RMSE'].min() else 'steelblue' 
                     for x in metrics_df['RMSE']]
            axes[1].barh(metrics_df.index, metrics_df['RMSE'], color=colors, alpha=0.7)
            axes[1].set_xlabel('RMSE')
            axes[1].set_title('Root Mean Squared Error')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(metrics_df['RMSE']):
                axes[1].text(v + 0.5, i, f'{v:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_seasonal_pattern(
        self,
        seasonal_components: np.ndarray,
        season_length: int = 7,
        title: str = 'Weekly Seasonal Pattern',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the learned seasonal pattern.
        
        Args:
            seasonal_components: Array of seasonal indices
            season_length: Length of seasonal period
            title: Plot title
            save_name: Optional filename to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        x_positions = range(min(season_length, len(seasonal_components)))
        
        ax.bar(x_positions, seasonal_components[:season_length], 
              color='steelblue', alpha=0.7, edgecolor='navy')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(days[:season_length], rotation=45, ha='right')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Seasonal Effect')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Seasonal pattern plot saved to {save_path}")
        
        return fig


def create_dashboard(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    forecasts: np.ndarray,
    metrics: Dict,
    decomposition: Optional[Dict] = None,
    output_dir: str = '../reports'
) -> List[str]:
    """
    Create a complete dashboard of visualizations.
    
    Args:
        train_data: Training DataFrame with date and sales columns
        test_data: Test/validation DataFrame
        forecasts: Forecasted values
        metrics: Dictionary of performance metrics
        decomposition: Optional dictionary with decomposition components
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot filenames
    """
    visualizer = ForecastVisualizer(output_dir)
    saved_files = []
    
    # Extract data
    train_dates = pd.to_datetime(train_data['date']).values
    train_sales = train_data['sales'].values
    
    test_dates = pd.to_datetime(test_data['date']).values
    test_sales = test_data['sales'].values
    
    # Create main comparison plot
    fig1 = visualizer.plot_actual_vs_forecast(
        train_dates=train_dates[-100:],  # Last 100 days of training
        train_actual=train_sales[-100:],
        test_dates=test_dates,
        test_actual=test_sales,
        test_forecast=forecasts,
        mape_value=metrics.get('MAPE', 0),
        title='Sales Forecasting: Actual vs Predicted',
        save_name='actual_vs_forecast'
    )
    saved_files.append('actual_vs_forecast.png')
    plt.close(fig1)
    
    # Create decomposition plot if components available
    if decomposition:
        fig2 = visualizer.plot_time_series_decomposition(
            dates=train_dates,
            actual=train_sales,
            level=decomposition.get('level', train_sales),
            trend=decomposition.get('trend', np.zeros_like(train_sales)),
            seasonal=decomposition.get('seasonal', np.zeros(7)),
            residuals=decomposition.get('residuals', np.zeros_like(train_sales)),
            title='Holt-Winters Time Series Decomposition',
            save_name='decomposition'
        )
        saved_files.append('decomposition.png')
        plt.close(fig2)
    
    # Create residual analysis
    if decomposition and 'residuals' in decomposition:
        fig3 = visualizer.plot_residual_analysis(
            residuals=decomposition['residuals'],
            fitted_values=decomposition.get('fitted', train_sales),
            dates=train_dates,
            title='Model Residual Analysis',
            save_name='residual_analysis'
        )
        saved_files.append('residual_analysis.png')
        plt.close(fig3)
    
    # Create seasonal pattern plot
    if decomposition and 'seasonal' in decomposition:
        fig4 = visualizer.plot_seasonal_pattern(
            seasonal_components=decomposition['seasonal'],
            season_length=7,
            title='Learned Weekly Seasonal Pattern',
            save_name='seasonal_pattern'
        )
        saved_files.append('seasonal_pattern.png')
        plt.close(fig4)
    
    logger.info(f"Dashboard created with {len(saved_files)} visualizations")
    
    return saved_files


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n = 365
    dates = pd.date_range('2024-01-01', periods=n)
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise
    
    # Split into train/test
    train_size = 300
    train_dates = dates[:train_size]
    train_sales = sales[:train_size]
    test_dates = dates[train_size:]
    test_sales = sales[train_size:]
    
    # Simulate forecasts
    forecasts = test_sales + np.random.normal(0, 3, len(test_sales))
    
    # Create visualizer
    viz = ForecastVisualizer()
    
    # Create comparison plot
    fig = viz.plot_actual_vs_forecast(
        train_dates=train_dates,
        train_actual=train_sales,
        test_dates=test_dates,
        test_actual=test_sales,
        test_forecast=forecasts,
        mape_value=3.6,
        save_name='example_comparison'
    )
    
    print("Example visualization created successfully")
