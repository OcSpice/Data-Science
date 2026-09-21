"""
Main Pipeline for Time Series Sales Forecasting
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This is the main orchestration module that integrates all components:
- Data loading and merging
- Feature engineering
- From-scratch Holt-Winters modeling
- Evaluation against baselines
- Visualization and reporting

The pipeline demonstrates senior-level data science capabilities with
mathematical rigor and production-ready code quality.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from holt_winters import HoltWintersModel, AggregatedHoltWinters
from evaluation import EvaluationMetrics, BaselineModels, ModelComparator
from visualization import ForecastVisualizer, create_dashboard
from report_generator import ReportGenerator, create_model_card

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    End-to-end forecasting pipeline for retail sales prediction.
    
    This class orchestrates the complete workflow from raw data to
    final reports, demonstrating production-grade implementation.
    """
    
    # Author metadata - persists across all pipeline runs
    AUTHOR_NAME = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    PORTFOLIO_CATEGORY = "Data Science"
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = '../reports',
        season_length: int = 7,
        seasonal_type: str = 'add'
    ):
        """
        Initialize the forecasting pipeline.
        
        Args:
            data_dir: Path to directory containing CSV data files
            output_dir: Directory for saving outputs
            season_length: Length of seasonal period (default 7 for weekly)
            seasonal_type: Type of seasonality ('add' or 'mul')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.season_length = season_length
        self.seasonal_type = seasonal_type
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.visualizer = ForecastVisualizer(str(self.output_dir))
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        # State variables
        self.merged_data: Optional[pd.DataFrame] = None
        self.aggregated_train: Optional[pd.DataFrame] = None
        self.aggregated_test: Optional[pd.DataFrame] = None
        self.model: Optional[AggregatedHoltWinters] = None
        self.forecasts: Optional[np.ndarray] = None
        self.metrics: Optional[Dict[str, float]] = None
        self.comparison_results: Optional[Dict] = None
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load all data sources and prepare merged dataset.
        
        Returns:
            Merged DataFrame with all features
        """
        logger.info("Loading and preparing data...")
        
        # Load all data sources
        self.data_loader.load_train_files()
        self.data_loader.load_calendar()
        self.data_loader.load_sell_prices()
        self.data_loader.load_validation()
        self.data_loader.load_test()
        
        # Merge into unified dataset
        self.merged_data = self.data_loader.merge_data(include_validation=True)
        
        logger.info(f"Merged data shape: {self.merged_data.shape}")
        logger.info(f"Date range: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        
        return self.merged_data
    
    def create_aggregated_series(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create aggregated daily sales time series for training and testing.
        
        Returns:
            Tuple of (train aggregated, test/validation aggregated)
        """
        logger.info("Creating aggregated time series...")
        
        if self.merged_data is None:
            raise ValueError("Must call load_and_prepare_data first")
        
        # Split by date to separate train and validation periods
        # Validation data starts from late 2025 based on dataset inspection
        validation_start = pd.Timestamp('2025-11-01')
        
        train_mask = self.merged_data['date'] < validation_start
        test_mask = self.merged_data['date'] >= validation_start
        
        # Aggregate by date
        train_daily = (
            self.merged_data[train_mask]
            .groupby('date')['sales']
            .sum()
            .reset_index()
        )
        train_daily.columns = ['date', 'total_sales']
        
        test_daily = (
            self.merged_data[test_mask]
            .groupby('date')['sales']
            .sum()
            .reset_index()
        )
        test_daily.columns = ['date', 'total_sales']
        
        self.aggregated_train = train_daily
        self.aggregated_test = test_daily
        
        logger.info(f"Training series length: {len(train_daily)}")
        logger.info(f"Test series length: {len(test_daily)}")
        
        return train_daily, test_daily
    
    def fit_model(self, optimize_params: bool = True) -> AggregatedHoltWinters:
        """
        Fit the Holt-Winters model to training data.
        
        Args:
            optimize_params: Whether to optimize smoothing parameters
            
        Returns:
            Fitted model instance
        """
        logger.info("Fitting Holt-Winters model...")
        
        if self.aggregated_train is None:
            self.create_aggregated_series()
        
        dates = self.aggregated_train['date'].values
        sales = self.aggregated_train['total_sales'].values
        
        # Create model
        self.model = AggregatedHoltWinters(
            season_length=self.season_length,
            seasonal_type=self.seasonal_type,
            alpha=0.3,
            beta=0.1,
            gamma=0.2
        )
        
        # Fit with optional parameter optimization
        if optimize_params:
            logger.info("Optimizing smoothing parameters...")
            self.model.model.optimize_parameters(sales)
        else:
            self.model.fit(dates, sales)
        
        logger.info(f"Model fitted. Final level: {self.model.model.level:.4f}")
        logger.info(f"Final trend: {self.model.model.trend:.4f}")
        
        return self.model
    
    def generate_forecasts(self, steps: Optional[int] = None) -> np.ndarray:
        """
        Generate forecasts for the test period.
        
        Args:
            steps: Number of steps to forecast. If None, uses length of test set.
            
        Returns:
            Array of forecasted values
        """
        logger.info("Generating forecasts...")
        
        if self.model is None:
            raise ValueError("Must call fit_model first")
        
        if steps is None:
            steps = len(self.aggregated_test)
        
        self.forecasts = self.model.model.predict(steps)
        
        # Ensure non-negative forecasts
        self.forecasts = np.maximum(self.forecasts, 0)
        
        logger.info(f"Generated {steps} forecasts")
        
        return self.forecasts
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model performance against actual values and baselines.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        if self.forecasts is None:
            self.generate_forecasts()
        
        actual = self.aggregated_test['total_sales'].values
        predicted = self.forecasts[:len(actual)]
        training_actual = self.aggregated_train['total_sales'].values
        
        # Create comparator
        comparator = ModelComparator(actual, training_actual)
        
        # Evaluate Holt-Winters
        hw_metrics = comparator.evaluate_model('HoltWinters', predicted)
        
        # Run baseline comparisons
        comparator.compare_baselines()
        
        # Get comparison table
        comparison_df = comparator.get_comparison_table()
        
        # Calculate improvements
        improvements = {}
        for baseline in ['Naive', 'Seasonal_Naive']:
            if baseline in comparator.results:
                improvements[f'improvement_over_{baseline}'] = comparator.calculate_improvement(
                    'HoltWinters', baseline, 'MAPE'
                )
        
        # Store results
        self.metrics = hw_metrics
        self.comparison_results = {
            'comparison_table': comparison_df,
            'all_results': comparator.results,
            'improvements': improvements,
            'report': comparator.generate_report()
        }
        
        logger.info(f"Holt-Winters MAPE: {hw_metrics['MAPE']:.4f}%")
        logger.info(f"Improvement over Naive: {improvements.get('improvement_over_Naive', 0):.2f}x")
        
        return hw_metrics
    
    def create_visualizations(self) -> List[str]:
        """
        Generate all visualization outputs.
        
        Returns:
            List of saved plot filenames
        """
        logger.info("Creating visualizations...")
        
        if self.model is None or self.forecasts is None:
            raise ValueError("Must fit model and generate forecasts first")
        
        saved_files = []
        
        # Prepare data for plotting
        train_dates = pd.to_datetime(self.aggregated_train['date']).values
        train_sales = self.aggregated_train['total_sales'].values
        test_dates = pd.to_datetime(self.aggregated_test['date']).values
        test_sales = self.aggregated_test['total_sales'].values
        
        # Main comparison plot
        fig1 = self.visualizer.plot_actual_vs_forecast(
            train_dates=train_dates[-60:],
            train_actual=train_sales[-60:],
            test_dates=test_dates,
            test_actual=test_sales,
            test_forecast=self.forecasts[:len(test_sales)],
            mape_value=self.metrics.get('MAPE', 0),
            title=f'Sales Forecasting: Actual vs Predicted\nMAPE: {self.metrics.get("MAPE", 0):.2f}%',
            save_name='actual_vs_forecast'
        )
        saved_files.append('actual_vs_forecast.png')
        plt.close(fig1)
        
        # Decomposition plot
        components = self.model.model.get_components()
        fig2 = self.visualizer.plot_time_series_decomposition(
            dates=train_dates,
            actual=train_sales,
            level=components['level'],
            trend=components['trend'],
            seasonal=components['seasonal'],
            residuals=components['residuals'],
            title='Holt-Winters Time Series Decomposition',
            save_name='decomposition'
        )
        saved_files.append('decomposition.png')
        plt.close(fig2)
        
        # Residual analysis
        fig3 = self.visualizer.plot_residual_analysis(
            residuals=components['residuals'],
            fitted_values=components['fitted'],
            dates=train_dates,
            title='Model Residual Analysis',
            save_name='residual_analysis'
        )
        saved_files.append('residual_analysis.png')
        plt.close(fig3)
        
        # Seasonal pattern
        fig4 = self.visualizer.plot_seasonal_pattern(
            seasonal_components=components['seasonal'],
            season_length=self.season_length,
            title='Learned Weekly Seasonal Pattern',
            save_name='seasonal_pattern'
        )
        saved_files.append('seasonal_pattern.png')
        plt.close(fig4)
        
        # Model comparison
        if self.comparison_results and 'comparison_table' in self.comparison_results:
            fig5 = self.visualizer.plot_model_comparison(
                metrics_df=self.comparison_results['comparison_table'],
                title='Model Performance Comparison',
                save_name='model_comparison'
            )
            saved_files.append('model_comparison.png')
            plt.close(fig5)
        
        logger.info(f"Created {len(saved_files)} visualizations")
        
        return saved_files
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all report outputs.
        
        Returns:
            Dictionary mapping report type to file path
        """
        logger.info("Generating reports...")
        
        if self.metrics is None:
            raise ValueError("Must evaluate model first")
        
        # Create full report package
        report_paths = self.report_generator.create_full_report_package(
            metrics=self.metrics,
            comparison_table=self.comparison_results['comparison_table'],
            improvements=self.comparison_results['improvements'],
            data_summary={
                'Training records': len(self.aggregated_train),
                'Test records': len(self.aggregated_test),
                'Total sales (train)': f"{self.aggregated_train['total_sales'].sum():,.0f}",
                'Avg daily sales (train)': f"{self.aggregated_train['total_sales'].mean():,.2f}",
                'Date range': f"{self.aggregated_train['date'].min()} to {self.aggregated_test['date'].max()}"
            }
        )
        
        # Create model card
        model_card_path = create_model_card(
            model_info={
                'type': 'Holt-Winters Triple Exponential Smoothing',
                'implementation': 'From Scratch',
                'author': self.AUTHOR_NAME
            },
            output_dir=str(self.output_dir)
        )
        report_paths['model_card'] = model_card_path
        
        logger.info(f"Generated {len(report_paths)} reports")
        
        return report_paths
    
    def run_full_pipeline(self) -> Dict:
        """
        Execute the complete forecasting pipeline.
        
        Returns:
            Dictionary with all pipeline outputs and results
        """
        logger.info("=" * 60)
        logger.info("STARTING TIME SERIES SALES FORECASTING PIPELINE")
        logger.info(f"Author: {self.AUTHOR_NAME}")
        logger.info(f"Portfolio Category: {self.PORTFOLIO_CATEGORY}")
        logger.info("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Create aggregated series
        self.create_aggregated_series()
        
        # Step 3: Fit model
        self.fit_model(optimize_params=True)
        
        # Step 4: Generate forecasts
        self.generate_forecasts()
        
        # Step 5: Evaluate model
        self.evaluate_model()
        
        # Step 6: Create visualizations
        viz_files = self.create_visualizations()
        
        # Step 7: Generate reports
        report_files = self.generate_reports()
        
        # Compile results
        results = {
            'author': self.AUTHOR_NAME,
            'portfolio_category': self.PORTFOLIO_CATEGORY,
            'metrics': self.metrics,
            'improvements': self.comparison_results['improvements'],
            'visualizations': viz_files,
            'reports': report_files,
            'comparison_report': self.comparison_results['report']
        }
        
        # Save summary JSON
        summary_path = self.output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'author': self.AUTHOR_NAME,
                'portfolio_category': self.PORTFOLIO_CATEGORY,
                'metrics': self.metrics,
                'improvements': self.comparison_results['improvements'],
                'status': 'SUCCESS'
            }, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"MAPE Achieved: {self.metrics['MAPE']:.4f}%")
        logger.info(f"Improvement over Naive: {self.comparison_results['improvements'].get('improvement_over_Naive', 0):.2f}x")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point for the pipeline."""
    # Determine paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'Datasets'
    output_dir = script_dir / 'reports'
    
    # Run pipeline
    pipeline = ForecastingPipeline(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        season_length=7,
        seasonal_type='add'
    )
    
    results = pipeline.run_full_pipeline()
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Author: {results['author']}")
    print(f"Portfolio Category: {results['portfolio_category']}")
    print(f"\nKey Metrics:")
    print(f"  - MAPE: {results['metrics']['MAPE']:.4f}%")
    print(f"  - RMSE: {results['metrics']['RMSE']:.4f}")
    print(f"  - MAE: {results['metrics']['MAE']:.4f}")
    print(f"\nBaseline Comparison:")
    print(f"  - Improvement over Naive: {results['improvements'].get('improvement_over_Naive', 0):.2f}x")
    print(f"  - Improvement over Seasonal Naive: {results['improvements'].get('improvement_over_Seasonal_Naive', 0):.2f}x")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    
    main()
