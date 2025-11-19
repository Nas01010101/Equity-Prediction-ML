"""
Equity Correction Forecasting

A machine learning research project for forecasting equity price corrections
using ensemble methods and feature engineering.

This package implements:
- Data loading and preprocessing
- Feature engineering for market regimes
- Ensemble ML models (XGBoost, Random Forest, Logistic Regression)
- Time series cross-validation
- Prediction generation and evaluation
"""

__version__ = "1.0.0"

from .data_module import MarketDataLoader, RegimeFeatureEngineer
from .model import CorrectionPredictor
from .statistical_tests import StatisticalTester

__all__ = [
    'MarketDataLoader',
    'RegimeFeatureEngineer',
    'CorrectionPredictor',
    'StatisticalTester',
]