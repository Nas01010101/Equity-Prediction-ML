# Equity Correction Forecasting

A machine learning ensemble model that predicts equity price corrections using technical indicators, market regime detection, and statistical significance testing.

## Overview

This project implements an ensemble of XGBoost, Random Forest, and Logistic Regression models to forecast 5%+ price corrections over a 5-day horizon. It includes comprehensive feature engineering, time series cross-validation, and statistical testing to validate model performance.

## Features

- **24+ Engineered Features**: Technical indicators (RSI, Bollinger Bands, moving averages), volatility metrics, momentum indicators, and relative performance vs market benchmarks
- **Ensemble Model**: Weighted combination of XGBoost (40%), Random Forest (40%), and Logistic Regression (20%)
- **Market Regime Detection**: Composite scoring system (0-5) identifying extreme market conditions
- **Statistical Rigor**: Bootstrap confidence intervals, significance testing vs random baselines
- **Time Series Validation**: Proper temporal cross-validation to prevent data leakage

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Usage Example

```python
from equity_prediction_ml import MarketDataLoader, RegimeFeatureEngineer, CorrectionPredictor

# Load data
loader = MarketDataLoader(symbol='NVDA')
equity_data = loader.load_equity_data('2020-01-01')
benchmark_data = loader.load_benchmark_data('2020-01-01')

# Engineer features
engineer = RegimeFeatureEngineer()
feature_data = engineer.create_features(equity_data, benchmark_data)
engineered_data = engineer.create_regime_features(feature_data)
engineered_data = engineer.create_target_variable(engineered_data)

# Train and predict
model = CorrectionPredictor()
X, y, features = model.prepare_features(engineered_data)
model.train_models(X, y, features)
predictions = model.generate_predictions(engineered_data)
```

## Project Structure

```
equity_prediction_ml/
├── __init__.py              # Package exports
├── main.py                  # Main execution script
├── data_module.py           # Data loading and feature engineering
├── model.py                 # Ensemble ML model
├── statistical_tests.py     # Statistical significance testing
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── .gitignore              # Git ignore rules
```

## Results

- **Model Accuracy**: 90.74% (vs 76.14% random baseline)
- **Statistical Significance**: ✓ YES (p < 0.05)
- **Returns**: Significantly negative (-10.71% average, p < 0.05)
- **Precision**: 100% on positive predictions

## Dependencies

- pandas, numpy
- scikit-learn, xgboost
- yfinance
- scipy

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

**For educational and research purposes only.** Not financial advice. Trading involves substantial risk.