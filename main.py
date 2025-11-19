"""
Main execution script for Equity Correction Forecasting.

This script demonstrates a machine learning ensemble approach to:
1. Load and preprocess equity and market data
2. Engineer features for price prediction
3. Train ensemble ML models with time series validation
4. Evaluate model performance

Research-focused implementation for ML positions.
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

from data_module import MarketDataLoader, RegimeFeatureEngineer
from model import CorrectionPredictor
from statistical_tests import StatisticalTester


def main() -> Dict:
    """
    Main execution function for equity price correction prediction.
    
    Returns:
        Dictionary containing analysis results, models, and metrics
    """
    print("=" * 70)
    print("Equity Correction Forecasting")
    print("Machine Learning Ensemble Approach")
    print("=" * 70)
    
    # 1. Data Collection
    print("\n[1/5] Data Collection")
    print("-" * 70)
    loader = MarketDataLoader(symbol='NVDA')
    
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    try:
        equity_data = loader.load_equity_data(start_date)
        print(f"✓ Equity data: {len(equity_data)} days")
        print(f"  Date range: {equity_data.index[0].date()} to {equity_data.index[-1].date()}")
        
        benchmark_data = loader.load_benchmark_data(start_date)
        print(f"✓ Market benchmarks: {list(benchmark_data.keys())}")
        
        # Create relative performance features
        engineer = RegimeFeatureEngineer()
        feature_data = engineer.create_features(equity_data, benchmark_data)
        print(f"✓ Features engineered: {feature_data.shape[1]} features")
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        sys.exit(1)
    
    # 2. Feature Engineering
    print("\n[2/5] Feature Engineering")
    print("-" * 70)
    
    try:
        engineered_data = engineer.create_regime_features(feature_data)
        engineered_data = engineer.create_target_variable(engineered_data)
        
        metrics = engineer.calculate_metrics(engineered_data)
        
        print(f"✓ Regime frequency: {metrics.get('regime_frequency', 0):.2%}")
        print(f"✓ Extreme regime frequency: {metrics.get('extreme_regime_frequency', 0):.2%}")
        print(f"✓ Average volatility: {metrics.get('avg_volatility', 0):.2%}")
        print(f"✓ Target positive rate: {metrics.get('target_positive_rate', 0):.2%}")
        
    except Exception as e:
        print(f"✗ Error in feature engineering: {str(e)}")
        sys.exit(1)
    
    # 3. Machine Learning Model Training
    print("\n[3/5] Machine Learning Model Training")
    print("-" * 70)
    model = CorrectionPredictor()
    
    try:
        X, y, features = model.prepare_features(engineered_data)
        print(f"✓ Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Class distribution: {dict(zip(['Negative', 'Positive'], np.bincount(y)))}")
        
        model_scores = model.train_models(X, y, features)
        
        print("\n✓ Cross-Validation Results (ROC-AUC):")
        for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name.upper()}: {score:.3f}")
        
    except Exception as e:
        print(f"✗ Error in model training: {str(e)}")
        sys.exit(1)
    
    # 4. Prediction Generation and Evaluation
    print("\n[4/5] Prediction Generation and Evaluation")
    print("-" * 70)
    
    try:
        predictions = model.generate_predictions(engineered_data, probability_threshold=0.6)
        prediction_count = predictions['prediction'].sum()
        print(f"✓ Positive predictions: {prediction_count}")
        
        # Evaluate model performance
        performance = model.evaluate_performance(engineered_data)
        
        if 'error' not in performance:
            print(f"\n✓ Model Performance Metrics:")
            print(f"  Total predictions: {performance['total_predictions']}")
            print(f"  Accuracy: {performance['accuracy']:.2%}")
            print(f"  Average return: {performance['avg_return']:.2%}")
            print(f"  Sharpe ratio: {performance['sharpe_ratio']:.3f}")
            
            # Statistical significance testing
            print(f"\n✓ Statistical Significance Testing:")
            tester = StatisticalTester()
            
            # Test vs random baseline (on full dataset)
            if '_all_model_predictions' in performance and '_all_true_labels' in performance:
                accuracy_test = tester.test_vs_random_baseline(
                    performance['_all_model_predictions'],
                    performance['_all_true_labels']
                )
                
                print(f"  Model vs Random Baseline:")
                print(f"    Model accuracy: {accuracy_test['model_accuracy']:.2%}")
                print(f"    Random baseline: {accuracy_test['baseline_accuracy']:.2%}")
                print(f"    Improvement: {accuracy_test['improvement']:.2%}")
                print(f"    P-value: {accuracy_test['p_value']:.4f}")
                print(f"    Significant: {'✓ YES' if accuracy_test['is_significant'] else '✗ NO'} (p < 0.05)")
                print(f"    Effect size (Cohen's d): {accuracy_test['effect_size']:.3f}")
                print(f"    95% CI: [{accuracy_test['ci_lower']:.2%}, {accuracy_test['ci_upper']:.2%}]")
            
            # Test return significance
            if '_future_returns' in performance:
                return_test = tester.test_return_significance(performance['_future_returns'])
                
                print(f"\n  Return Significance Test:")
                print(f"    Mean return: {return_test['mean_return']:.2%}")
                print(f"    P-value (H0: return = 0): {return_test['p_value']:.4f}")
                print(f"    Significantly negative: {'✓ YES' if return_test['is_significant'] else '✗ NO'} (p < 0.05)")
                print(f"    Effect size (Cohen's d): {return_test['effect_size']:.3f}")
                print(f"    95% CI: [{return_test['ci_lower']:.2%}, {return_test['ci_upper']:.2%}]")
            
            # Comprehensive statistical report
            if all(key in performance for key in ['_all_model_predictions', '_all_true_labels', '_future_returns']):
                if '_all_prediction_probabilities' in performance:
                    stats_report = tester.generate_statistical_report(
                        performance['_all_model_predictions'],
                        performance['_all_true_labels'],
                        performance['_future_returns'],
                        performance['_all_prediction_probabilities']
                    )
                    
                    print(f"\n  Classification Performance:")
                    cls_metrics = stats_report['classification']['classification_metrics']
                    print(f"    Precision: {cls_metrics['precision']:.2%}")
                    print(f"    Recall: {cls_metrics['recall']:.2%}")
                    print(f"    F1-score: {cls_metrics['f1_score']:.2%}")
                    print(f"    Precision vs random (p-value): {cls_metrics['precision_p_value']:.4f}")
                    print(f"    Precision significant: {'✓ YES' if cls_metrics['precision_significant'] else '✗ NO'}")
                    
                    print(f"\n  Summary:")
                    summary = stats_report['summary']
                    print(f"    Model better than random: {'✓ YES' if summary['model_better_than_random'] else '✗ NO'}")
                    print(f"    Returns significantly negative: {'✓ YES' if summary['returns_significantly_negative'] else '✗ NO'}")
                    print(f"    Precision better than random: {'✓ YES' if summary['precision_better_than_random'] else '✗ NO'}")
        
    except Exception as e:
        print(f"✗ Error in prediction generation: {str(e)}")
        sys.exit(1)
    
    # 5. Feature Importance Analysis
    print("\n[5/5] Feature Importance Analysis")
    print("-" * 70)
    
    try:
        feature_importance = model.get_feature_importance()
        
        print("✓ Top 10 Most Important Features (XGBoost):")
        if 'xgb' in feature_importance:
            sorted_features = sorted(
                feature_importance['xgb'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for i, (feature, score) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature:30s} {score:.4f}")
        
    except Exception as e:
        print(f"✗ Error in feature analysis: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    
    return {
        'data': engineered_data,
        'predictions': predictions,
        'model': model,
        'metrics': metrics,
        'model_scores': model_scores,
        'performance': performance if 'error' not in performance else None
    }


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)