"""
Statistical significance testing for model evaluation.
Tests whether model predictions are significantly better than random baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats
from scipy.stats import ttest_1samp, mannwhitneyu
try:
    from scipy.stats import binomtest  # Newer scipy versions
except ImportError:
    from scipy.stats import binom_test as binomtest  # Older scipy versions
import warnings

warnings.filterwarnings('ignore')


class StatisticalTester:
    """Statistical testing for model performance evaluation."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize statistical tester.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def bootstrap_confidence_interval(
        self, 
        data: np.ndarray, 
        statistic: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Array of observations
            statistic: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (statistic value, lower bound, upper bound)
        """
        n = len(data)
        bootstrap_samples = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_samples.append(statistic(sample))
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
        point_estimate = statistic(data)
        
        return point_estimate, lower, upper
    
    def test_vs_random_baseline(
        self,
        model_predictions: np.ndarray,
        true_labels: np.ndarray,
        n_trials: int = 1000
    ) -> Dict[str, float]:
        """
        Test if model accuracy is significantly better than random baseline.
        
        Args:
            model_predictions: Binary model predictions (0 or 1)
            true_labels: True binary labels (0 or 1)
            n_trials: Number of random trials for baseline
            
        Returns:
            Dictionary with test results
        """
        # Calculate model accuracy
        model_accuracy = (model_predictions == true_labels).mean()
        
        # Generate random baseline accuracies
        n_samples = len(true_labels)
        positive_rate = true_labels.mean()
        
        random_accuracies = []
        for _ in range(n_trials):
            # Random predictions with same class distribution
            random_preds = np.random.binomial(1, positive_rate, n_samples)
            random_acc = (random_preds == true_labels).mean()
            random_accuracies.append(random_acc)
        
        random_accuracies = np.array(random_accuracies)
        baseline_accuracy = random_accuracies.mean()
        
        # Statistical test: Compare model accuracy to random baseline distribution
        # Use z-test approximation (since we have many random trials)
        if random_accuracies.std() > 0:
            z_stat = (model_accuracy - baseline_accuracy) / random_accuracies.std()
            # Two-tailed p-value from z-statistic
            from scipy.stats import norm
            p_value = 1 - norm.cdf(z_stat)  # One-tailed: model > baseline
        else:
            # If no variance, check if model is better
            p_value = 0.0 if model_accuracy > baseline_accuracy else 1.0
            z_stat = 0.0
        
        # Calculate effect size (Cohen's d)
        effect_size = (model_accuracy - baseline_accuracy) / random_accuracies.std()
        
        # Bootstrap confidence interval for model accuracy
        ci_mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(
            (model_predictions == true_labels).astype(float)
        )
        
        return {
            'model_accuracy': model_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'improvement': model_accuracy - baseline_accuracy,
            'z_statistic': z_stat if random_accuracies.std() > 0 else 0.0,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'baseline_std': random_accuracies.std()
        }
    
    def test_return_significance(
        self,
        returns: np.ndarray,
        null_hypothesis: float = 0.0
    ) -> Dict[str, float]:
        """
        Test if average returns are significantly different from null hypothesis.
        
        Args:
            returns: Array of returns
            null_hypothesis: Null hypothesis value (default: 0.0)
            
        Returns:
            Dictionary with test results
        """
        # One-sample t-test
        t_stat, p_value = ttest_1samp(returns, null_hypothesis, alternative='less')
        
        # Bootstrap confidence interval
        ci_mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(returns)
        
        # Effect size (Cohen's d)
        effect_size = (returns.mean() - null_hypothesis) / returns.std()
        
        return {
            'mean_return': returns.mean(),
            'median_return': np.median(returns),
            'std_return': returns.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_observations': len(returns)
        }
    
    def test_classification_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Comprehensive statistical tests for classification performance.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Accuracy test vs random
        accuracy_test = self.test_vs_random_baseline(y_pred, y_true)
        results['accuracy_test'] = accuracy_test
        
        # Precision and recall
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Binomial test for precision (is it better than random?)
        if tp + fp > 0:
            # Random precision would be equal to positive class rate
            positive_rate = y_true.mean()
            # Use binomtest (newer scipy) or binom_test (older)
            try:
                result = binomtest(tp, tp + fp, positive_rate, alternative='greater')
                precision_p_value = result.pvalue
            except TypeError:
                # Fallback for older scipy versions
                precision_p_value = binomtest(tp, tp + fp, positive_rate)
        else:
            precision_p_value = 1.0
        
        results['classification_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'precision_p_value': precision_p_value,
            'precision_significant': precision_p_value < 0.05,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        return results
    
    def test_prediction_returns(
        self,
        predicted_returns: np.ndarray,
        benchmark_returns: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Test if predicted returns are significantly negative and better than benchmark.
        
        Args:
            predicted_returns: Returns on days with positive predictions
            benchmark_returns: Benchmark returns for comparison (optional)
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Test if returns are significantly negative
        return_test = self.test_return_significance(predicted_returns, null_hypothesis=0.0)
        results['return_test'] = return_test
        
        # Compare to benchmark if provided
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Mann-Whitney U test (non-parametric)
            u_stat, mw_p_value = mannwhitneyu(
                predicted_returns,
                benchmark_returns,
                alternative='less'
            )
            
            results['benchmark_comparison'] = {
                'predicted_mean': predicted_returns.mean(),
                'benchmark_mean': benchmark_returns.mean(),
                'difference': predicted_returns.mean() - benchmark_returns.mean(),
                'u_statistic': u_stat,
                'p_value': mw_p_value,
                'is_significant': mw_p_value < 0.05
            }
        
        return results
    
    def generate_statistical_report(
        self,
        model_predictions: np.ndarray,
        true_labels: np.ndarray,
        predicted_returns: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, Dict]:
        """
        Generate comprehensive statistical report.
        
        Args:
            model_predictions: Binary model predictions
            true_labels: True binary labels
            predicted_returns: Returns on prediction days
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all statistical test results
        """
        report = {}
        
        # Classification performance tests
        classification_results = self.test_classification_performance(
            true_labels, model_predictions, y_proba
        )
        report['classification'] = classification_results
        
        # Return significance tests
        return_results = self.test_prediction_returns(predicted_returns)
        report['returns'] = return_results
        
        # Overall significance summary
        report['summary'] = {
            'model_better_than_random': classification_results['accuracy_test']['is_significant'],
            'returns_significantly_negative': return_results['return_test']['is_significant'],
            'precision_better_than_random': classification_results['classification_metrics']['precision_significant']
        }
        
        return report
