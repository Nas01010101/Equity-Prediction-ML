"""
Machine learning ensemble model for equity price correction prediction.
Uses XGBoost, Random Forest, and Logistic Regression in a weighted ensemble.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class CorrectionPredictor:
    """Ensemble machine learning model for equity price correction prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize prediction model.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_fitted = False
        
        # Model configurations - Ensemble of best-performing models
        self.model_configs = {
            'xgb': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            },
            'rf': {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'logistic': {
                'random_state': 42,
                'max_iter': 1000,
                'C': 0.1
            }
        }
    
    def prepare_features(self, data: pd.DataFrame, 
                        target_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets for model training.
        
        Args:
            data: DataFrame with features and target
            target_horizon: Prediction horizon in days
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define feature columns
        feature_cols = [
            'returns', 'volatility_20', 'volatility_60', 'rsi_14',
            'price_sma20_ratio', 'price_sma50_ratio', 'price_sma200_ratio',
            'bb_position', 'bb_width', 'volume_ratio',
            'price_momentum_3m', 'price_momentum_6m', 'price_momentum_1y',
            'vol_regime_change', 'extreme_up_days', 'extreme_down_days',
            'price_acceleration', 'equity_spy_ratio', 'equity_spy_returns_diff',
            'equity_qqq_ratio', 'equity_qqq_returns_diff', 'equity_soxx_ratio',
            'equity_soxx_returns_diff', 'regime_score'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in data.columns]
        
        # Clean data
        data_clean = data.dropna()
        if len(data_clean) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Create target if not exists
        if 'target' not in data_clean.columns:
            future_returns = data_clean['Close'].shift(-target_horizon) / data_clean['Close'] - 1
            target = (future_returns < -0.05).astype(int)
        else:
            target = data_clean['target']
        
        # Prepare features and targets
        X = data_clean[available_features].values
        y = target.values
        
        # Remove rows where target is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y, available_features
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    features: List[str]) -> Dict[str, float]:
        """
        Train ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target vector
            features: List of feature names
            
        Returns:
            Dictionary mapping model names to CV scores
        """
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_scores = {}
        
        for model_name, config in self.model_configs.items():
            # Initialize model
            if model_name == 'xgb':
                model = xgb.XGBClassifier(**config)
            elif model_name == 'rf':
                model = RandomForestClassifier(**config)
            elif model_name == 'logistic':
                model = LogisticRegression(**config)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[model_name] = scaler
            else:
                continue
            
            # Cross-validation
            if model_name == 'logistic':
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            
            # Train on full dataset
            if model_name == 'logistic':
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
            
            # Store model and scores
            self.models[model_name] = model
            model_scores[model_name] = cv_scores.mean()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = dict(zip(features, np.abs(model.coef_[0])))
        
        self.is_fitted = True
        return model_scores
    
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of price correction.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'logistic':
                X_scaled = self.scalers[model_name].transform(X)
                pred = model.predict_proba(X_scaled)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]
            
            predictions[model_name] = pred
        
        # Ensemble prediction (weighted average)
        weights = {'xgb': 0.4, 'rf': 0.4, 'logistic': 0.2}
        ensemble_pred = np.zeros(len(X))
        
        for model_name, pred in predictions.items():
            ensemble_pred += weights.get(model_name, 0) * pred
        
        return ensemble_pred
    
    def generate_predictions(self, data: pd.DataFrame, 
                            probability_threshold: float = 0.6,
                            target_horizon: int = 5) -> pd.DataFrame:
        """
        Generate predictions for the dataset.
        
        Args:
            data: DataFrame with features
            probability_threshold: Threshold for positive prediction
            target_horizon: Prediction horizon in days
            
        Returns:
            DataFrame with predictions added
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating predictions")
        
        # Prepare features
        X, _, features = self.prepare_features(data, target_horizon)
        
        # Get predictions
        probabilities = self.predict_probability(X)
        
        # Create predictions dataframe
        predictions_df = data.copy()
        predictions_df['prediction_probability'] = np.nan
        predictions_df['prediction'] = 0
        
        # Map predictions back to original data
        valid_indices = data.dropna().index
        for i, idx in enumerate(valid_indices):
            if i < len(probabilities):
                predictions_df.loc[idx, 'prediction_probability'] = probabilities[i]
                if probabilities[i] > probability_threshold:
                    predictions_df.loc[idx, 'prediction'] = 1
        
        return predictions_df
    
    def evaluate_performance(self, data: pd.DataFrame, 
                            target_horizon: int = 5) -> Dict:
        """
        Evaluate model performance on historical data.
        
        Args:
            data: DataFrame with features and targets
            target_horizon: Prediction horizon in days
            
        Returns:
            Dictionary of performance metrics
        """
        # Generate predictions for all data points
        predictions_df = self.generate_predictions(data, target_horizon=target_horizon)
        
        # Calculate true labels for all data points
        all_true_labels = []
        all_predictions = []
        all_probabilities = []
        all_returns = []
        
        for idx in predictions_df.index:
            future_idx = data.index.get_loc(idx) + target_horizon
            if future_idx < len(data):
                future_return = (data.iloc[future_idx]['Close'] / data.loc[idx, 'Close'] - 1)
                true_label = 1 if future_return < -0.05 else 0
                
                all_true_labels.append(true_label)
                all_predictions.append(predictions_df.loc[idx, 'prediction'])
                all_probabilities.append(predictions_df.loc[idx, 'prediction_probability'])
                all_returns.append(future_return)
        
        if len(all_returns) == 0:
            return {"error": "No valid future returns calculated"}
        
        all_true_labels = np.array(all_true_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_returns = np.array(all_returns)
        
        # Filter to positive predictions for return analysis
        positive_mask = all_predictions == 1
        positive_returns = all_returns[positive_mask]
        positive_predictions = predictions_df[predictions_df['prediction'] == 1]
        
        if len(positive_returns) == 0:
            return {"error": "No positive predictions generated"}
        
        # Performance metrics
        metrics = {
            'total_predictions': len(positive_predictions),
            'accuracy': (positive_returns < 0).mean(),
            'avg_return': positive_returns.mean(),
            'median_return': np.median(positive_returns),
            'max_return': positive_returns.max(),
            'min_return': positive_returns.min(),
            'sharpe_ratio': positive_returns.mean() / positive_returns.std() if positive_returns.std() > 0 else 0,
            'avg_probability': positive_predictions['prediction_probability'].mean(),
            # Store for statistical testing (full dataset)
            '_future_returns': positive_returns,
            '_all_true_labels': all_true_labels,
            '_all_model_predictions': all_predictions,
            '_all_prediction_probabilities': all_probabilities,
            '_all_returns': all_returns
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from all models.
        
        Returns:
            Dictionary mapping model names to feature importance dictionaries
        """
        return self.feature_importance
