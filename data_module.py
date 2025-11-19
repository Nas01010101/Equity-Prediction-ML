"""
Data loading and preprocessing for equity price prediction.
Fetches stock data and market benchmarks for feature engineering.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class MarketDataLoader:
    """Loads and preprocesses equity and market data for ML prediction."""
    
    def __init__(self, symbol: str = 'NVDA'):
        """
        Initialize data loader.
        
        Args:
            symbol: Stock symbol to analyze (default: NVDA)
        """
        self.symbol = symbol
        self.benchmarks = {
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QQQ': 'Invesco QQQ Trust',
            'SOXX': 'iShares Semiconductor ETF',
            'XLK': 'Technology Select Sector SPDR Fund',
        }
    
    def load_equity_data(self, start_date: str = '2007-01-01',
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load equity stock data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            
        Returns:
            DataFrame with OHLCV data and technical indicators
            
        Raises:
            ValueError: If no data is retrieved
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                raise ValueError(f"No data retrieved for {self.symbol} from {start_date} to {end_date}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
        except Exception as e:
            raise ValueError(f"Error fetching {self.symbol} data: {str(e)}")
    
    def load_benchmark_data(self, start_date: str = '2007-01-01', 
                           end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load market benchmark data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            
        Returns:
            Dictionary mapping symbol to DataFrame with market data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        benchmark_data = {}
        
        for symbol, name in self.benchmarks.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not data.empty:
                    benchmark_data[symbol] = data
            except Exception:
                continue
        
        return benchmark_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        df = data.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Moving averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # Price ratios
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        df['price_sma200_ratio'] = df['Close'] / df['sma_200']
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'], 20, 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower


class RegimeFeatureEngineer:
    """Engineers regime indicators and targets for price correction prediction."""

    def __init__(self):
        """Initialize regime thresholds."""
        self.thresholds = {
            'price_sma_ratio': 2.0,
            'rsi_extreme': 80,
            'volatility_spike': 2.0,
            'momentum_extreme': 0.5,
            'volume_spike': 3.0,
            'relative_strength': 1.5,
        }

    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market regime indicator features."""
        df = data.copy()

        # Price regime indicators
        df['price_regime'] = (
            (df['price_sma200_ratio'] > self.thresholds['price_sma_ratio']) |
            (df['price_sma50_ratio'] > 1.5) |
            (df['price_sma20_ratio'] > 1.2)
        ).astype(int)

        # Momentum regime indicators
        df['momentum_regime'] = (
            (df['rsi_14'] > self.thresholds['rsi_extreme']) |
            (df['price_momentum_3m'] > self.thresholds['momentum_extreme']) |
            (df['price_momentum_6m'] > 0.8)
        ).astype(int)

        # Volatility regime indicators
        df['volatility_regime'] = (
            (df['vol_regime_change'] > self.thresholds['volatility_spike']) |
            (df['bb_width'] > df['bb_width'].rolling(252).quantile(0.9))
        ).astype(int)

        # Volume regime indicators
        df['volume_regime'] = (
            (df['volume_ratio'] > self.thresholds['volume_spike']) |
            (df['extreme_up_days'] >= 3)
        ).astype(int)

        # Relative strength regime indicators
        df['relative_regime'] = (
            (df['equity_spy_ratio'] > df['equity_spy_ratio'].rolling(252).quantile(0.95)) |
            (df['equity_qqq_ratio'] > df['equity_qqq_ratio'].rolling(252).quantile(0.95)) |
            (df['equity_soxx_ratio'] > df['equity_soxx_ratio'].rolling(252).quantile(0.95))
        ).astype(int)

        # Composite regime score
        regime_indicators = [
            'price_regime',
            'momentum_regime',
            'volatility_regime',
            'volume_regime',
            'relative_regime',
        ]
        df['regime_score'] = df[regime_indicators].sum(axis=1)

        # Regime severity labels
        df['regime_severity'] = pd.cut(
            df['regime_score'],
            bins=[-1, 0, 1, 2, 3, 5],
            labels=['None', 'Low', 'Medium', 'High', 'Extreme'],
        )

        return df

    def create_target_variable(
        self,
        data: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Create binary target based on future returns."""
        df = data.copy()

        df['future_return_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['future_return_20d'] = df['Close'].shift(-20) / df['Close'] - 1

        df['target'] = (df['future_return_5d'] < -threshold).astype(int)
        return df

    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime and target distribution metrics."""
        metrics: Dict[str, float] = {}

        if 'regime_score' in data.columns:
            metrics['regime_frequency'] = (data['regime_score'] >= 3).mean()
            metrics['extreme_regime_frequency'] = (data['regime_score'] >= 4).mean()

        if 'target' in data.columns:
            target_data = data[data['target'].notna()]
            if len(target_data) > 0:
                metrics['target_positive_rate'] = target_data['target'].mean()

        if 'volatility_20' in data.columns:
            metrics['avg_volatility'] = data['volatility_20'].mean()
            metrics['max_volatility'] = data['volatility_20'].max()
            metrics['volatility_spike_frequency'] = (data['vol_regime_change'] > 1.5).mean()

        if 'price_momentum_3m' in data.columns:
            metrics['avg_3m_momentum'] = data['price_momentum_3m'].mean()
            metrics['extreme_momentum_frequency'] = (data['price_momentum_3m'] > 0.5).mean()

        return metrics

    def get_regime_periods(self, data: pd.DataFrame, min_duration: int = 5) -> List[Tuple]:
        """Identify continuous regime periods."""
        regime_periods: List[Tuple] = []
        in_regime = False
        regime_start = None

        for date, row in data.iterrows():
            is_regime = row.get('regime_score', 0) >= 3

            if is_regime and not in_regime:
                regime_start = date
                in_regime = True
            elif not is_regime and in_regime:
                if regime_start is not None:
                    duration = (date - regime_start).days
                    if duration >= min_duration:
                        regime_periods.append((regime_start, date, duration))
                in_regime = False
                regime_start = None

        if in_regime and regime_start is not None:
            duration = (data.index[-1] - regime_start).days
            if duration >= min_duration:
                regime_periods.append((regime_start, data.index[-1], duration))

        return regime_periods
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def create_features(self, equity_data: pd.DataFrame, 
                       benchmark_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features for machine learning.
        
        Args:
            equity_data: Equity stock data with technical indicators
            benchmark_data: Dictionary of benchmark data
            
        Returns:
            DataFrame with engineered features
        """
        df = equity_data.copy()
        
        # Relative performance vs market
        if 'SPY' in benchmark_data:
            spy_data = benchmark_data['SPY']
            spy_returns = spy_data['Close'].pct_change()
            df['equity_spy_ratio'] = df['Close'] / spy_data['Close']
            df['equity_spy_returns_diff'] = df['returns'] - spy_returns
            df['equity_spy_volatility_ratio'] = df['volatility_20'] / spy_data['Close'].pct_change().rolling(20).std()
        
        # Tech sector performance
        if 'QQQ' in benchmark_data:
            qqq_data = benchmark_data['QQQ']
            df['equity_qqq_ratio'] = df['Close'] / qqq_data['Close']
            df['equity_qqq_returns_diff'] = df['returns'] - qqq_data['Close'].pct_change()
        
        # Sector performance
        if 'SOXX' in benchmark_data:
            soxx_data = benchmark_data['SOXX']
            df['equity_soxx_ratio'] = df['Close'] / soxx_data['Close']
            df['equity_soxx_returns_diff'] = df['returns'] - soxx_data['Close'].pct_change()
        
        # Momentum features
        df['price_momentum_3m'] = df['Close'] / df['Close'].shift(63) - 1
        df['price_momentum_6m'] = df['Close'] / df['Close'].shift(126) - 1
        df['price_momentum_1y'] = df['Close'] / df['Close'].shift(252) - 1
        
        # Volatility regime
        df['vol_regime_change'] = df['volatility_20'] / df['volatility_60']
        
        # Extreme movements
        df['extreme_up_days'] = (df['returns'] > df['returns'].rolling(252).quantile(0.95)).rolling(5).sum()
        df['extreme_down_days'] = (df['returns'] < df['returns'].rolling(252).quantile(0.05)).rolling(5).sum()
        
        # Price acceleration
        df['price_acceleration'] = df['returns'].diff()
        
        return df
