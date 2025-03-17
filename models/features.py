import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import ta

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering for memecoin price prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', [3, 6, 12, 24])
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw price data
        
        Args:
            df: DataFrame with price data including timestamp, price, volume, token
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
            
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Sort by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Price-based features
        self._add_price_features(df)
        
        # Volume-based features
        self._add_volume_features(df)
        
        # Pattern-based features
        self._add_pattern_features(df)
        
        # Trend-based features
        self._add_trend_features(df)
        
        # Technical indicators
        self._add_technical_indicators(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
        
    def _add_price_features(self, df: pd.DataFrame) -> None:
        """Add price-based features"""
        # Returns over different periods
        for period in self.lookback_periods:
            df[f'price_return_{period}'] = df['close'].pct_change(period)
            df[f'price_high_return_{period}'] = df['high'].pct_change(period)
            df[f'price_low_return_{period}'] = df['low'].pct_change(period)
            
        # Price volatility
        for period in self.lookback_periods:
            df[f'price_volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            
        # Price momentum
        df['price_momentum'] = df['close'] - df['close'].rolling(12).mean()
        df['price_acceleration'] = df['price_momentum'].diff()
        
        # Candlestick features
        df['candle_body'] = df['close'] - df['open']
        df['candle_upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['candle_lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_size'] = df['high'] - df['low']
        
    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """Add volume-based features"""
        # Volume changes
        for period in self.lookback_periods:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
            
        # Volume moving averages
        for period in self.lookback_periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            
        # Volume price trend
        df['volume_price_trend'] = df['volume'] * df['close'].pct_change().abs()
        
        # Volume intensity
        df['volume_intensity'] = df['volume'] / df['volume'].rolling(20).mean()
        
    def _add_pattern_features(self, df: pd.DataFrame) -> None:
        """Add pattern-based features"""
        # Swing detection
        for period in self.lookback_periods:
            df[f'pattern_swing_high_{period}'] = (
                df['high'].rolling(period, center=True).apply(
                    lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0
                )
            )
            df[f'pattern_swing_low_{period}'] = (
                df['low'].rolling(period, center=True).apply(
                    lambda x: 1 if x.iloc[len(x)//2] == min(x) else 0
                )
            )
            
        # Support and resistance levels
        for period in self.lookback_periods:
            df[f'pattern_support_{period}'] = df['low'].rolling(period).min()
            df[f'pattern_resistance_{period}'] = df['high'].rolling(period).max()
            
        # Price levels
        for period in self.lookback_periods:
            pivot = (df['high'].rolling(period).max() + 
                    df['low'].rolling(period).min() + 
                    df['close'].rolling(period).mean()) / 3
            df[f'pattern_pivot_{period}'] = pivot
            
        # Breakout detection
        for period in self.lookback_periods:
            resistance = df['high'].rolling(period).max()
            support = df['low'].rolling(period).min()
            df[f'pattern_breakout_{period}'] = (
                ((df['close'] > resistance.shift(1)) & (df['volume'] > df['volume'].rolling(period).mean())).astype(int) -
                ((df['close'] < support.shift(1)) & (df['volume'] > df['volume'].rolling(period).mean())).astype(int)
            )
            
    def _add_trend_features(self, df: pd.DataFrame) -> None:
        """Add trend-based features"""
        # Moving averages
        for period in self.lookback_periods:
            df[f'trend_ma_{period}'] = df['close'].rolling(period).mean()
            
        # Moving average crossovers
        for i, period1 in enumerate(self.lookback_periods[:-1]):
            for period2 in self.lookback_periods[i+1:]:
                ma1 = df['close'].rolling(period1).mean()
                ma2 = df['close'].rolling(period2).mean()
                df[f'trend_ma_cross_{period1}_{period2}'] = (ma1 > ma2).astype(int)
                
        # Micro trend detection
        df['trend_micro'] = (
            (df['close'] > df['close'].rolling(3).mean()) &
            (df['close'].rolling(3).mean() > df['close'].rolling(6).mean())
        ).astype(int)
        
        # Volatility regime
        df['trend_volatility_regime'] = pd.qcut(
            df['close'].pct_change().rolling(20).std(),
            q=3,
            labels=['low', 'medium', 'high']
        )
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add technical indicators"""
        # RSI
        df['tech_rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['tech_macd'] = macd.macd()
        df['tech_macd_signal'] = macd.macd_signal()
        df['tech_macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['tech_bb_high'] = bollinger.bollinger_hband()
        df['tech_bb_mid'] = bollinger.bollinger_mavg()
        df['tech_bb_low'] = bollinger.bollinger_lband()
        
        # Average True Range
        df['tech_atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['tech_stoch_k'] = stoch.stoch()
        df['tech_stoch_d'] = stoch.stoch_signal()
        
    def select_features(self, features: pd.DataFrame, target_column: str = 'future_return_6') -> pd.DataFrame:
        """
        Select most important features for a given target
        
        Args:
            features: DataFrame with all engineered features
            target_column: Target column to use for feature selection
            
        Returns:
            DataFrame with selected features
        """
        # Drop non-feature columns
        exclude_cols = ['timestamp', 'token']
        drop_cols = [col for col in exclude_cols if col in features.columns]
        feature_cols = features.drop(columns=drop_cols)
        
        # Exclude target columns except the one we're predicting
        target_cols = [col for col in feature_cols.columns if col.startswith('future_return_') or col.startswith('target_up_')]
        other_targets = [col for col in target_cols if col != target_column]
        
        # Return selected features and target
        X = feature_cols.drop(columns=other_targets)
        
        return X 