import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering for memecoin price prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', [8, 24, 72])
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw price data
        
        Args:
            df: DataFrame with price data including timestamp, price, volume, token
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Empty dataframe provided for feature engineering")
            return pd.DataFrame()
            
        # Ensure we have timestamp as a column
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp':
                df = df.reset_index()
            else:
                logger.warning("No timestamp column found in data")
                return pd.DataFrame()
                
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Use price column if available, otherwise close
        price_col = 'price' if 'price' in df.columns else 'close'
        if price_col not in df.columns:
            logger.warning(f"No price or close column found in data")
            return pd.DataFrame()
            
        features = df.copy()
        
        # Price momentum features
        for period in self.lookback_periods:
            # Returns over period
            features[f'return_{period}'] = features[price_col].pct_change(period)
            
            # Volatility over period
            features[f'volatility_{period}'] = features[price_col].pct_change().rolling(period).std()
            
            # Moving averages
            features[f'ma_{period}'] = features[price_col].rolling(period).mean()
            
            # Distance from moving average
            features[f'ma_dist_{period}'] = (features[price_col] / features[f'ma_{period}'] - 1)
            
            # Price acceleration (change in returns)
            features[f'accel_{period}'] = features[f'return_{period}'].diff()
            
        # Volume features
        if 'volume' in features.columns:
            for period in self.lookback_periods:
                # Volume momentum
                features[f'vol_change_{period}'] = features['volume'].pct_change(period)
                
                # Volume moving average
                features[f'vol_ma_{period}'] = features['volume'].rolling(period).mean()
                
                # Volume relative to moving average
                features[f'vol_ma_ratio_{period}'] = features['volume'] / features[f'vol_ma_{period}']
                
                # Price-volume correlation
                features[f'price_vol_corr_{period}'] = features[price_col].rolling(period).corr(features['volume'])
        
        # Liquidity features
        if 'liquidity_score' in features.columns:
            for period in self.lookback_periods:
                # Liquidity trend
                features[f'liq_change_{period}'] = features['liquidity_score'].pct_change(period)
                
                # Liquidity moving average
                features[f'liq_ma_{period}'] = features['liquidity_score'].rolling(period).mean()
        
        # Time-based features
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # RSI (Relative Strength Index)
        for period in [14, 28]:
            delta = features[price_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [20, 40]:
            features[f'bb_middle_{period}'] = features[price_col].rolling(period).mean()
            features[f'bb_std_{period}'] = features[price_col].rolling(period).std()
            features[f'bb_upper_{period}'] = features[f'bb_middle_{period}'] + 2 * features[f'bb_std_{period}']
            features[f'bb_lower_{period}'] = features[f'bb_middle_{period}'] - 2 * features[f'bb_std_{period}']
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / features[f'bb_middle_{period}']
            features[f'bb_pct_{period}'] = (features[price_col] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # MACD (Moving Average Convergence Divergence)
        short_window = 12
        long_window = 26
        signal_window = 9
        
        features['macd_short_ma'] = features[price_col].ewm(span=short_window, adjust=False).mean()
        features['macd_long_ma'] = features[price_col].ewm(span=long_window, adjust=False).mean()
        features['macd_line'] = features['macd_short_ma'] - features['macd_long_ma']
        features['macd_signal'] = features['macd_line'].ewm(span=signal_window, adjust=False).mean()
        features['macd_histogram'] = features['macd_line'] - features['macd_signal']
        
        # Target variable for prediction (future returns)
        for horizon in [1, 3, 6, 12, 24]:
            features[f'future_return_{horizon}'] = features[price_col].pct_change(horizon).shift(-horizon)
            # Binary targets for classification
            features[f'target_up_{horizon}'] = (features[f'future_return_{horizon}'] > 0).astype(int)
        
        # Drop NaN values 
        features = features.dropna(subset=[col for col in features.columns if col.startswith('future_return_')])
        
        return features
        
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