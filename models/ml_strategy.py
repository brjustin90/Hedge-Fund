import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)

class MLStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.ml_params = config['ml_strategy_params']
        self.pattern_params = config['pattern_params']
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        
    def train(self, data: Dict[str, pd.DataFrame]) -> None:
        """Train ML models for each token"""
        for token, df in data.items():
            logger.info(f"Training models for {token}")
            
            # Prepare features and target
            X, y = self._prepare_training_data(df)
            
            if len(X) == 0:
                logger.warning(f"No training data available for {token}")
                continue
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[token] = scaler
            
            # Train models
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            rf_model.fit(X_scaled, y)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_scaled, y)
            
            self.models[token] = {
                'rf': rf_model,
                'xgb': xgb_model
            }
            
            logger.info(f"Trained models for {token}")
            
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Select relevant features
        feature_cols = [col for col in df.columns if col.startswith(('price_', 'volume_', 'pattern_', 'trend_'))]
        
        if not feature_cols:
            return np.array([]), np.array([])
            
        X = df[feature_cols].values
        
        # Create target based on future returns
        future_returns = df['close'].pct_change(periods=6).shift(-6)  # 30-minute forward returns
        y = (future_returns > self.ml_params['min_pattern_score']).astype(int)
        
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y
        
    def get_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals for each token"""
        signals = {}
        
        for token, df in data.items():
            if token not in self.models:
                logger.warning(f"No trained models available for {token}")
                continue
                
            # Get ML predictions
            X = df[[col for col in df.columns if col.startswith(('price_', 'volume_', 'pattern_', 'trend_'))]].values
            X_scaled = self.scalers[token].transform(X)
            
            rf_pred = self.models[token]['rf'].predict_proba(X_scaled)[:, 1]
            xgb_pred = self.models[token]['xgb'].predict_proba(X_scaled)[:, 1]
            
            # Combine predictions
            ml_signal = (rf_pred + xgb_pred) / 2
            
            # Get pattern-based signals
            pattern_signal = self._get_pattern_signals(df)
            
            # Calculate combined signal
            combined_signal = (
                self.ml_params['prediction_weight'] * ml_signal +
                self.ml_params['pattern_weight'] * pattern_signal
            )
            
            # Generate entry/exit signals
            entry_signal = (combined_signal > self.ml_params['confidence_threshold']).astype(int)
            exit_signal = self._get_exit_signals(df, entry_signal)
            
            # Store signals
            signals[token] = pd.DataFrame({
                'timestamp': df['timestamp'],
                'price': df['close'],
                'ml_signal': ml_signal,
                'pattern_signal': pattern_signal,
                'combined_signal': combined_signal,
                'entry_signal': entry_signal,
                'exit_signal': exit_signal
            })
            
        return signals
        
    def _get_pattern_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate signals based on price patterns"""
        # Initialize signal array
        pattern_signal = np.zeros(len(df))
        
        # Detect swing points
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()
        
        # Detect breakouts
        upper_band = df['high'].rolling(window=20).max()
        lower_band = df['low'].rolling(window=20).min()
        
        # Detect volume spikes
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_spike = df['volume'] > (volume_ma * self.pattern_params['volume_spike_threshold'])
        
        # Detect micro trends
        short_ma = df['close'].rolling(window=5).mean()
        long_ma = df['close'].rolling(window=20).mean()
        micro_trend = (short_ma > long_ma).astype(float)
        
        # Combine pattern signals
        for i in range(4, len(df)-4):
            # Bullish swing
            if (df['low'].iloc[i] > lows.iloc[i] * (1 + self.pattern_params['swing_threshold']) and
                volume_spike.iloc[i] and
                micro_trend.iloc[i] > 0):
                pattern_signal[i] = 1
                
            # Bearish swing
            elif (df['high'].iloc[i] < highs.iloc[i] * (1 - self.pattern_params['swing_threshold']) and
                  volume_spike.iloc[i] and
                  micro_trend.iloc[i] < 0):
                pattern_signal[i] = -1
                
            # Breakout
            if (df['close'].iloc[i] > upper_band.iloc[i] * (1 + self.pattern_params['breakout_threshold']) and
                volume_spike.iloc[i]):
                pattern_signal[i] = 1
            elif (df['close'].iloc[i] < lower_band.iloc[i] * (1 - self.pattern_params['breakout_threshold']) and
                  volume_spike.iloc[i]):
                pattern_signal[i] = -1
                
        return pattern_signal
        
    def _get_exit_signals(self, df: pd.DataFrame, entry_signal: np.ndarray) -> np.ndarray:
        """Generate exit signals based on price action and risk management"""
        exit_signal = np.zeros(len(df))
        
        # Track active trades
        in_trade = False
        entry_price = 0
        entry_idx = 0
        
        for i in range(1, len(df)):
            if not in_trade and entry_signal[i] == 1:
                # Enter trade
                in_trade = True
                entry_price = df['close'].iloc[i]
                entry_idx = i
                
            elif in_trade:
                current_price = df['close'].iloc[i]
                price_change = (current_price - entry_price) / entry_price
                
                # Check exit conditions
                stop_loss = price_change < -self.config['stop_loss_pct']
                take_profit = price_change > self.config['take_profit_pct']
                trailing_stop = (
                    price_change > self.config['min_profit_to_trail'] and
                    (current_price - df['close'].iloc[i-1]) / df['close'].iloc[i-1] < -self.config['trailing_stop_pct']
                )
                
                # Exit trade if any condition is met
                if stop_loss or take_profit or trailing_stop:
                    exit_signal[i] = 1
                    in_trade = False
                    
                # Force exit if held too long
                elif i - entry_idx > 12:  # 1 hour maximum hold time
                    exit_signal[i] = 1
                    in_trade = False
                    
        return exit_signal 