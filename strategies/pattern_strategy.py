"""Pattern-based trading strategy"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class PatternStrategy:
    """Strategy that uses candlestick patterns and technical indicators to generate signals"""
    
    def __init__(self, config: Dict):
        """Initialize the strategy with configuration parameters"""
        self.config = config
        
        # Pattern strategy parameters
        strategy_params = config.get('pattern_strategy', {})
        self.use_margin = strategy_params.get('use_margin', False)
        self.max_leverage = strategy_params.get('max_leverage', 3.0)
        
        # Pattern recognition parameters
        self.breakout_lookback = strategy_params.get('breakout_lookback', 20)
        self.volatility_lookback = strategy_params.get('volatility_lookback', 20)
        self.rsi_period = strategy_params.get('rsi_period', 14)
        self.rsi_overbought = strategy_params.get('rsi_overbought', 70)
        self.rsi_oversold = strategy_params.get('rsi_oversold', 30)
        
        # MACD parameters
        self.macd_fast = strategy_params.get('macd_fast', 12)
        self.macd_slow = strategy_params.get('macd_slow', 26)
        self.macd_signal = strategy_params.get('macd_signal', 9)
        
        # Bollinger Bands parameters
        self.bb_period = strategy_params.get('bb_period', 20)
        self.bb_std = strategy_params.get('bb_std', 2)
        
        # Strategy settings
        self.confirmation_needed = strategy_params.get('confirmation_needed', 2)
        self.min_price_action_score = strategy_params.get('min_price_action_score', 2)
        self.min_indicator_score = strategy_params.get('min_indicator_score', 2)
        self.pattern_score_threshold = strategy_params.get('pattern_score_threshold', 4)
        
        # Risk management
        self.stop_loss_atr_multiplier = strategy_params.get('stop_loss_atr_multiplier', 1.5)
        self.take_profit_atr_multiplier = strategy_params.get('take_profit_atr_multiplier', 3.0)
        self.trailing_stop_activation = strategy_params.get('trailing_stop_activation', 0.02)
        self.trailing_stop_distance = strategy_params.get('trailing_stop_distance', 0.015)
        
        # ML enhancement
        self.use_ml = strategy_params.get('use_ml', False)
        self.ml_models = {}
        self.ml_horizons = [1, 3, 5]  # Prediction horizons in bars
        self.ml_features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width', 'atr',
            'volume_change', 'price_change', 'volatility_5',
            'volatility_20', 'ema_9', 'sma_20', 'sma_50'
        ]
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and patterns for the strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated indicators and patterns
        """
        df = df.copy()
        
        # Ensure we have the required columns and convert timestamp to datetime if needed
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Map columns if they have different names
        column_mapping = {}
        for col in required_cols:
            # Try to find a matching column ignoring case
            matches = [c for c in df.columns if c.lower() == col]
            if matches:
                column_mapping[matches[0]] = col
                
        # Apply column mapping if needed
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Convert timestamp to datetime if it exists and is not already datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            except Exception as e:
                logger.warning(f"Error converting timestamps: {e}")
            
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Volume changes
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            
        # Calculate indicators
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_moving_averages(df)
        df = self._calculate_atr(df)
        df = self._calculate_volatility(df)
        
        # Calculate candlestick patterns
        df = self._calculate_candlestick_patterns(df)
        
        # Calculate support and resistance levels
        df = self._calculate_support_resistance(df)
        
        # Calculate trend indicators
        df = self._calculate_trend_indicators(df)
        
        # ML predictions are handled separately during signal generation
        # We don't need to call _add_ml_predictions here
            
        return df
        
    def get_signals(self, data: pd.DataFrame, portfolio: Dict) -> Dict:
        """
        Generate trading signals based on pattern recognition
        
        Args:
            data: DataFrame with price data and calculated indicators
            portfolio: Current portfolio state
            
        Returns:
            Dictionary with trading signals for each token
        """
        signals = {}
        
        for token in data['token'].unique():
            token_data = data[data['token'] == token].copy()
            
            if token_data.empty:
                continue
                
            # Skip if we don't have enough data
            if len(token_data) < 20:
                continue
                
            # Train ML models if not already trained
            if self.use_ml and token not in [t for t, _ in self.ml_models.keys()]:
                self._train_ml_models(token_data, token)
                
            # Get latest data point
            latest = token_data.iloc[-1]
            
            # Calculate pattern score for long and short signals
            long_score, long_reasons = self._calculate_long_score(latest, token_data, token)
            short_score, short_reasons = self._calculate_short_score(latest, token_data, token)
            
            # Calculate risk parameters
            stop_loss, take_profit = self._calculate_risk_parameters(latest, token_data)
            
            # Determine signal direction and confidence
            signal = 0
            confidence = 0
            reasons = []
            leverage = 1.0
            position_size = 0.0
            
            # Long signal
            if long_score > self.pattern_score_threshold and long_score >= short_score:
                signal = 1
                confidence = min(0.5 + (long_score - self.pattern_score_threshold) / 10, 0.95)
                reasons = long_reasons
                leverage = self._calculate_position_leverage(confidence, 'long')
                position_size = self._calculate_position_size(confidence, latest)
            
            # Short signal
            elif short_score > self.pattern_score_threshold and short_score > long_score:
                signal = -1
                confidence = min(0.5 + (short_score - self.pattern_score_threshold) / 10, 0.95)
                reasons = short_reasons
                leverage = self._calculate_position_leverage(confidence, 'short')
                position_size = self._calculate_position_size(confidence, latest)
                
            # Store signal data
            signals[token] = {
                'signal': signal,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reasons': reasons,
                'leverage': leverage if self.use_margin else 1.0,
                'position_size': position_size
            }
            
        return signals
        
    def _calculate_position_size(self, confidence: float, latest: pd.Series) -> float:
        """
        Calculate position size based on confidence and market conditions
        
        Args:
            confidence: Signal confidence score (0-1)
            latest: Latest market data
            
        Returns:
            Position size as a percentage of capital (0-1)
        """
        # Get base position size from config
        base_size = self.config['trading'].get('base_position_size', 0.25)
        max_size = self.config['trading'].get('max_position_size', 0.40)
        min_size = self.config['trading'].get('min_position_size', 0.10)
        
        # Start with base position size
        position_size = base_size
        
        # Adjust based on confidence
        confidence_multiplier = 1.0 + (confidence - 0.5) * 2 if confidence > 0.5 else 0.5
        position_size *= confidence_multiplier
        
        # Adjust based on volatility
        if 'volatility_5' in latest:
            vol_ratio = latest['volatility_5'] / latest['volatility_20'] if 'volatility_20' in latest else 1.0
            if vol_ratio > 1.5:  # High volatility
                position_size *= 0.8  # Reduce position size
            elif vol_ratio < 0.7:  # Low volatility
                position_size *= 1.2  # Increase position size
                
        # Adjust based on trend strength
        if latest.get('strong_trend', 0) == 1:
            position_size *= 1.2
            
        # Adjust based on ML confidence if available
        if self.use_ml:
            ml_threshold = self.config['pattern_strategy'].get('ml_confidence_threshold', 0.65)
            ml_confidence = latest.get('ml_confidence', 0.5)
            if ml_confidence > ml_threshold:
                position_size *= 1.1
            elif ml_confidence < 0.4:
                position_size *= 0.9
                
        # Cap position size
        return min(max(position_size, min_size), max_size)
        
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI conditions
        df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
        df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
        
        return df
        
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_bullish_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_bearish_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return df
        
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['sma_20'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * self.bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Bollinger Band conditions
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).mean()
        df['price_above_upper'] = df['close'] > df['bb_upper']
        df['price_below_lower'] = df['close'] < df['bb_lower']
        
        return df
        
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # MA crossovers
        df['ema9_cross_sma20'] = (df['ema_9'] > df['sma_20']) & (df['ema_9'].shift(1) <= df['sma_20'].shift(1))
        df['sma20_cross_sma50'] = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        
        # Price vs MAs
        df['price_above_ema9'] = df['close'] > df['ema_9']
        df['price_above_sma20'] = df['close'] > df['sma_20']
        df['price_above_sma50'] = df['close'] > df['sma_50']
        
        return df
        
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Normalize ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
        
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price volatility"""
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        
        # Identify high volatility periods
        df['high_volatility'] = df['volatility_5'] > df['volatility_5'].rolling(window=20).mean() * 1.5
        
        return df
        
    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick patterns"""
        # Doji
        df['doji'] = abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1
        
        # Hammer
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        body = abs(df['close'] - df['open'])
        
        df['hammer'] = (lower_wick > body * 2) & (upper_wick < body * 0.5) & (body > 0)
        
        # Shooting Star
        df['shooting_star'] = (upper_wick > body * 2) & (lower_wick < body * 0.5) & (body > 0)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (df['open'] < df['close']) & (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1)) & (df['close'].shift(1) < df['open'].shift(1))
        df['bearish_engulfing'] = (df['open'] > df['close']) & (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1)) & (df['close'].shift(1) > df['open'].shift(1))
        
        # Harami
        df['bullish_harami'] = (df['open'].shift(1) > df['close'].shift(1)) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['close'] > df['open'])
        df['bearish_harami'] = (df['open'].shift(1) < df['close'].shift(1)) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['close'] < df['open'])
        
        # Morning/Evening Star (simplified)
        df['morning_star'] = (df['open'].shift(2) > df['close'].shift(2)) & df['doji'].shift(1) & (df['close'] > df['open']) & (df['close'] > df['close'].shift(2))
        df['evening_star'] = (df['open'].shift(2) < df['close'].shift(2)) & df['doji'].shift(1) & (df['close'] < df['open']) & (df['close'] < df['close'].shift(2))
        
        # Three White Soldiers / Three Black Crows
        df['three_white_soldiers'] = (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) & (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
        df['three_black_crows'] = (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) & (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
        
        # Additional patterns
        df['tweezer_top'] = (abs(df['high'] - df['high'].shift(1)) / df['high'] < 0.001) & (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))
        df['tweezer_bottom'] = (abs(df['low'] - df['low'].shift(1)) / df['low'] < 0.001) & (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))
        
        # Pattern strength - combines multiple patterns
        df['bullish_pattern_count'] = (df['hammer'] | 0) + (df['bullish_engulfing'] | 0) + (df['bullish_harami'] | 0) + (df['morning_star'] | 0) + (df['three_white_soldiers'] | 0) + (df['tweezer_bottom'] | 0)
        df['bearish_pattern_count'] = (df['shooting_star'] | 0) + (df['bearish_engulfing'] | 0) + (df['bearish_harami'] | 0) + (df['evening_star'] | 0) + (df['three_black_crows'] | 0) + (df['tweezer_top'] | 0)
        
        return df
        
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        # Use rolling max/min to identify support and resistance levels
        window = self.breakout_lookback
        
        df['recent_high'] = df['high'].rolling(window=window).max()
        df['recent_low'] = df['low'].rolling(window=window).min()
        
        # Create columns for being near support/resistance
        price_buffer = 0.01  # 1% buffer
        df['near_resistance'] = df['high'] > df['recent_high'].shift(1) * (1 - price_buffer)
        df['near_support'] = df['low'] < df['recent_low'].shift(1) * (1 + price_buffer)
        
        # Identify breakouts
        df['resistance_breakout'] = df['close'] > df['recent_high'].shift(1)
        df['support_breakdown'] = df['close'] < df['recent_low'].shift(1)
        
        return df
        
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend strength indicators"""
        # Trend direction using moving averages
        df['uptrend'] = (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])
        df['downtrend'] = (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])
        
        # Strength of trend using ADX (simplified)
        df['trend_direction'] = df['uptrend'].astype(int) - df['downtrend'].astype(int)
        df['adx'] = df['trend_direction'].abs().rolling(window=14).mean() * 10
        df['strong_trend'] = df['adx'] > 3
        
        # Trend momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['strong_momentum'] = df['momentum'].abs() > df['momentum'].rolling(window=20).std() * 2
        
        return df
        
    def _calculate_long_score(self, latest: pd.Series, data: pd.DataFrame, token: str) -> Tuple[float, List[str]]:
        """Calculate score for long entry signals"""
        score = 0.0
        reasons = []
        
        # Check for various bullish conditions
        
        # 1. Price action patterns
        if latest.get('bullish_engulfing', False):
            score += 1.0
            reasons.append("Bullish engulfing pattern")
        
        if latest.get('hammer', False):
            score += 0.8
            reasons.append("Hammer pattern")
            
        if latest.get('morning_star', False):
            score += 1.5
            reasons.append("Morning star pattern")
            
        if latest.get('three_white_soldiers', False):
            score += 2.0
            reasons.append("Three white soldiers pattern")
            
        if latest.get('tweezer_bottom', False):
            score += 1.0
            reasons.append("Tweezer bottom pattern")
        
        # 2. Technical indicators
        if latest.get('rsi_oversold', False):
            score += 1.0
            reasons.append("RSI oversold")
            
        if latest.get('macd_bullish_cross', False):
            score += 1.5
            reasons.append("MACD bullish crossover")
            
        if latest.get('price_below_lower', False):
            score += 1.0
            reasons.append("Price below lower Bollinger Band")
            
        if latest.get('resistance_breakout', False):
            score += 2.0
            reasons.append("Breakout above resistance")
            
        if latest.get('sma20_cross_sma50', False):
            score += 1.5
            reasons.append("SMA20 crossed above SMA50")
            
        # 3. Trend and momentum
        if latest.get('uptrend', False):
            score += 1.0
            reasons.append("Uptrend on moving averages")
            
        if latest.get('strong_momentum', False) and latest.get('momentum', 0) > 0:
            score += 1.0
            reasons.append("Strong bullish momentum")
            
        # 4. Support/Resistance
        if latest.get('near_support', False):
            score += 1.0
            reasons.append("Price near a support level")
            
        # 5. Volatility conditions
        if latest.get('bb_squeeze', False):
            score += 0.5
            reasons.append("Bollinger Band squeeze (potential breakout)")
            
        # 6. ML model predictions
        if self.use_ml and token in [t for t, _ in self.ml_models.keys()]:
            for horizon in self.ml_horizons:
                model_key = (token, horizon)
                if model_key in self.ml_models:
                    model = self.ml_models[model_key]
                    
                    # Prepare features for prediction
                    features = self.ml_features
                    X = pd.DataFrame([latest[features].fillna(0).to_dict()])
                    
                    # Make prediction
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]
                    
                    # Add to score if bullish prediction with high confidence
                    if pred == 1 and prob[1] > 0.6:
                        score += 1.0 * prob[1]
                        reasons.append(f"ML model predicts upward move (score: {prob[1]:.1f})")
            
        return score, reasons
        
    def _calculate_short_score(self, latest: pd.Series, data: pd.DataFrame, token: str) -> Tuple[float, List[str]]:
        """Calculate score for short entry signals"""
        score = 0.0
        reasons = []
        
        # Check for various bearish conditions
        
        # 1. Price action patterns
        if latest.get('bearish_engulfing', False):
            score += 1.0
            reasons.append("Bearish engulfing pattern")
        
        if latest.get('shooting_star', False):
            score += 0.8
            reasons.append("Shooting star pattern")
            
        if latest.get('evening_star', False):
            score += 1.5
            reasons.append("Evening star pattern")
            
        if latest.get('three_black_crows', False):
            score += 2.0
            reasons.append("Three black crows pattern")
            
        if latest.get('tweezer_top', False):
            score += 1.0
            reasons.append("Tweezer top pattern")
        
        # 2. Technical indicators
        if latest.get('rsi_overbought', False):
            score += 1.0
            reasons.append("RSI overbought")
            
        if latest.get('macd_bearish_cross', False):
            score += 1.5
            reasons.append("MACD bearish crossover")
            
        if latest.get('price_above_upper', False):
            score += 1.0
            reasons.append("Price above upper Bollinger Band")
            
        if latest.get('support_breakdown', False):
            score += 2.0
            reasons.append("Breakdown below support")
            
        # 3. Trend and momentum
        if latest.get('downtrend', False):
            score += 1.0
            reasons.append("Downtrend on moving averages")
            
        if latest.get('strong_momentum', False) and latest.get('momentum', 0) < 0:
            score += 1.0
            reasons.append("Strong downtrend")
            
        # 4. Support/Resistance
        if latest.get('near_resistance', False):
            score += 1.0
            reasons.append("Price near a resistance level")
            
        # 5. Stochastic conditions (if available)
        if 'stoch_k' in latest and 'stoch_d' in latest:
            if latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                score += 1.0
                reasons.append("Stochastic overbought")
                
            if latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'].shift(1) >= latest['stoch_d'].shift(1):
                score += 1.0
                reasons.append("Stochastic bearish crossover")
            
        # 6. ML model predictions
        if self.use_ml and token in [t for t, _ in self.ml_models.keys()]:
            for horizon in self.ml_horizons:
                model_key = (token, horizon)
                if model_key in self.ml_models:
                    model = self.ml_models[model_key]
                    
                    # Prepare features for prediction
                    features = self.ml_features
                    X = pd.DataFrame([latest[features].fillna(0).to_dict()])
                    
                    # Make prediction
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]
                    
                    # Add to score if bearish prediction with high confidence
                    if pred == 0 and prob[0] > 0.6:
                        score += 1.0 * prob[0]
                        reasons.append(f"ML model predicts downward move (score: {prob[0]:.1f})")
            
        return score, reasons
        
    def _calculate_risk_parameters(self, latest: pd.Series, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels based on ATR"""
        # Use ATR for determining stop loss and take profit
        atr = latest.get('atr_pct', 0.01)  # Default to 1% if ATR not available
        
        stop_loss = atr * self.stop_loss_atr_multiplier
        take_profit = atr * self.take_profit_atr_multiplier
        
        return stop_loss, take_profit
    
    def _train_ml_models(self, data: pd.DataFrame, token: str) -> None:
        """Train ML models for trend prediction"""
        if len(data) < 100:
            logger.warning(f"Not enough data to train ML models for {token}")
            return
            
        try:
            # Prepare features and labels for training
            features = self.ml_features
            
            # Filter rows with NaN values
            df_ml = data.dropna(subset=features)
            
            if len(df_ml) < 100:
                logger.warning(f"Not enough clean data to train ML models for {token}")
                return
                
            X = df_ml[features]
            
            # Train models for different horizons
            for horizon in self.ml_horizons:
                # Create labels: 1 if price goes up after horizon bars, 0 otherwise
                df_ml[f'future_return_{horizon}'] = df_ml['close'].shift(-horizon) / df_ml['close'] - 1
                df_ml[f'label_{horizon}'] = (df_ml[f'future_return_{horizon}'] > 0).astype(int)
                
                # Filter out NaN labels
                mask = ~df_ml[f'label_{horizon}'].isna()
                X_filtered = X[mask]
                y = df_ml[f'label_{horizon}'][mask]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
                
                # Train Random Forest model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"ML model for {token} horizon {horizon} accuracy: {accuracy:.4f}")
                
                # Store model
                self.ml_models[(token, horizon)] = model
                
                # Add ML predictions to dataframe
                latest = data.iloc[-1]
                features_dict = {f: latest.get(f, 0) for f in features}
                X_latest = pd.DataFrame([features_dict])
                
                # Add prediction probability to the latest row
                prob = model.predict_proba(X_latest)[0][1]  # Probability of class 1 (up)
                data.loc[data.index[-1], f'ml_pred_{horizon}'] = prob
                
            # Calculate average ML confidence
            ml_columns = [f'ml_pred_{h}' for h in self.ml_horizons if f'ml_pred_{h}' in data.columns]
            if ml_columns:
                data.loc[data.index[-1], 'ml_confidence'] = data.loc[data.index[-1], ml_columns].mean()
                
        except Exception as e:
            logger.error(f"Error training ML models for {token}: {e}")
            
    def _calculate_position_leverage(self, confidence: float, direction: str) -> float:
        """
        Calculate position leverage based on signal confidence
        
        Args:
            confidence: Signal confidence (0-1)
            direction: 'long' or 'short'
            
        Returns:
            Leverage multiplier
        """
        # Base leverage is 1.0
        base_leverage = 1.0
        
        # Increase leverage based on confidence
        confidence_factor = (confidence - 0.5) * 2 if confidence > 0.5 else 0.0
        base_leverage += confidence_factor * self.max_leverage
        
        # Cap leverage
        return min(base_leverage, self.max_leverage)
        
    def optimize_parameters(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Optimize strategy parameters using historical data
        
        Args:
            historical_data: Dictionary of token -> DataFrame with historical data
            
        Returns:
            Dictionary with optimized parameters
        """
        # Placeholder for parameter optimization
        # In a real implementation, this would test different parameter combinations
        # and return the best performing set
        return self.config['pattern_strategy'] 