import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.pattern_strategy import PatternStrategy

logger = logging.getLogger(__name__)

class StrategySelector:
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_weights = config.get('strategy_weights', {
            'trend_following': 0.3,
            'mean_reversion': 0.2,
            'momentum': 0.15,
            'liquidity': 0.05,
            'pattern': 0.3
        })
        
        # Initialize pattern strategy
        self.pattern_strategy = PatternStrategy(config)
        
    def get_signals(self, data: pd.DataFrame, portfolio: Dict) -> Dict:
        """Generate trading signals from all strategies"""
        try:
            signals = {}
            
            # Process data with pattern strategy
            processed_data = self._prepare_data_for_strategies(data)
            
            # Get pattern strategy signals
            pattern_signals = self.pattern_strategy.get_signals(processed_data, portfolio)
            
            for token, token_data in data.groupby('token'):
                # Skip if we already have a position in this token
                if token in portfolio['positions']:
                    continue
                    
                # Calculate signals from each strategy
                trend_signal = self._trend_following_strategy(token_data)
                mean_rev_signal = self._mean_reversion_strategy(token_data)
                momentum_signal = self._momentum_strategy(token_data)
                liquidity_signal = self._liquidity_strategy(token_data)
                
                # Get pattern signal (default to 0 if not available)
                pattern_signal = 0
                if token in pattern_signals:
                    pattern_signal = pattern_signals[token].get('signal', 0)
                
                # Combine signals using strategy weights
                combined_signal = (
                    trend_signal * self.strategy_weights['trend_following'] +
                    mean_rev_signal * self.strategy_weights['mean_reversion'] +
                    momentum_signal * self.strategy_weights['momentum'] +
                    liquidity_signal * self.strategy_weights['liquidity'] +
                    pattern_signal * self.strategy_weights['pattern']
                )
                
                # Get current price - check if 'close' exists, otherwise use 'price'
                if 'close' in token_data.columns:
                    price = token_data['close'].iloc[-1]
                elif 'price' in token_data.columns:
                    price = token_data['price'].iloc[-1]
                else:
                    logger.warning(f"No price data found for {token}, skipping signal generation")
                    continue
                    
                # Calculate confidence based on signal strength
                confidence = min(abs(combined_signal), 1.0)
                
                # Calculate position leverage if pattern strategy provided it
                leverage = 1.0
                if token in pattern_signals:
                    leverage = pattern_signals[token].get('leverage', 1.0)
                
                # Generate trading decision
                if abs(combined_signal) > 0.5:  # Threshold for taking action
                    # Extract stop loss and take profit from pattern strategy if available
                    stop_loss = None
                    take_profit = None
                    if token in pattern_signals:
                        stop_loss = pattern_signals[token].get('stop_loss')
                        take_profit = pattern_signals[token].get('take_profit')
                    
                    # Use default values if not provided by pattern strategy
                    if stop_loss is None:
                        stop_loss = 0.05  # Default 5% stop loss
                    if take_profit is None:
                        take_profit = 0.15  # Default 15% take profit
                        
                    signals[token] = {
                        'signal': 1 if combined_signal > 0 else -1,  # 1 for buy, -1 for sell
                        'confidence': confidence,
                        'price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'leverage': leverage
                    }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}
            
    def _prepare_data_for_strategies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for all strategies, especially the pattern strategy"""
        # Create a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Use the pattern strategy's prepare_data method
        return self.pattern_strategy.prepare_data(processed_data)
        
    def _trend_following_strategy(self, data: pd.DataFrame) -> float:
        """
        Simple trend following strategy based on moving average crossover
        
        Args:
            data: DataFrame with price data
            
        Returns:
            float: Signal strength (-1.0 to 1.0)
        """
        if len(data) < 24:  # Need at least 24 periods for longer MA
            return 0.0
            
        try:
            # Use 'close' if available, otherwise use 'price'
            price_col = 'close' if 'close' in data.columns else 'price'
            
            # Calculate moving averages
            ma_short = data[price_col].rolling(window=8).mean()
            ma_long = data[price_col].rolling(window=24).mean()
            
            # Generate signal based on crossover
            if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
                return 1.0  # Strong buy signal on upward crossover
            elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]:
                return -1.0  # Strong sell signal on downward crossover
            elif ma_short.iloc[-1] > ma_long.iloc[-1]:
                return 0.5  # Weak buy signal while above MA
            elif ma_short.iloc[-1] < ma_long.iloc[-1]:
                return -0.5  # Weak sell signal while below MA
            
            return 0.0
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return 0.0
            
    def _mean_reversion_strategy(self, data: pd.DataFrame) -> float:
        """
        Mean reversion strategy based on z-score
        
        Args:
            data: DataFrame with price data
            
        Returns:
            float: Signal strength (-1.0 to 1.0)
        """
        if len(data) < 24:  # Need at least 24 periods for reliable mean/std
            return 0.0
            
        try:
            # Use 'close' if available, otherwise use 'price'
            price_col = 'close' if 'close' in data.columns else 'price'
            
            # Calculate z-score
            rolling_mean = data[price_col].rolling(window=24).mean()
            rolling_std = data[price_col].rolling(window=24).std()
            z_score = (data[price_col] - rolling_mean) / rolling_std
            
            # Generate signal based on z-score
            current_z = z_score.iloc[-1]
            
            if pd.isna(current_z):
                return 0.0
                
            # Stronger signals at extreme z-scores
            if current_z <= -2.0:
                return 1.0  # Strong buy when very oversold
            elif current_z >= 2.0:
                return -1.0  # Strong sell when very overbought
            elif current_z <= -1.0:
                return 0.5  # Moderate buy when oversold
            elif current_z >= 1.0:
                return -0.5  # Moderate sell when overbought
            
            return 0.0
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return 0.0
            
    def _momentum_strategy(self, data: pd.DataFrame) -> float:
        """
        Momentum strategy based on recent returns
        
        Args:
            data: DataFrame with price data
            
        Returns:
            float: Signal strength (-1.0 to 1.0)
        """
        if len(data) < 5:  # Need at least 5 periods
            return 0.0
            
        try:
            # Use 'close' if available, otherwise use 'price'
            price_col = 'close' if 'close' in data.columns else 'price'
            
            # Calculate returns
            returns = data[price_col].pct_change()
            
            # Sum of last 5 period returns as momentum indicator
            momentum = returns.iloc[-5:].sum()
            
            # Generate signal based on momentum
            if pd.isna(momentum):
                return 0.0
                
            if momentum > 0.15:
                return 1.0  # Strong buy on high positive momentum
            elif momentum < -0.15:
                return -1.0  # Strong sell on high negative momentum
            elif momentum > 0.05:
                return 0.5  # Moderate buy on positive momentum
            elif momentum < -0.05:
                return -0.5  # Moderate sell on negative momentum
            
            return 0.0
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return 0.0
            
    def _liquidity_strategy(self, data: pd.DataFrame) -> float:
        """
        Liquidity-based strategy using liquidity score
        
        Args:
            data: DataFrame with price and liquidity data
            
        Returns:
            float: Signal strength (-1.0 to 1.0)
        """
        try:
            # Check if we have liquidity_score column
            if 'liquidity_score' not in data.columns:
                return 0.0
                
            # Get latest liquidity score
            liquidity_score = data['liquidity_score'].iloc[-1]
            
            if pd.isna(liquidity_score):
                return 0.0
                
            # Generate signal based on liquidity
            if liquidity_score > 0.8:
                return 0.8  # Strong buy for high liquidity
            elif liquidity_score > 0.5:
                return 0.4  # Moderate buy for medium liquidity
            elif liquidity_score < 0.2:
                return -0.8  # Strong sell for very low liquidity
            elif liquidity_score < 0.4:
                return -0.4  # Moderate sell for low liquidity
                
            return 0.0
        except Exception as e:
            logger.error(f"Error in liquidity strategy: {e}")
            return 0.0 