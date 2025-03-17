import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from models.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)

class MLStrategy:
    """ML-enhanced strategy selector that combines ML predictions with traditional indicators"""
    
    def __init__(self, config: Dict):
        self.config = config
        ml_config = config.get('ml', {})
        self.ml_predictor = MLPredictor(ml_config)
        self.prediction_weight = ml_config.get('prediction_weight', 0.7)
        self.indicator_weight = 1 - self.prediction_weight
        self.confidence_threshold = ml_config.get('confidence_threshold', 0.65)
        self.use_stoploss_prediction = ml_config.get('use_stoploss_prediction', True)
        self.portfolio_tokens = set()  # Track tokens in portfolio
        
    def train_models(self, historical_data_dict: Dict[str, pd.DataFrame]):
        """
        Train ML models for all tokens
        
        Args:
            historical_data_dict: Dictionary with historical data for each token
        """
        logger.info(f"Training ML models for {len(historical_data_dict)} tokens")
        results = self.ml_predictor.batch_train(historical_data_dict)
        
        # Log training results
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"Successfully trained {success_count}/{len(results)} models")
        
        # Log metrics for successful models
        for token, result in results.items():
            if result.get('success', False):
                metrics = result.get('metrics', {})
                logger.info(f"Model metrics for {token}: {metrics}")
                
    def get_signals(self, data: pd.DataFrame, portfolio: Dict) -> Dict:
        """
        Generate ML-enhanced trading signals
        
        Args:
            data: DataFrame with current price data
            portfolio: Current portfolio state
        """
        signals = {}
        
        # Get ML predictions
        predictions = self.ml_predictor.predict(data)
        
        # Pattern-based signals
        pattern_signals = self._get_pattern_signals(data)
        
        # Combine ML and pattern signals
        for token in data['token'].unique():
            token_data = data[data['token'] == token].iloc[-1]
            
            # Skip if we don't have predictions for this token
            if token not in predictions:
                continue
                
            pred = predictions[token]
            pattern = pattern_signals.get(token, {})
            
            # Calculate combined signal
            ml_signal = pred.get('signal', 0) * self.prediction_weight
            pattern_signal = pattern.get('signal', 0) * self.indicator_weight
            combined_signal = ml_signal + pattern_signal
            
            # Calculate confidence score
            ml_confidence = pred.get('confidence', 0)
            pattern_confidence = pattern.get('confidence', 0)
            combined_confidence = (ml_confidence * self.prediction_weight + 
                                pattern_confidence * self.indicator_weight)
            
            # Entry conditions
            entry_signal = False
            if token not in portfolio and combined_confidence > self.confidence_threshold:
                if combined_signal > 0:  # Bullish signal
                    # Check for confirmation from patterns
                    if (pattern.get('swing_low', False) or 
                        pattern.get('support_break', False) or 
                        pattern.get('volume_spike', False)):
                        entry_signal = True
                        
            # Exit conditions
            exit_signal = False
            if token in portfolio:
                position = portfolio[token]
                # Check for take profit or stop loss
                current_return = (token_data['close'] - position['entry_price']) / position['entry_price']
                
                if current_return <= -self.stop_loss_pct:
                    exit_signal = True
                    exit_reason = 'stop_loss'
                elif current_return >= self.take_profit_pct:
                    exit_signal = True
                    exit_reason = 'take_profit'
                # Check for pattern-based exits
                elif (pattern.get('swing_high', False) or 
                      pattern.get('resistance_break', False) or 
                      combined_signal < -0.5):
                    exit_signal = True
                    exit_reason = 'pattern'
                    
            signals[token] = {
                'signal': 1 if entry_signal else (-1 if exit_signal else 0),
                'confidence': combined_confidence,
                'entry_price': token_data['close'] if entry_signal else None,
                'stop_loss': token_data['close'] * (1 - self.stop_loss_pct) if entry_signal else None,
                'take_profit': token_data['close'] * (1 + self.take_profit_pct) if entry_signal else None,
                'exit_reason': exit_reason if exit_signal else None
            }
            
        return signals
        
    def _get_pattern_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate signals based on price patterns
        """
        signals = {}
        
        for token in data['token'].unique():
            token_data = data[data['token'] == token]
            
            # Skip if not enough data
            if len(token_data) < 12:  # Need at least 1 hour of data
                continue
                
            latest = token_data.iloc[-1]
            
            # Initialize signal components
            signal = 0
            confidence = 0
            patterns_found = []
            
            # Check for swing points
            if latest['swing_low']:
                signal += 1
                confidence += 0.2
                patterns_found.append('swing_low')
                
            if latest['swing_high']:
                signal -= 1
                confidence += 0.2
                patterns_found.append('swing_high')
                
            # Check for breakouts
            if latest['support_break']:
                signal -= 1
                confidence += 0.15
                patterns_found.append('support_break')
                
            if latest['resistance_break']:
                signal += 1
                confidence += 0.15
                patterns_found.append('resistance_break')
                
            # Check volume profile
            if 'volume_intensity' in latest:
                if latest['volume_intensity'] > 2.0:  # Volume spike
                    if latest['price_velocity'] > 0:
                        signal += 0.5
                        confidence += 0.1
                        patterns_found.append('volume_spike_up')
                    else:
                        signal -= 0.5
                        confidence += 0.1
                        patterns_found.append('volume_spike_down')
                        
            # Check micro trend
            if latest['micro_trend'] == 1:
                signal += 0.3
                confidence += 0.1
                patterns_found.append('micro_trend_up')
            elif latest['micro_trend'] == -1:
                signal -= 0.3
                confidence += 0.1
                patterns_found.append('micro_trend_down')
                
            # Normalize signal to [-1, 1] range
            signal = np.clip(signal, -1, 1)
            
            # Normalize confidence to [0, 1] range
            confidence = min(confidence, 1.0)
            
            signals[token] = {
                'signal': signal,
                'confidence': confidence,
                'patterns': patterns_found,
                'swing_low': latest['swing_low'],
                'swing_high': latest['swing_high'],
                'support_break': latest['support_break'],
                'resistance_break': latest['resistance_break'],
                'volume_spike': latest.get('volume_intensity', 1.0) > 2.0
            }
            
        return signals
        
    def optimize_parameters(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Optimize ML strategy parameters using historical data
        
        Args:
            historical_data: Dictionary of historical price data by token
            
        Returns:
            Dict with optimized parameters
        """
        # Implementation depends on optimization approach
        # Could use cross-validation, grid search, or bayesian optimization
        logger.info("Parameter optimization not implemented yet")
        return {
            'prediction_weight': 0.7,
            'confidence_threshold': 0.65
        }
        
    def calculate_entry_exit_params(self, token: str, price_data: pd.DataFrame) -> Dict:
        """
        Calculate optimal entry/exit parameters for a token
        
        Args:
            token: Token symbol
            price_data: Historical price data
            
        Returns:
            Dict with entry/exit parameters
        """
        try:
            # Calculate price volatility
            price_col = 'price' if 'price' in price_data.columns else 'close'
            returns = price_data[price_col].pct_change().dropna()
            volatility = returns.std()
            
            # Adjust stop loss based on volatility
            # Higher volatility = wider stop loss
            stop_loss = min(0.10, max(0.05, volatility * 3))
            
            # Adjust take profit based on volatility
            # Higher volatility = higher take profit
            take_profit = max(0.15, min(0.40, volatility * 6))
            
            # Adjust trailing stop based on volatility
            trailing_stop = min(0.08, max(0.03, volatility * 2))
            
            # Adjust entry confirmation based on volatility
            # More volatile = require more confirmations
            entry_confirmation = 1 if volatility < 0.05 else 2
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'entry_confirmation': entry_confirmation
            }
            
        except Exception as e:
            logger.error(f"Error calculating parameters for {token}: {e}")
            return {
                'stop_loss': 0.07,  # Default 7%
                'take_profit': 0.20,  # Default 20%
                'trailing_stop': 0.05,  # Default 5%
                'entry_confirmation': 1  # Default 1 confirmation
            } 