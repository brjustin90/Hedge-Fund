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
            
        Returns:
            Dict with trading signals for each token
        """
        try:
            signals = {}
            
            # Update portfolio tokens
            self.portfolio_tokens = set(portfolio['positions'].keys())
            
            # Process each token
            for token, token_data in data.groupby('token'):
                # Skip tokens with less than 24 data points (not enough history)
                if len(token_data) < 24:
                    continue
                    
                # Get ML prediction
                ml_result = self.ml_predictor.predict(token_data, token)
                ml_signal = ml_result['signal']
                ml_confidence = ml_result['confidence']
                
                # Skip if ML confidence is too low
                if ml_confidence < self.confidence_threshold:
                    continue
                    
                # Determine if we should generate a signal
                generate_signal = False
                
                # For tokens not in portfolio, look for buy signals
                if token not in self.portfolio_tokens and ml_signal > 0:
                    generate_signal = True
                    action = 'buy'
                    
                # For tokens in portfolio, check for sell signals if enabled
                elif token in self.portfolio_tokens and self.use_stoploss_prediction and ml_signal < 0:
                    generate_signal = True
                    action = 'sell'
                    
                # Generate signal if conditions met
                if generate_signal:
                    # Get price from data
                    price_col = 'price' if 'price' in token_data.columns else 'close'
                    if price_col in token_data.columns:
                        price = token_data[price_col].iloc[-1]
                        
                        signals[token] = {
                            'action': action,
                            'price': price,
                            'confidence': ml_confidence,
                            'prediction': ml_result['prediction']
                        }
                        
                        logger.info(f"ML {action.upper()} signal for {token} at {price:.6f} with confidence {ml_confidence:.4f}")
                    else:
                        logger.warning(f"No price data found for {token}, skipping signal generation")
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return {}
            
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