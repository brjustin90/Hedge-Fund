import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(self, config):
        self.config = config
        self.max_position_size = config['risk']['max_position_size']
        self.base_size = config['risk']['position_sizing']['base_size']
        self.max_size = config['risk']['position_sizing']['max_size']
        self.min_size = config['risk']['position_sizing']['min_size']
        
    def calculate_position_size(self, signal, conviction, volatility, available_capital):
        """Calculate appropriate position size"""
        try:
            # Start with base position size
            position_pct = self.base_size
            
            # Adjust for signal strength (0 to 1)
            signal_strength = abs(signal)
            position_pct *= (0.5 + 0.5 * signal_strength)  # Scale between 50-100% of base size
            
            # Adjust for conviction (0 to 1)
            position_pct *= (0.5 + 0.5 * conviction)  # Scale between 50-100% based on conviction
            
            # Adjust for volatility
            # Reduce position size as volatility increases
            vol_factor = 1.0 - min(0.5, volatility)  # Cap volatility reduction at 50%
            position_pct *= vol_factor
            
            # Ensure position size is within limits
            position_pct = max(self.min_size, min(self.max_size, position_pct))
            
            # Calculate actual position size in currency
            position_size = available_capital * position_pct
            
            return {
                'size': position_size,
                'pct_of_capital': position_pct,
                'metadata': {
                    'signal_strength': signal_strength,
                    'conviction': conviction,
                    'volatility': volatility,
                    'vol_factor': vol_factor
                }
            }
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'size': 0,
                'pct_of_capital': 0,
                'metadata': {}
            }
            
    def adjust_for_correlation(self, position_sizes, correlation_matrix):
        """Adjust position sizes based on portfolio correlation"""
        try:
            # Calculate portfolio risk contribution
            n = len(position_sizes)
            if n <= 1:
                return position_sizes
                
            # Create correlation-weighted position adjustments
            adjustments = {}
            
            for symbol1, pos1 in position_sizes.items():
                total_correlation = 0
                for symbol2, pos2 in position_sizes.items():
                    if symbol1 != symbol2:
                        correlation = correlation_matrix.get((symbol1, symbol2), 0)
                        total_correlation += abs(correlation)
                        
                # Calculate adjustment factor (reduce size for high correlations)
                if total_correlation > 0:
                    adj_factor = 1.0 / (1.0 + total_correlation)
                    adjustments[symbol1] = adj_factor
                else:
                    adjustments[symbol1] = 1.0
                    
            # Apply adjustments
            adjusted_sizes = {}
            for symbol, position in position_sizes.items():
                adjusted_sizes[symbol] = {
                    'size': position['size'] * adjustments[symbol],
                    'pct_of_capital': position['pct_of_capital'] * adjustments[symbol],
                    'metadata': {
                        **position['metadata'],
                        'correlation_adjustment': adjustments[symbol]
                    }
                }
                
            return adjusted_sizes
        except Exception as e:
            logger.error(f"Error adjusting for correlation: {e}")
            return position_sizes
            
    def adjust_for_market_regime(self, position_sizes, market_regime):
        """Adjust position sizes based on market regime"""
        try:
            # Define adjustment factors for different regimes
            regime_factors = {
                'bull': 1.0,    # Full size in bull market
                'neutral': 0.8,  # 80% size in neutral market
                'bear': 0.6     # 60% size in bear market
            }
            
            factor = regime_factors.get(market_regime, 0.8)  # Default to neutral
            
            # Apply regime adjustment
            adjusted_sizes = {}
            for symbol, position in position_sizes.items():
                adjusted_sizes[symbol] = {
                    'size': position['size'] * factor,
                    'pct_of_capital': position['pct_of_capital'] * factor,
                    'metadata': {
                        **position['metadata'],
                        'regime_adjustment': factor
                    }
                }
                
            return adjusted_sizes
        except Exception as e:
            logger.error(f"Error adjusting for market regime: {e}")
            return position_sizes
            
    def adjust_for_volatility_regime(self, position_sizes, volatility_regime):
        """Adjust position sizes based on volatility regime"""
        try:
            # Define adjustment factors for different volatility regimes
            vol_factors = {
                'low': 1.0,     # Full size in low volatility
                'normal': 0.8,  # 80% size in normal volatility
                'high': 0.5     # 50% size in high volatility
            }
            
            factor = vol_factors.get(volatility_regime, 0.8)  # Default to normal
            
            # Apply volatility adjustment
            adjusted_sizes = {}
            for symbol, position in position_sizes.items():
                adjusted_sizes[symbol] = {
                    'size': position['size'] * factor,
                    'pct_of_capital': position['pct_of_capital'] * factor,
                    'metadata': {
                        **position['metadata'],
                        'volatility_adjustment': factor
                    }
                }
                
            return adjusted_sizes
        except Exception as e:
            logger.error(f"Error adjusting for volatility regime: {e}")
            return position_sizes
            
    def calculate_portfolio_positions(self, signals, available_capital, correlation_matrix, market_regime, volatility_regime):
        """Calculate position sizes for entire portfolio"""
        try:
            # Initial position sizes
            position_sizes = {}
            
            for symbol, signal_data in signals.items():
                position = self.calculate_position_size(
                    signal_data['signal'],
                    signal_data['conviction'],
                    signal_data.get('volatility', 0.5),  # Default to medium volatility
                    available_capital
                )
                
                if position['size'] > 0:
                    position_sizes[symbol] = position
                    
            # Apply adjustments
            position_sizes = self.adjust_for_correlation(position_sizes, correlation_matrix)
            position_sizes = self.adjust_for_market_regime(position_sizes, market_regime)
            position_sizes = self.adjust_for_volatility_regime(position_sizes, volatility_regime)
            
            # Ensure total allocation doesn't exceed available capital
            total_allocation = sum(p['size'] for p in position_sizes.values())
            if total_allocation > available_capital:
                scale_factor = available_capital / total_allocation
                for symbol in position_sizes:
                    position_sizes[symbol]['size'] *= scale_factor
                    position_sizes[symbol]['pct_of_capital'] *= scale_factor
                    position_sizes[symbol]['metadata']['final_adjustment'] = scale_factor
                    
            return position_sizes
        except Exception as e:
            logger.error(f"Error calculating portfolio positions: {e}")
            return {} 