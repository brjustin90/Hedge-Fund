"""Risk management system for backtesting"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk parameters and calculations for backtesting"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Get risk parameters from config
        risk_config = config.get('risk', {})
        self.max_drawdown = risk_config.get('max_drawdown', 0.15)
        self.daily_var_limit = risk_config.get('daily_var_limit', 0.02)
        self.position_var_limit = risk_config.get('position_var_limit', 0.01)
        
        # Portfolio tracking
        self.current_drawdown = 0.0
        self.portfolio_peak = 0.0
        self.current_var = 0.0
        self.positions = {}
        
    def calculate_position_sizes(self, signals: Dict, portfolio_value: float) -> Dict:
        """Calculate position sizes based on signals and risk parameters"""
        position_sizes = {}
        
        for symbol, signal in signals.items():
            # Skip weak signals
            if abs(signal['signal']) < 0.2:
                continue
                
            # Base position size as percentage of portfolio
            base_size = portfolio_value * 0.02  # Default 2% per position
            
            # Adjust by signal strength
            adjusted_size = base_size * abs(signal['signal'])
            
            # Adjust by conviction
            if 'conviction' in signal:
                adjusted_size *= signal['conviction']
                
            # Apply risk limits
            adjusted_size = min(
                adjusted_size,
                portfolio_value * self.position_var_limit / 0.01  # Scale by risk
            )
            
            # Determine direction
            direction = 1 if signal['signal'] > 0 else -1
            
            position_sizes[symbol] = {
                'size': adjusted_size * direction,
                'risk_score': min(1.0, abs(signal['signal'])),
                'max_drawdown': self.max_drawdown
            }
            
        return position_sizes
        
    def monitor_risk(self, portfolio_value: float, positions: Dict) -> Dict:
        """Monitor current risk levels"""
        # Update portfolio peak
        if portfolio_value > self.portfolio_peak:
            self.portfolio_peak = portfolio_value
            
        # Calculate current drawdown
        if self.portfolio_peak > 0:
            self.current_drawdown = 1 - (portfolio_value / self.portfolio_peak)
            
        # Simple VaR calculation (position-weighted)
        total_exposure = sum(abs(p.get('size', 0)) for p in positions.values())
        weighted_var = 0
        
        if total_exposure > 0:
            for symbol, position in positions.items():
                position_weight = abs(position.get('size', 0)) / total_exposure
                position_var = position.get('risk_score', 0.5) * self.position_var_limit
                weighted_var += position_weight * position_var
        
        self.current_var = weighted_var
        
        return {
            'drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'current_var': self.current_var,
            'daily_var_limit': self.daily_var_limit,
            'risk_utilization': self.current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0
        }
        
    def check_emergency_close(self) -> bool:
        """Check if emergency position closure is needed"""
        # Close all positions if drawdown exceeds max drawdown
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Emergency close triggered: Drawdown {self.current_drawdown:.2%} exceeds maximum {self.max_drawdown:.2%}")
            return True
            
        # Close all positions if VaR exceeds limit
        if self.current_var > self.daily_var_limit:
            logger.warning(f"Emergency close triggered: VaR {self.current_var:.2%} exceeds limit {self.daily_var_limit:.2%}")
            return True
            
        return False 