"""Execution model for backtesting"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ExecutionDetails:
    """Container for execution details"""
    def __init__(self, platform: str, execution_price: float, fee: float, slippage: float, liquidity_score: float):
        self.platform = platform
        self.execution_price = execution_price
        self.fee = fee
        self.slippage = slippage
        self.liquidity_score = liquidity_score

class ExecutionModel:
    """Model for simulating trade execution across different platforms"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Platform specific parameters
        self.platform_fees = {
            'jupiter': config.get('platforms', {}).get('jupiter', {}).get('fee_rate', 0.0035),
            'raydium': config.get('platforms', {}).get('raydium', {}).get('fee_rate', 0.003),
            'openbook': config.get('platforms', {}).get('openbook', {}).get('fee_rate', 0.002),
        }
        
        # Slippage models per platform
        self.slippage_models = {
            'jupiter': self._jupiter_slippage,
            'raydium': self._raydium_slippage,
            'openbook': self._openbook_slippage
        }
        
        # Liquidity thresholds per platform (max trade size with acceptable slippage)
        self.liquidity_thresholds = {
            'jupiter': 100000,  # $100k
            'raydium': 50000,   # $50k
            'openbook': 30000   # $30k
        }
        
    def get_best_execution(self, symbol: str, size: float, side: str, market_data: Dict) -> Optional[ExecutionDetails]:
        """Get best execution details across platforms"""
        executions = []
        
        for platform in self.platform_fees.keys():
            try:
                # Calculate price, fees, slippage
                price = float(market_data.get('close', 0))
                if price == 0:
                    continue
                    
                fee_rate = self.platform_fees[platform]
                slippage_rate = self.slippage_models[platform](size, market_data)
                
                # Adjust price based on side and slippage
                execution_price = price * (1 + (slippage_rate if side == 'buy' else -slippage_rate))
                
                # Calculate liquidity score (0 to 1)
                liquidity_score = self._calculate_liquidity_score(platform, size)
                
                # Add to executions list
                executions.append(
                    ExecutionDetails(
                        platform=platform,
                        execution_price=execution_price,
                        fee=fee_rate,
                        slippage=slippage_rate,
                        liquidity_score=liquidity_score
                    )
                )
                
            except Exception as e:
                logger.error(f"Error calculating execution for {platform}: {e}")
                
        # Sort executions by total cost (price + fees + slippage)
        if side == 'buy':
            executions.sort(key=lambda x: x.execution_price * (1 + x.fee))
        else:
            executions.sort(key=lambda x: -x.execution_price * (1 - x.fee))
            
        # Return best execution with sufficient liquidity
        for execution in executions:
            # Skip if liquidity is too low
            if execution.liquidity_score < 0.5:
                continue
                
            return execution
            
        # If no suitable execution found
        return None
        
    def _jupiter_slippage(self, size: float, market_data: Dict) -> float:
        """Calculate slippage for Jupiter"""
        # Base slippage rate
        base_slippage = 0.001  # 0.1%
        
        # Volume-based adjustment (higher volume = lower slippage)
        volume = float(market_data.get('volume', 0))
        if volume > 0:
            volume_factor = min(1, size / volume)
            return base_slippage * (1 + volume_factor)
            
        return base_slippage * 2  # Double slippage if no volume data
        
    def _raydium_slippage(self, size: float, market_data: Dict) -> float:
        """Calculate slippage for Raydium"""
        # Similar to Jupiter but with higher base slippage
        base_slippage = 0.0015  # 0.15%
        
        # Volume-based adjustment
        volume = float(market_data.get('volume', 0))
        if volume > 0:
            volume_factor = min(1, size / volume)
            return base_slippage * (1 + volume_factor)
            
        return base_slippage * 2
        
    def _openbook_slippage(self, size: float, market_data: Dict) -> float:
        """Calculate slippage for OpenBook"""
        # Higher slippage for OpenBook
        base_slippage = 0.002  # 0.2%
        
        # Volume-based adjustment
        volume = float(market_data.get('volume', 0))
        if volume > 0:
            volume_factor = min(1, size / volume)
            return base_slippage * (1 + volume_factor)
            
        return base_slippage * 2
        
    def _calculate_liquidity_score(self, platform: str, size: float) -> float:
        """Calculate liquidity score based on trade size relative to threshold"""
        threshold = self.liquidity_thresholds.get(platform, 10000)
        
        # Linear score based on size relative to threshold
        if size > threshold:
            return max(0, 1 - (size - threshold) / threshold)
            
        return 1.0  # Full liquidity for small trades 