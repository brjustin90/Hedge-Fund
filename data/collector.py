import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Simple data collector for backtesting that holds loaded data in memory
    """
    
    def __init__(self):
        self.data = {}
        
    def register_data(self, token: str, df: pd.DataFrame) -> None:
        """
        Register data for a token
        
        Args:
            token: Token symbol or identifier
            df: DataFrame with the token's data
        """
        self.data[token] = df
        logger.info(f"Registered data for {token} with {len(df)} rows")
        
    async def get_historical_data(self, token: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Get historical data for a token within the specified date range
        
        Args:
            token: Token symbol or identifier
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical data or None if not available
        """
        if token not in self.data:
            logger.warning(f"No data registered for token {token}")
            return None
            
        # Filter for date range
        df = self.data[token]
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            logger.warning(f"No data for token {token} in date range {start_date} to {end_date}")
            return None
            
        return filtered_df
        
    async def get_current_data(self, token: str, timestamp) -> Optional[Dict]:
        """
        Get data for a specific timestamp
        
        Args:
            token: Token symbol or identifier
            timestamp: Timestamp to get data for
            
        Returns:
            Dictionary with data or None if not available
        """
        if token not in self.data:
            logger.warning(f"No data registered for token {token}")
            return None
            
        # Find the closest data point to the requested timestamp
        df = self.data[token]
        df['time_diff'] = abs(df['timestamp'] - timestamp)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        
        # Convert row to dictionary and remove the temporary time_diff column
        data_dict = closest_row.to_dict()
        data_dict.pop('time_diff', None)
        
        return data_dict 