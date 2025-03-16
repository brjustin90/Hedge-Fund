import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

class PriceDataCollector:
    def __init__(self, config):
        self.config = config
        self.session = None
        self.exchanges = {}
        self.cache = {}
        self.cache_expiry = {}
        
    async def initialize(self):
        """Initialize connections"""
        self.session = aiohttp.ClientSession()
        
        # Initialize exchange connections
        self.exchanges = {
            'jupiter': None,  # Using REST API
            'raydium': None,  # Using REST API
            'binance': ccxt.binance({'enableRateLimit': True}),
            'gate': ccxt.gateio({'enableRateLimit': True})
        }
        
    async def collect_historical_data(self, token, timeframe='1h', days_back=90):
        """Collect historical price data for token"""
        try:
            # Try Jupiter first for most recent data
            jupiter_data = await self._get_jupiter_history(token, days_back)
            
            # Try DEX data
            dex_data = await self._get_dex_history(token, days_back)
            
            # Try CEX data
            cex_data = await self._get_cex_history(token, timeframe, days_back)
            
            # Combine and normalize data
            combined_data = self._combine_price_data([jupiter_data, dex_data, cex_data])
            
            return combined_data
        except Exception as e:
            logger.error(f"Error collecting historical data for {token}: {e}")
            return None
            
    async def collect_current_data(self, token):
        """Collect current price data"""
        try:
            # Check cache first
            if token in self.cache and datetime.now() < self.cache_expiry.get(token, datetime.min):
                return self.cache[token]
                
            # Get current price from multiple sources
            prices = []
            
            # Jupiter price
            jupiter_price = await self._get_jupiter_price(token)
            if jupiter_price:
                prices.append(jupiter_price)
                
            # Raydium price
            raydium_price = await self._get_raydium_price(token)
            if raydium_price:
                prices.append(raydium_price)
                
            # CEX prices
            cex_price = await self._get_cex_price(token)
            if cex_price:
                prices.append(cex_price)
                
            if not prices:
                return None
                
            # Calculate weighted average price
            avg_price = np.average(prices)
            
            # Cache the result
            self.cache[token] = avg_price
            self.cache_expiry[token] = datetime.now() + timedelta(seconds=self.config['data']['price_update_interval'])
            
            return avg_price
        except Exception as e:
            logger.error(f"Error collecting current data for {token}: {e}")
            return None
            
    async def _get_jupiter_history(self, token, days_back):
        """Get historical data from Jupiter"""
        try:
            url = f'https://price.jup.ag/v4/price?id={token}&history=1d'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('data', {}).get('prices', [])
                    return pd.DataFrame(prices, columns=['timestamp', 'price'])
        except Exception as e:
            logger.error(f"Error getting Jupiter history for {token}: {e}")
        return pd.DataFrame()
        
    async def _get_dex_history(self, token, days_back):
        """Get historical data from DEXes"""
        try:
            # Try Raydium first
            url = f'https://api.raydium.io/v2/main/price/{token}'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('data', [])
                    return pd.DataFrame(prices, columns=['timestamp', 'price'])
        except Exception as e:
            logger.error(f"Error getting DEX history for {token}: {e}")
        return pd.DataFrame()
        
    async def _get_cex_history(self, token, timeframe, days_back):
        """Get historical data from CEXes"""
        dfs = []
        
        for exchange_id, exchange in self.exchanges.items():
            if exchange_id in ['jupiter', 'raydium']:
                continue
                
            try:
                symbol = f"{token}/USDT"
                since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['price'] = df['close']
                    dfs.append(df)
            except Exception as e:
                logger.debug(f"No CEX data for {token} on {exchange_id}: {e}")
                
        if dfs:
            return pd.concat(dfs)
        return pd.DataFrame()
        
    async def _get_jupiter_price(self, token):
        """Get current price from Jupiter"""
        try:
            url = f'https://price.jup.ag/v4/price?id={token}'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('price')
        except Exception as e:
            logger.error(f"Error getting Jupiter price for {token}: {e}")
        return None
        
    async def _get_raydium_price(self, token):
        """Get current price from Raydium"""
        try:
            url = f'https://api.raydium.io/v2/main/price?token={token}'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('price')
        except Exception as e:
            logger.error(f"Error getting Raydium price for {token}: {e}")
        return None
        
    async def _get_cex_price(self, token):
        """Get current price from CEXes"""
        prices = []
        
        for exchange_id, exchange in self.exchanges.items():
            if exchange_id in ['jupiter', 'raydium']:
                continue
                
            try:
                symbol = f"{token}/USDT"
                ticker = await exchange.fetch_ticker(symbol)
                if ticker and ticker['last']:
                    prices.append(ticker['last'])
            except Exception as e:
                logger.debug(f"No CEX price for {token} on {exchange_id}: {e}")
                
        if prices:
            return np.median(prices)
        return None
        
    def _combine_price_data(self, dataframes):
        """Combine price data from multiple sources"""
        # Filter out empty dataframes
        dfs = [df for df in dataframes if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine all dataframes
        combined = pd.concat(dfs)
        
        # Convert timestamp to datetime if needed
        if combined['timestamp'].dtype == np.int64:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'], unit='ms')
            
        # Sort by timestamp
        combined = combined.sort_values('timestamp')
        
        # Remove duplicates
        combined = combined.drop_duplicates('timestamp', keep='first')
        
        # Resample to regular intervals
        combined = combined.set_index('timestamp')
        combined = combined.resample('1H').mean()
        
        # Forward fill missing values
        combined = combined.fillna(method='ffill')
        
        return combined.reset_index()
        
    async def close(self):
        """Close connections"""
        if self.session:
            await self.session.close()
            
        for exchange in self.exchanges.values():
            if exchange and hasattr(exchange, 'close'):
                await exchange.close() 