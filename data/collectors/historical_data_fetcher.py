import logging
import asyncio
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    def __init__(self, config):
        self.config = config
        self.session = None
        self.exchanges = {}
        self.cache = {}
        
    async def initialize(self):
        """Initialize connections"""
        self.session = aiohttp.ClientSession()
        
        # Initialize exchange connections
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'gate': ccxt.gateio({'enableRateLimit': True}),
            'huobi': ccxt.huobi({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True})
        }
        
    async def fetch_historical_data(self, symbol, start_date, end_date=None, timeframe='1h'):
        """Fetch historical data from multiple sources"""
        if end_date is None:
            end_date = datetime.now()
            
        try:
            # Try Jupiter first
            jupiter_data = await self._fetch_jupiter_history(symbol, start_date, end_date)
            
            # Try Raydium
            raydium_data = await self._fetch_raydium_history(symbol, start_date, end_date)
            
            # Try CEX data
            cex_data = await self._fetch_cex_history(symbol, start_date, end_date, timeframe)
            
            # Combine all data sources
            combined_data = self._combine_historical_data([jupiter_data, raydium_data, cex_data])
            
            if combined_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
                
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
            
    async def _fetch_jupiter_history(self, symbol, start_date, end_date):
        """Fetch historical data from Jupiter"""
        try:
            url = f'https://price.jup.ag/v4/price?id={symbol}&history=1d'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('data', {}).get('prices', [])
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    return df
        except Exception as e:
            logger.error(f"Error fetching Jupiter history: {e}")
        return pd.DataFrame()
        
    async def _fetch_raydium_history(self, symbol, start_date, end_date):
        """Fetch historical data from Raydium"""
        try:
            url = f'https://api.raydium.io/v2/main/price/{symbol}/history'
            params = {
                'from': int(start_date.timestamp() * 1000),
                'to': int(end_date.timestamp() * 1000),
                'resolution': '60'  # 1 hour intervals
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('data', [])
                    df = pd.DataFrame(prices)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                    return df[['timestamp', 'price', 'volume']]
        except Exception as e:
            logger.error(f"Error fetching Raydium history: {e}")
        return pd.DataFrame()
        
    async def _fetch_cex_history(self, symbol, start_date, end_date, timeframe):
        """Fetch historical data from CEXes"""
        dfs = []
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Format symbol for CEX (assuming USDT pair)
                cex_symbol = f"{symbol}/USDT"
                since = int(start_date.timestamp() * 1000)
                
                # Fetch OHLCV data
                all_candles = []
                while since < end_date.timestamp() * 1000:
                    candles = await exchange.fetch_ohlcv(cex_symbol, timeframe, since)
                    if not candles:
                        break
                        
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1
                    await asyncio.sleep(exchange.rateLimit / 1000)  # Respect rate limits
                    
                if all_candles:
                    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['source'] = exchange_id
                    dfs.append(df)
                    
            except Exception as e:
                logger.debug(f"Error fetching {exchange_id} history: {e}")
                
        if dfs:
            return pd.concat(dfs)
        return pd.DataFrame()
        
    def _combine_historical_data(self, dataframes):
        """Combine historical data from multiple sources"""
        # Filter out empty dataframes
        dfs = [df for df in dataframes if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine all dataframes
        combined = pd.concat(dfs)
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp')
        
        # Remove duplicates
        combined = combined.drop_duplicates('timestamp', keep='first')
        
        # Resample to regular intervals and forward fill missing values
        combined = combined.set_index('timestamp')
        combined = combined.resample('1H').mean()
        combined = combined.fillna(method='ffill')
        
        return combined.reset_index()
        
    async def fetch_multiple_symbols(self, symbols, start_date, end_date=None, timeframe='1h'):
        """Fetch historical data for multiple symbols"""
        tasks = []
        for symbol in symbols:
            task = self.fetch_historical_data(symbol, start_date, end_date, timeframe)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        return {symbol: data for symbol, data in zip(symbols, results) if data is not None}
        
    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
            
        for exchange in self.exchanges.values():
            if exchange:
                await exchange.close()
                
    def save_to_file(self, data, filename):
        """Save historical data to file"""
        try:
            # Convert DataFrames to dictionary format
            save_data = {}
            for symbol, df in data.items():
                save_data[symbol] = df.to_dict(orient='records')
                
            with open(filename, 'w') as f:
                json.dump(save_data, f)
                
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            
    def load_from_file(self, filename):
        """Load historical data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Convert dictionary back to DataFrames
            loaded_data = {}
            for symbol, records in data.items():
                df = pd.DataFrame.from_records(records)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                loaded_data[symbol] = df
                
            return loaded_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return {} 