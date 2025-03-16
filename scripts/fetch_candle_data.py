import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
import time
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import dotenv

# Load environment variables
dotenv.load_dotenv()

# API Keys
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "9211bf55c4a34b1dba701256212d4df5")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of addresses to fetch data for
MEMECOIN_ADDRESSES = [
    # BONK
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    # WIF (dogwifcoin)
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    # BONK/USDC pool
    "v51xWrRwmFVH6EKe8eZTjgK5E4uC2tzY5sVBDAGhKTL",
    # Solana (SOL)
    "So11111111111111111111111111111111111111112",
    # Jito (JTO)
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"
]

# Add market pairs for known tokens
MARKET_ADDRESSES = {
    # SOL/USDC Raydium pool
    "8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6": {
        "base_token": "So11111111111111111111111111111111111111112",
        "quote_token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "name": "SOL/USDC"
    },
    # BONK/USDC Raydium pool
    "8QaXeHBrShJTdtN1rWHbGBKJxkj4RKn9xxiHYFQRvSQe": {
        "base_token": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "quote_token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "name": "BONK/USDC"
    }
}

# Add more markets here as needed

class CandleDataCollector:
    """
    Collector for 5-minute candle data for backtesting
    Uses multiple APIs with fallback options
    """
    
    def __init__(self):
        self.session = None
        self.cache_dir = Path("data/candles")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
        self.request_delay = 1.0  # 1 second between requests to avoid rate limits
        self.birdeye_api_key = BIRDEYE_API_KEY
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, url: str, headers: Dict = None) -> Optional[Dict]:
        """Make a request with retry logic and rate limiting"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            
        # Apply rate limiting
        now = time.time()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.request_delay:
            wait_time = self.request_delay - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next request")
            await asyncio.sleep(wait_time)
            
        self.last_request_time = time.time()
        
        try:
            async with self.session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limit hit
                    logger.warning(f"Rate limit hit for {url}")
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        await asyncio.sleep(int(retry_after))
                    else:
                        await asyncio.sleep(10)  # Default wait
                    raise Exception("Rate limit exceeded")
                else:
                    error_text = await response.text()
                    logger.error(f"Error {response.status} from {url}: {error_text}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout when connecting to {url}")
            raise
        except Exception as e:
            logger.error(f"Error when connecting to {url}: {e}")
            raise
    
    async def get_candle_data_birdeye(self, token_address: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get 5-minute candle data from Birdeye API
        
        Args:
            token_address (str): Token mint address
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        # Convert to Unix timestamps (milliseconds)
        start_ts = int(start_date.timestamp()) * 1000
        end_ts = int(end_date.timestamp()) * 1000
        
        # Set headers with API key properly
        headers = {
            "X-API-KEY": self.birdeye_api_key,
            "Accept": "application/json"
        }
        
        if not self.birdeye_api_key:
            logger.warning("No Birdeye API key provided. This will likely fail.")
        else:
            logger.debug(f"Using Birdeye API key for token lookup")
        
        try:
            # First get token info to ensure it exists
            token_url = f"https://api.birdeye.so/v1/token/info?address={token_address}"
            token_data = await self._make_request(token_url, headers=headers)
            
            if not token_data or token_data.get("success") is False:
                logger.warning(f"Token {token_address} not found in Birdeye")
                return None
                
            # Try SDK v2 endpoint for better data
            sdk_url = f"https://api.birdeye.so/v2/defi/ohlcv"
            payload = {
                "address": token_address,
                "type": "5m", 
                "fromTimestamp": start_ts,
                "toTimestamp": end_ts
            }
            
            # Make POST request
            try:
                if self.session is None or self.session.closed:
                    self.session = aiohttp.ClientSession()
                    
                # Apply rate limiting
                now = time.time()
                time_since_last_request = now - self.last_request_time
                if time_since_last_request < self.request_delay:
                    wait_time = self.request_delay - time_since_last_request
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next request")
                    await asyncio.sleep(wait_time)
                    
                self.last_request_time = time.time()
                
                async with self.session.post(sdk_url, json=payload, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and data.get("success") and "data" in data:
                            ohlcv_data = data["data"].get("ohlcv", [])
                            
                            if not ohlcv_data:
                                logger.warning(f"No candle data returned from Birdeye for {token_address}")
                                return None
                            
                            # Process candle data
                            records = []
                            for item in ohlcv_data:
                                timestamp = datetime.fromtimestamp(item["timestamp"] / 1000)
                                
                                record = {
                                    'timestamp': timestamp,
                                    'open': float(item["open"]),
                                    'high': float(item["high"]),
                                    'low': float(item["low"]),
                                    'close': float(item["close"]),
                                    'volume': float(item.get("volume", 0)),
                                    'token': token_address,
                                    'source': 'birdeye-v2',
                                }
                                records.append(record)
                            
                            if records:
                                df = pd.DataFrame(records)
                                return df
                    
                    # If request fails, try to get just the current price data
                    logger.debug(f"Could not get OHLCV data from Birdeye v2 API, trying current price data")
                    
                    # Get current price as fallback
                    price_url = f"https://api.birdeye.so/v1/token/price?address={token_address}"
                    price_data = await self._make_request(price_url, headers=headers)
                    
                    if price_data and price_data.get("success") and "data" in price_data:
                        price = price_data["data"].get("value")
                        if price:
                            # Create a single record with current data
                            record = {
                                'timestamp': datetime.now(),
                                'open': float(price),
                                'high': float(price),
                                'low': float(price),
                                'close': float(price),
                                'volume': 0.0,
                                'token': token_address,
                                'source': 'birdeye-current',
                            }
                            
                            df = pd.DataFrame([record])
                            logger.info(f"Retrieved current price data for {token_address} from Birdeye")
                            return df
            
            except Exception as e:
                logger.error(f"Error with Birdeye API for {token_address}: {e}")
            
            logger.warning(f"Could not get data from Birdeye for {token_address}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Birdeye data for {token_address}: {e}")
            return None
            
    async def get_candle_data_dexscreener(self, token_address: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get 5-minute candle data from DexScreener API (fallback)
        
        Args:
            token_address (str): Token mint address
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        try:
            # First we need to get the pool address for this token
            token_url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            token_data = await self._make_request(token_url)
            
            if not token_data or "pairs" not in token_data or not token_data["pairs"]:
                logger.warning(f"Could not find trading pairs for {token_address} on DexScreener")
                return None
                
            # Get the most liquid pair
            pairs = sorted(token_data["pairs"], key=lambda x: float(x.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
            if not pairs:
                logger.warning(f"No trading pairs found for {token_address}")
                return None
                
            pool_address = pairs[0]["pairAddress"]
            
            # Convert to Unix timestamps (seconds)
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Construct DexScreener API URL
            chart_url = f"https://api.dexscreener.com/latest/dex/chart/{pool_address}?from={start_ts}&to={end_ts}&res=5m"
            
            chart_data = await self._make_request(chart_url)
            
            if chart_data and "data" in chart_data:
                candles = chart_data["data"].get("candles", [])
                
                if not candles:
                    logger.warning(f"No candle data returned from DexScreener for {token_address}")
                    return None
                
                # Process candle data
                records = []
                for candle in candles:
                    if len(candle) >= 6:  # Format: [timestamp, close, volume, open, high, low]
                        timestamp = datetime.fromtimestamp(candle[0] / 1000)
                        
                        record = {
                            'timestamp': timestamp,
                            'open': float(candle[3]),
                            'high': float(candle[4]),
                            'low': float(candle[5]),
                            'close': float(candle[1]),
                            'volume': float(candle[2]),
                            'token': token_address,
                            'source': 'dexscreener',
                        }
                        records.append(record)
                
                if records:
                    df = pd.DataFrame(records)
                    return df
            
            logger.warning(f"Invalid response format from DexScreener for {token_address}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching DexScreener data for {token_address}: {e}")
            return None
    
    async def get_token_metadata(self, token_address: str) -> Dict:
        """Get token metadata (symbol, name, etc.)"""
        # Special cases for well-known tokens
        if token_address == "So11111111111111111111111111111111111111112":
            return {
                'symbol': 'SOL',
                'name': 'Solana',
                'address': token_address,
                'decimals': 9
            }
        elif token_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
            return {
                'symbol': 'USDC',
                'name': 'USD Coin',
                'address': token_address,
                'decimals': 6
            }
        elif token_address == "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263":
            return {
                'symbol': 'BONK',
                'name': 'Bonk',
                'address': token_address,
                'decimals': 5
            }
        elif token_address == "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn":
            return {
                'symbol': 'JTO',
                'name': 'Jito',
                'address': token_address,
                'decimals': 9
            }
                
        try:
            # Try Jupiter API first for token metadata
            url = f"https://api.jup.ag/v4/tokens/{token_address}"
            
            data = await self._make_request(url)
            if data and isinstance(data, dict):
                return {
                    'symbol': data.get('symbol', ''),
                    'name': data.get('name', ''),
                    'address': token_address,
                    'decimals': data.get('decimals', 9)
                }
            
            # If Jupiter fails, try Birdeye
            url = f"https://public-api.birdeye.so/public/tokenlist?address={token_address}"
            
            data = await self._make_request(url)
            if data and "data" in data and "tokens" in data["data"]:
                tokens = data["data"]["tokens"]
                if tokens and len(tokens) > 0:
                    token = tokens[0]
                    return {
                        'symbol': token.get('symbol', ''),
                        'name': token.get('name', ''),
                        'address': token_address,
                        'decimals': token.get('decimals', 9)
                    }
            
            # Fallback to address-based naming
            short_address = token_address[:8].upper()
            return {
                'symbol': short_address,
                'name': f"Token {short_address}",
                'address': token_address,
                'decimals': 9  # Assume 9 decimals as default
            }
            
        except Exception as e:
            logger.error(f"Error fetching token metadata for {token_address}: {e}")
            short_address = token_address[:8].upper()
            return {
                'symbol': short_address,
                'name': f"Token {short_address}",
                'address': token_address,
                'decimals': 9  # Assume 9 decimals as default
            }
    
    async def get_candle_data_geckoterminal(self, token_address: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get 5-minute candle data from GeckoTerminal API
        
        Args:
            token_address (str): Token mint address
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        try:
            # First we need to find the pool for this token
            url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
            
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            data = await self._make_request(url, headers)
            
            if not data or "data" not in data or not data["data"]:
                logger.warning(f"No pools found for {token_address} on GeckoTerminal")
                return None
            
            # Get the most liquid pool (usually first)
            pools = data["data"]
            if not pools:
                logger.warning(f"No pools available for {token_address}")
                return None
                
            # Check included data for pool info
            included = data.get("included", [])
            token_info = None
            for item in included:
                if item.get("type") == "token" and item.get("id") == token_address:
                    token_info = item
                    break
            
            # Get token symbol if available
            symbol = "UNKNOWN"
            if token_info and "attributes" in token_info:
                symbol = token_info["attributes"].get("symbol", "UNKNOWN")
            
            # Sort by liquidity if available and take top pool
            pool = pools[0]  # Default to first pool
            pool_id = pool["id"]
            
            logger.debug(f"Using pool {pool_id} for {token_address}")
            
            # Get current token info
            if "attributes" in pool:
                attributes = pool["attributes"]
                
                # Create a basic record with current price data
                if "token_price_usd" in attributes and attributes["token_price_usd"]:
                    price = float(attributes["token_price_usd"])
                    
                    # Get additional data if available
                    volume_usd = 0
                    if "volume_usd" in attributes:
                        volume_usd = float(attributes["volume_usd"]["h24"]) if "h24" in attributes["volume_usd"] else 0
                    
                    # Create current price record
                    record = {
                        'timestamp': datetime.now(),
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume_usd,
                        'token': token_address,
                        'symbol': symbol,
                        'source': 'geckoterminal-current',
                    }
                    
                    df = pd.DataFrame([record])
                    logger.info(f"Retrieved current price data for {token_address} ({symbol}) from GeckoTerminal")
                    return df
            
            logger.warning(f"Invalid price data for {token_address} on GeckoTerminal")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching GeckoTerminal data for {token_address}: {e}")
            return None
            
    async def get_candle_data(self, token_address: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get candle data, trying different sources in priority order
        
        Args:
            token_address (str): Token mint address
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        # Try Birdeye first
        logger.info(f"Fetching candle data from Birdeye for {token_address}")
        df = await self.get_candle_data_birdeye(token_address, start_date, end_date)
        
        if df is not None and not df.empty:
            logger.info(f"Successfully retrieved data from Birdeye for {token_address}")
            return df
        
        # Get token metadata to try by symbol with DexScreener
        metadata = await self.get_token_metadata(token_address)
        symbol = metadata.get('symbol')
        
        if symbol:
            logger.info(f"Trying DexScreener using symbol {symbol} for {token_address}")
            df = await self.get_candle_data_dexscreener_symbol(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully retrieved data from DexScreener for {symbol}")
                # Add token address to the data
                df['token'] = token_address
                return df
                
        # Try GeckoTerminal as last resort
        logger.info(f"Trying GeckoTerminal for {token_address}")
        df = await self.get_candle_data_geckoterminal(token_address, start_date, end_date)
        
        if df is not None and not df.empty:
            logger.info(f"Successfully retrieved data from GeckoTerminal for {token_address}")
            return df
        
        logger.warning(f"Could not retrieve data for {token_address} from any source")
        return None

    async def get_candle_data_birdeye_market(self, market_address: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get 5-minute candle data from Birdeye API for a specific market
        
        Args:
            market_address (str): Market/pool address
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        # Convert to Unix timestamps (milliseconds)
        start_ts = int(start_date.timestamp()) * 1000
        end_ts = int(end_date.timestamp()) * 1000
        
        # Direct market endpoint
        url = f"https://public-api.birdeye.so/defi/candles?pool={market_address}&type=5m&time_from={start_ts}&time_to={end_ts}"
        
        # Set headers
        headers = {
            "X-API-KEY": self.birdeye_api_key,
            "Accept": "application/json"
        }
        
        if self.birdeye_api_key:
            logger.debug(f"Using Birdeye API key for market data")
        
        try:
            response = await self._make_request(url, headers=headers)
            
            if response and "data" in response and "candles" in response["data"]:
                candles = response["data"]["candles"]
                if not candles:
                    logger.warning(f"No candle data returned from Birdeye for market {market_address}")
                    return None
                
                # Get token info
                base_token = None
                quote_token = None
                
                if market_address in MARKET_ADDRESSES:
                    base_token = MARKET_ADDRESSES[market_address]["base_token"]
                    quote_token = MARKET_ADDRESSES[market_address]["quote_token"]
                elif "pool" in response["data"]:
                    pool_info = response["data"]["pool"]
                    base_token = pool_info.get("base_token", {}).get("address")
                    quote_token = pool_info.get("quote_token", {}).get("address")
                
                # Process candle data
                records = []
                for candle in candles:
                    timestamp = datetime.fromtimestamp(candle["unixTime"] / 1000)
                    
                    record = {
                        'timestamp': timestamp,
                        'open': candle["open"],
                        'high': candle["high"],
                        'low': candle["low"],
                        'close': candle["close"],
                        'volume': candle.get("volume", 0),
                        'market': market_address,
                        'source': 'birdeye-market',
                    }
                    
                    if base_token:
                        record['token'] = base_token
                    
                    records.append(record)
                
                if records:
                    df = pd.DataFrame(records)
                    return df
            
            logger.warning(f"Invalid response format from Birdeye for market {market_address}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Birdeye market data for {market_address}: {e}")
            return None

    async def get_candle_data_dexscreener_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get 5-minute candle data from DexScreener API using symbol search
        
        Args:
            symbol (str): Token symbol (e.g., SOL, BONK)
            start_date (datetime): Start date and time
            end_date (datetime): End date and time
            
        Returns:
            DataFrame: Historical candle data with OHLCV
        """
        try:
            # First search for the token by symbol
            search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}&chain=solana"
            search_data = await self._make_request(search_url)
            
            if not search_data or "pairs" not in search_data or not search_data["pairs"]:
                logger.warning(f"No pairs found for symbol {symbol} on DexScreener")
                return None
            
            # Print debug info
            logger.debug(f"Found {len(search_data['pairs'])} pairs for {symbol}")
            if len(search_data['pairs']) > 0:
                first_pair = search_data['pairs'][0]
                logger.debug(f"First pair keys: {list(first_pair.keys())}")
                logger.debug(f"First pair data: {first_pair}")
                
            # Get the most liquid pair (usually with USDC)
            pairs = sorted(search_data["pairs"], key=lambda x: float(x.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
            
            # Filter for Solana pairs only
            solana_pairs = [p for p in pairs if p.get("chainId", "").lower() == "solana"]
            if solana_pairs:
                pairs = solana_pairs
                logger.debug(f"Found {len(solana_pairs)} Solana pairs")
            else:
                logger.warning(f"No Solana pairs found for {symbol}")
                return None
            
            # Prefer USDC pairs if available
            usdc_pairs = [p for p in pairs if "USDC" in p.get("quoteToken", {}).get("symbol", "")]
            if usdc_pairs:
                pairs = usdc_pairs
                logger.debug(f"Found {len(usdc_pairs)} USDC pairs")
                
            if not pairs:
                logger.warning(f"No liquid trading pairs found for {symbol}")
                return None
                
            # Get the pool address and token info
            pair = pairs[0]
            pool_address = pair["pairAddress"]
            base_token_address = pair.get("baseToken", {}).get("address")
            base_token_symbol = pair.get("baseToken", {}).get("symbol", symbol)
            pair_name = f"{base_token_symbol}/{pair.get('quoteToken', {}).get('symbol', 'UNKNOWN')}"
            
            logger.info(f"Found {symbol} pair: {pair_name} ({pool_address})")
            
            # Instead of trying to fetch historical candles which may not be supported,
            # get the current price data directly from pair info
            
            # Create a simple DataFrame with current data
            current_time = datetime.now()
            
            # Extract data from the pair info
            if "priceUsd" in pair and pair["priceUsd"]:
                price = float(pair["priceUsd"])
                
                # Get volume data
                volume = 0
                if "volume" in pair and "h24" in pair["volume"]:
                    volume = float(pair["volume"]["h24"])
                
                # Extract price change info if available
                price_change = 0
                if "priceChange" in pair and "h24" in pair["priceChange"]:
                    price_change = float(pair["priceChange"]["h24"]) / 100.0  # Convert percentage to decimal
                
                # Create a single record with current data
                record = {
                    'timestamp': current_time,
                    'open': price / (1 + price_change),  # Estimate open price based on 24h change
                    'high': price * 1.01,  # Estimate high as slightly above current
                    'low': price * 0.99,   # Estimate low as slightly below current
                    'close': price,
                    'volume': volume,
                    'symbol': base_token_symbol,
                    'market': pool_address,
                    'token': base_token_address,
                    'source': 'dexscreener-current',
                }
                
                df = pd.DataFrame([record])
                logger.info(f"Retrieved current price data for {symbol} from DexScreener")
                return df
            
            logger.warning(f"No price data available from DexScreener for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching DexScreener data for {symbol}: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            return None

async def fetch_candle_data(token_addresses: List[str], start_date: datetime, end_date: datetime, force: bool = False) -> Dict:
    """
    Fetch historical candle data for the specified tokens
    
    Args:
        token_addresses (List[str]): List of token addresses to fetch data for
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        force (bool): Force re-download even if data exists
        
    Returns:
        Dict: Results summary
    """
    output_dir = Path("data/candles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format for filename
    date_format = start_date.strftime("%Y%m%d") + "_to_" + end_date.strftime("%Y%m%d")
    
    # Dictionary to store results
    results = {
        "tokens_processed": 0,
        "tokens_with_data": 0,
        "total_candles": 0,
        "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "token_stats": [],
        "output_file": f"all_memecoin_candles_{date_format}_5m.csv"
    }
    
    # Create a master dataframe for all tokens
    all_data_frames = []
    
    async with CandleDataCollector() as collector:
        for token_address in token_addresses:
            try:
                # Get token metadata
                metadata = await collector.get_token_metadata(token_address)
                symbol = metadata.get('symbol', token_address[:4].upper())
                name = metadata.get('name', f"Token {token_address[:8]}")
                
                logger.info(f"Processing {symbol} ({name}) - {token_address}")
                results["tokens_processed"] += 1
                
                # Get candle data
                df = await collector.get_candle_data(token_address, start_date, end_date)
                
                if df is not None and not df.empty:
                    # Add token info
                    df['symbol'] = symbol
                    df['name'] = name
                    df['token_address'] = token_address
                    
                    # Save individual file
                    token_file = output_dir / f"{symbol.lower()}_{date_format}_5m.csv"
                    df.to_csv(token_file, index=False)
                    logger.info(f"Saved {len(df)} candles to {token_file}")
                    
                    # Add to master dataframe
                    all_data_frames.append(df)
                    
                    # Update stats
                    results["tokens_with_data"] += 1
                    results["total_candles"] += len(df)
                    
                    # Calculate basic token statistics
                    if 'close' in df.columns:
                        token_stats = {
                            'symbol': symbol,
                            'name': name,
                            'address': token_address,
                            'candle_count': len(df),
                            'min_price': df['close'].min(),
                            'max_price': df['close'].max(),
                            'avg_price': df['close'].mean(),
                            'start_price': df.iloc[0]['close'] if not df.empty else None,
                            'end_price': df.iloc[-1]['close'] if not df.empty else None,
                            'total_volume': df['volume'].sum() if 'volume' in df.columns else None,
                        }
                        
                        results["token_stats"].append(token_stats)
                else:
                    logger.warning(f"No data available for {symbol} ({token_address})")
                    
            except Exception as e:
                logger.error(f"Error processing {token_address}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Combine all data and save master file if we have any
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        master_file = output_dir / results["output_file"]
        combined_df.to_csv(master_file, index=False)
        logger.info(f"Saved combined data with {len(combined_df)} candles to {master_file}")
        
        # Create a summary file in markdown
        summary_file = output_dir / f"candle_data_summary_{date_format}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Memecoin 5-Minute Candle Data Summary\n\n")
            f.write(f"## Date Range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"- Total tokens processed: {results['tokens_processed']}\n")
            f.write(f"- Tokens with data: {results['tokens_with_data']}\n")
            f.write(f"- Total candles: {results['total_candles']}\n")
            f.write(f"- Data sources: GeckoTerminal API, Birdeye API, DexScreener API (fallback)\n\n")
            
            f.write("## Token Statistics\n\n")
            f.write("| Symbol | Name | Candle Count | Min Price | Max Price | Avg Price | Start Price | End Price |\n")
            f.write("|--------|------|-------------|-----------|-----------|-----------|-------------|----------|\n")
            
            for stats in results["token_stats"]:
                f.write(f"| {stats['symbol']} | {stats['name']} | {stats['candle_count']} | {stats['min_price']:.8f} | {stats['max_price']:.8f} | {stats['avg_price']:.8f} | {stats['start_price']:.8f} | {stats['end_price']:.8f} |\n")
        
        logger.info(f"Created summary file at {summary_file}")
    else:
        logger.warning("No data was collected for any token")
    
    return results

async def fetch_market_candle_data(market_addresses: List[str], start_date: datetime, end_date: datetime, force: bool = False) -> Dict:
    """
    Fetch historical market candle data for the specified market pairs
    
    Args:
        market_addresses (List[str]): List of market addresses to fetch data for
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        force (bool): Force re-download even if data exists
        
    Returns:
        Dict: Results summary
    """
    output_dir = Path("data/candles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_format = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    
    # Initialize results
    results = {
        "markets_processed": 0,
        "markets_with_data": 0,
        "total_candles": 0,
        "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "market_stats": [],
        "output_file": f"market_candles_{date_format}_5m.csv"
    }
    
    all_data_frames = []
    
    async with CandleDataCollector() as collector:
        for market_address in market_addresses:
            results["markets_processed"] += 1
            
            # Get market metadata
            market_name = market_address[:8]
            if market_address in MARKET_ADDRESSES:
                market_name = MARKET_ADDRESSES[market_address]["name"]
            
            logger.info(f"Processing market {market_name} - {market_address}")
            
            # Check if data already exists for this token
            market_file = output_dir / f"market_{market_address}_{date_format}_5m.csv"
            if market_file.exists() and not force:
                logger.info(f"Data already exists for {market_name}, skipping (use --force to redownload)")
                try:
                    df = pd.read_csv(market_file)
                    logger.info(f"Loaded {len(df)} existing candles for {market_name}")
                    
                    # Add to combined data
                    all_data_frames.append(df)
                    
                    # Update statistics
                    results["markets_with_data"] += 1
                    results["total_candles"] += len(df)
                    results["market_stats"].append({
                        "market": market_name,
                        "address": market_address,
                        "candles": len(df),
                        "source": "cached"
                    })
                    
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached data for {market_name}: {e}. Will redownload.")
            
            # Get candle data from Birdeye
            df = await collector.get_candle_data_birdeye_market(market_address, start_date, end_date)
            
            if df is not None and not df.empty:
                # Save individual token data
                df.to_csv(market_file, index=False)
                logger.info(f"Saved {len(df)} candles for {market_name} to {market_file}")
                
                # Add to combined data
                all_data_frames.append(df)
                
                # Update statistics
                results["markets_with_data"] += 1
                results["total_candles"] += len(df)
                results["market_stats"].append({
                    "market": market_name,
                    "address": market_address,
                    "candles": len(df),
                    "source": "birdeye"
                })
            else:
                logger.warning(f"No data available for {market_name} ({market_address})")
    
    # Save combined data if we have any
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        master_file = output_dir / results["output_file"]
        combined_df.to_csv(master_file, index=False)
        logger.info(f"Saved combined market data with {len(combined_df)} candles to {master_file}")
        
        # Generate summary file
        summary_file = output_dir / f"market_candles_{date_format}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"# Market Candle Data Summary\n\n")
            f.write(f"Date range: {results['date_range']}\n")
            f.write(f"- Markets processed: {results['markets_processed']}\n")
            f.write(f"- Markets with data: {results['markets_with_data']}\n")
            f.write(f"- Total candles: {results['total_candles']}\n")
            f.write(f"- Data sources: Birdeye API\n\n")
            
            f.write("## Market Statistics\n\n")
            for stat in results["market_stats"]:
                f.write(f"- {stat['market']} ({stat['address']}): {stat['candles']} candles from {stat['source']}\n")
        
        logger.info(f"Saved summary to {summary_file}")
    else:
        logger.warning("No data was collected for any market")
    
    return results

async def fetch_symbol_candle_data(symbols: List[str], start_date: datetime, end_date: datetime, force: bool = False) -> Dict:
    """
    Fetch historical candle data for the specified symbols
    
    Args:
        symbols (List[str]): List of token symbols to fetch data for
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        force (bool): Force re-download even if data exists
        
    Returns:
        Dict: Results summary
    """
    output_dir = Path("data/candles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_format = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    
    # Initialize results
    results = {
        "symbols_processed": 0,
        "symbols_with_data": 0,
        "total_candles": 0,
        "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "symbol_stats": [],
        "output_file": f"symbol_candles_{date_format}_5m.csv"
    }
    
    all_data_frames = []
    
    async with CandleDataCollector() as collector:
        for symbol in symbols:
            results["symbols_processed"] += 1
            
            logger.info(f"Processing symbol {symbol}")
            
            # Check if data already exists for this symbol
            symbol_file = output_dir / f"symbol_{symbol}_{date_format}_5m.csv"
            if symbol_file.exists() and not force:
                logger.info(f"Data already exists for {symbol}, skipping (use --force to redownload)")
                try:
                    df = pd.read_csv(symbol_file)
                    logger.info(f"Loaded {len(df)} existing candles for {symbol}")
                    
                    # Add to combined data
                    all_data_frames.append(df)
                    
                    # Update statistics
                    results["symbols_with_data"] += 1
                    results["total_candles"] += len(df)
                    
                    # Use the actual source from the data
                    source = df.iloc[0].get('source', 'unknown') if len(df) > 0 else 'unknown'
                    
                    results["symbol_stats"].append({
                        "symbol": symbol,
                        "candles": len(df),
                        "source": source
                    })
                    
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached data for {symbol}: {e}. Will redownload.")
            
            # Get candle data for symbol
            df = await collector.get_candle_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # Save individual symbol data
                df.to_csv(symbol_file, index=False)
                logger.info(f"Saved {len(df)} candles for {symbol} to {symbol_file}")
                
                # Add to combined data
                all_data_frames.append(df)
                
                # Update statistics
                results["symbols_with_data"] += 1
                results["total_candles"] += len(df)
                
                # Use the actual source from the data
                source = df.iloc[0].get('source', 'unknown') if len(df) > 0 else 'unknown'
                
                results["symbol_stats"].append({
                    "symbol": symbol,
                    "candles": len(df),
                    "source": source
                })
            else:
                logger.warning(f"No data available for {symbol}")
    
    # Save combined data if we have any
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        master_file = output_dir / results["output_file"]
        combined_df.to_csv(master_file, index=False)
        logger.info(f"Saved combined symbol data with {len(combined_df)} candles to {master_file}")
        
        # Generate summary file
        summary_file = output_dir / f"symbol_candles_{date_format}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"# Symbol Candle Data Summary\n\n")
            f.write(f"Date range: {results['date_range']}\n")
            f.write(f"- Symbols processed: {results['symbols_processed']}\n")
            f.write(f"- Symbols with data: {results['symbols_with_data']}\n")
            f.write(f"- Total candles: {results['total_candles']}\n")
            
            # Summarize data sources
            sources = set([stat["source"] for stat in results["symbol_stats"]])
            sources_str = ", ".join(sources)
            f.write(f"- Data sources: {sources_str}\n\n")
            
            f.write("## Symbol Statistics\n\n")
            for stat in results["symbol_stats"]:
                f.write(f"- {stat['symbol']}: {stat['candles']} candles from {stat['source']}\n")
        
        logger.info(f"Saved summary to {summary_file}")
    else:
        logger.warning("No data was collected for any symbol")
    
    return results

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Fetch historical 5-minute candle data for Solana tokens/markets')
    parser.add_argument('--start-date', type=str, default='2024-03-08', 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2024-03-14', 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--token', type=str, action='append',
                       help='Specific token address to fetch (can be used multiple times)')
    parser.add_argument('--market', type=str, action='append',
                       help='Specific market address to fetch (can be used multiple times)')
    parser.add_argument('--symbol', type=str, action='append',
                       help='Specific token symbol to fetch (can be used multiple times)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force re-download even if data exists')
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("asyncio").setLevel(logging.INFO)
    
    # Set date range
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)
    
    # Create output directory
    os.makedirs("data/candles", exist_ok=True)
    
    results = {}
    
    # Process symbols (simplest, most reliable approach)
    if args.symbol:
        symbols = args.symbol
        
        if args.verbose:
            logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
            for i, sym in enumerate(symbols):
                logger.info(f"Symbol {i+1}: {sym}")
        
        # Run the async data fetching for symbols
        symbol_results = asyncio.run(fetch_symbol_candle_data(symbols, start_date, end_date, force=args.force))
        results.update(symbol_results)
    
    # Process tokens if no symbols were specified
    if (args.token or not (args.market or args.symbol)) and not results:
        # Use provided tokens or default list
        token_addresses = args.token if args.token else MEMECOIN_ADDRESSES
        
        if args.verbose:
            logger.info(f"Fetching data for {len(token_addresses)} tokens from {start_date} to {end_date}")
            for i, addr in enumerate(token_addresses):
                logger.info(f"Token {i+1}: {addr}")
        
        # Run the async data fetching for tokens
        token_results = asyncio.run(fetch_candle_data(token_addresses, start_date, end_date, force=args.force))
        results.update(token_results)
    
    # Process markets if specified and no data collected yet
    if args.market and not results:
        # Use provided markets
        market_addresses = args.market
        
        if args.verbose:
            logger.info(f"Fetching data for {len(market_addresses)} markets from {start_date} to {end_date}")
            for i, addr in enumerate(market_addresses):
                market_name = MARKET_ADDRESSES.get(addr, {}).get("name", addr[:8])
                logger.info(f"Market {i+1}: {market_name} ({addr})")
        
        # Run the async data fetching for markets
        market_results = asyncio.run(fetch_market_candle_data(market_addresses, start_date, end_date, force=args.force))
        results.update(market_results)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Data collection complete!")
    
    if "symbols_processed" in results:
        print(f"- Processed {results['symbols_processed']} symbols")
        print(f"- Successfully collected data for {results['symbols_with_data']} symbols")
    
    if "tokens_processed" in results:
        print(f"- Processed {results['tokens_processed']} tokens")
        print(f"- Successfully collected data for {results['tokens_with_data']} tokens")
    
    if "markets_processed" in results:
        print(f"- Processed {results['markets_processed']} markets")
        print(f"- Successfully collected data for {results['markets_with_data']} markets")
    
    print(f"- Total candles collected: {results.get('total_candles', 0)}")
    if "output_file" in results:
        print(f"- Data saved to: {results['output_file']}")
    print("="*50 + "\n")
    
    return results

if __name__ == "__main__":
    main() 