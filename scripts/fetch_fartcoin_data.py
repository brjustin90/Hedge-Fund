import asyncio
import logging
import pandas as pd
import os
import sys
import aiohttp
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FARTCOIN token address and pool address
FARTCOIN_ADDRESS = "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump"  
FARTCOIN_SOL_POOL = "Bzc9NZfMqkXR6fz1DBph7BDf9BroyEf6pnzESP7v5iiw"  # SOL pair address

# Birdeye API headers - free tier doesn't require an API key for some endpoints
BIRDEYE_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

async def fetch_birdeye_data(token_address, start_date, end_date):
    """
    Fetch historical price data from Birdeye using the token address
    """
    # Birdeye OHLCV endpoint for 5-minute candles
    url = f"https://public-api.birdeye.so/defi/price_ohlcv?address={token_address}&type=5m&time_from={int(start_date.timestamp())}&time_to={int(end_date.timestamp())}"
    logger.info(f"Requesting data from Birdeye: {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=BIRDEYE_HEADERS) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error {response.status} from Birdeye: {error_text[:200]}...")
                
                # Try alternative Birdeye endpoint for historical prices
                alt_url = f"https://public-api.birdeye.so/defi/history_price?address={token_address}&type=5m&time_from={int(start_date.timestamp())}&time_to={int(end_date.timestamp())}"
                logger.info(f"Trying alternative Birdeye endpoint: {alt_url}")
                
                async with session.get(alt_url, headers=BIRDEYE_HEADERS) as alt_response:
                    if alt_response.status != 200:
                        alt_error = await alt_response.text()
                        logger.error(f"Error {alt_response.status} from alternative Birdeye endpoint: {alt_error[:200]}...")
                        return None
                    
                    data = await alt_response.json()
                    
                    if not data or 'data' not in data or not data['data'] or 'items' not in data['data']:
                        logger.warning(f"Invalid response format from Birdeye alt endpoint for {token_address}")
                        return None
                    
                    # Process alternative endpoint data format
                    items = data['data']['items']
                    if not items:
                        logger.warning(f"No price data found in Birdeye alt endpoint for {token_address}")
                        return None
                    
                    records = []
                    for item in items:
                        timestamp = datetime.fromtimestamp(item['unixTime'], tz=timezone.utc)
                        records.append({
                            'timestamp': timestamp,
                            'open': item.get('value', item.get('price', 0)),
                            'high': item.get('value', item.get('price', 0)),
                            'low': item.get('value', item.get('price', 0)),
                            'close': item.get('value', item.get('price', 0)),
                            'volume': item.get('volume', 0),
                            'source': 'birdeye'
                        })
                    
                    df = pd.DataFrame(records)
                    logger.info(f"Retrieved {len(df)} price points from Birdeye alt endpoint for {token_address}")
                    return df
            
            # Process response from primary endpoint
            data = await response.json()
            
            if not data or 'data' not in data or not data['data'] or 'items' not in data['data']:
                logger.warning(f"Invalid response format from Birdeye for {token_address}")
                return None
            
            # Extract OHLCV data
            items = data['data']['items']
            if not items:
                logger.warning(f"No OHLCV data found in Birdeye for {token_address}")
                return None
            
            # Convert to DataFrame
            records = []
            for item in items:
                timestamp = datetime.fromtimestamp(item['unixTime'], tz=timezone.utc)
                records.append({
                    'timestamp': timestamp,
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volume'],
                    'source': 'birdeye'
                })
            
            df = pd.DataFrame(records)
            logger.info(f"Retrieved {len(df)} candles from Birdeye for {token_address}")
            return df

# Try to fetch data from multiple sources
async def fetch_multiple_sources(token_address, pool_address, start_date, end_date):
    """Try multiple data sources to get the most complete data"""
    
    # First try Birdeye
    logger.info("Trying Birdeye API...")
    data = await fetch_birdeye_data(token_address, start_date, end_date)
    
    if data is not None and not data.empty:
        return data
    
    # If no data from any source, return empty DataFrame
    logger.error("Could not retrieve data from any source")
    return pd.DataFrame()

async def fetch_fartcoin_data():
    """
    Fetch historical data for FARTCOIN for the last 30 days at 5-minute intervals
    """
    try:
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Fetching FARTCOIN historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create output directory
        output_dir = Path("data/historical/fartcoin")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try multiple data sources
        raw_data = await fetch_multiple_sources(FARTCOIN_ADDRESS, FARTCOIN_SOL_POOL, start_date, end_date)
        
        if raw_data is None or raw_data.empty:
            logger.error("No data could be retrieved from any source")
            return
        
        # Convert timezone-aware timestamps to naive for consistent processing
        if pd.api.types.is_datetime64tz_dtype(raw_data['timestamp']):
            raw_data['timestamp'] = raw_data['timestamp'].dt.tz_localize(None)
            
        # Group data by day for daily files
        raw_data['date'] = raw_data['timestamp'].dt.date
        daily_groups = raw_data.groupby('date')
        
        all_data = []
        for date, daily_data in daily_groups:
            # Make a copy without the date column for processing
            day_df = daily_data.drop('date', axis=1).copy()
            
            # Set timestamp as index for resampling
            day_df = day_df.set_index('timestamp')
            
            # Resample to 5-minute intervals if needed
            # This step is needed if we have irregular timestamps
            candles_5min = day_df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'source': 'first'
            }).reset_index()
            
            # Fill missing source values
            candles_5min['source'] = candles_5min['source'].fillna('interpolated')
            
            # Filter out rows with NaN values in price columns (rows with no trades)
            candles_5min = candles_5min.dropna(subset=['open', 'high', 'low', 'close'])
            
            if not candles_5min.empty:
                # Save daily file
                current_date = pd.to_datetime(date)
                file_path = output_dir / f"fartcoin_5min_{current_date.strftime('%Y-%m-%d')}.csv"
                candles_5min.to_csv(file_path, index=False)
                logger.info(f"Saved {len(candles_5min)} 5-minute candles to {file_path}")
                
                # Add to combined data
                all_data.append(candles_5min)
        
        # Combine all data and save to a single file
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            date_range_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
            combined_file_path = output_dir / f"fartcoin_5min_{date_range_str}.csv"
            combined_data.to_csv(combined_file_path, index=False)
            logger.info(f"Saved combined dataset with {len(combined_data)} 5-minute candles to {combined_file_path}")
            
            # Print summary
            print("\nFARTCOIN 5-Minute Candles Summary:")
            print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"Total Candles: {len(combined_data)}")
            print(f"First Timestamp: {combined_data['timestamp'].min()}")
            print(f"Last Timestamp: {combined_data['timestamp'].max()}")
            
            # Calculate price range if we have data
            if not combined_data.empty and 'low' in combined_data and 'high' in combined_data:
                print(f"Price Range: ${combined_data['low'].min():.8f} - ${combined_data['high'].max():.8f}")
            
            # Show data sources
            if 'source' in combined_data:
                sources = combined_data['source'].unique()
                print(f"Data Sources: {', '.join(sources)}")
            
            print(f"Output Location: {combined_file_path}")
        else:
            logger.error("No valid data collected for the specified period")
            
    except Exception as e:
        logger.error(f"Error fetching FARTCOIN data: {e}", exc_info=True)

# Run the async function
if __name__ == "__main__":
    asyncio.run(fetch_fartcoin_data()) 