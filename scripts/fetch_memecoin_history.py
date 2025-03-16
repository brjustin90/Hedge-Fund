import asyncio
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Import data collector
from data.collectors.jupiter_collector import JupiterCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_memecoin_historical_data():
    """
    Fetch historical data for top 20 memecoins from Helius API and save to CSV
    """
    try:
        # Set date range: March 8th to March 14th
        start_date = datetime(2025, 3, 8)
        end_date = datetime(2025, 3, 14)

        logger.info(f"Fetching historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create output directory
        output_dir = Path("data/historical")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collector with synthetic data mode due to API issues
        collector = JupiterCollector()
        
        # Get top memecoin tokens
        logger.info("Fetching top memecoin tokens...")
        tokens = []
        try:
            tokens = await collector.get_top_tokens_by_market_cap(limit=20)
        except Exception as e:
            logger.warning(f"Error fetching tokens from API: {e}")
            
        if not tokens or len(tokens) == 0:
            logger.warning("No tokens retrieved, using fallback list")
            tokens = collector._get_fallback_memecoin_tokens()  # Use the fallback list directly
        
        logger.info(f"Retrieved {len(tokens)} memecoin tokens")
        
        # Create a master DataFrame to hold all data
        all_data = []
        
        # Track token metadata for summary
        token_metadata = {}
        
        # Fetch data for each token
        for token in tqdm(tokens, desc="Fetching token data"):
            token_data = None
            try:
                # Get token metadata for additional information
                metadata = None
                try:
                    metadata = await collector.get_token_metadata(token)
                except Exception as e:
                    logger.warning(f"Error fetching metadata for {token}: {e}")
                
                if metadata:
                    token_metadata[token] = {
                        'symbol': metadata.get('symbol', ''),
                        'name': metadata.get('name', ''),
                        'decimals': metadata.get('decimals', 9),
                        'created_at': metadata.get('created_at', '')
                    }
                else:
                    # Use placeholder metadata
                    token_short = token[:8]
                    token_metadata[token] = {
                        'symbol': token_short,
                        'name': f"Token {token_short}",
                        'decimals': 9
                    }
                
                # Get current price for reference
                current_price = None
                try:
                    current_price = await collector.get_current_price(token)
                except Exception as e:
                    logger.warning(f"Error fetching current price for {token}: {e}")
                
                if current_price:
                    if token in token_metadata:
                        token_metadata[token]['current_price'] = current_price
                
                # Get price history for the token
                logger.info(f"Fetching price history for {token}")
                try:
                    token_data = await collector.get_price_history(token, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Error fetching price history for {token}: {e}")
                    token_data = None
                
                # If no data was returned, generate synthetic data
                if token_data is None or isinstance(token_data, dict) or token_data.empty:
                    logger.warning(f"No price data returned for {token}, generating synthetic data")
                    base_price = current_price if current_price else None
                    token_data = collector._generate_synthetic_price_data(start_date, end_date, base_price=base_price, token=token)
                
                if token_data is not None and not token_data.empty:
                    # Add token info and metadata
                    if 'token' not in token_data.columns:
                        token_data['token'] = token
                    
                    if metadata:
                        token_data['symbol'] = metadata.get('symbol', '')
                        token_data['name'] = metadata.get('name', '')
                        token_data['decimals'] = metadata.get('decimals', 9)
                    else:
                        token_data['symbol'] = token_metadata[token]['symbol']
                        token_data['name'] = token_metadata[token]['name']
                        token_data['decimals'] = token_metadata[token]['decimals']
                    
                    # Add to master dataset
                    all_data.append(token_data)
                    logger.info(f"Added {len(token_data)} data points for {token}")
                    
                    # Store data points count
                    if token in token_metadata:
                        token_metadata[token]['data_points'] = len(token_data)
                else:
                    logger.warning(f"No valid data returned for {token}")
                    if token in token_metadata:
                        token_metadata[token]['data_points'] = 0
            except Exception as e:
                logger.error(f"Error processing token {token}: {e}")
                if token in token_metadata:
                    token_metadata[token]['error'] = str(e)
        
        # Check if we have any data
        if not all_data:
            logger.error("No data was collected for any token")
            # Generate at least some synthetic data for the first token
            if tokens:
                logger.info(f"Generating synthetic data for {tokens[0]}")
                synt_data = collector._generate_synthetic_price_data(start_date, end_date, token=tokens[0])
                synt_data['symbol'] = 'SYNT'
                synt_data['name'] = 'Synthetic Token'
                synt_data['decimals'] = 9
                all_data.append(synt_data)
            else:
                return
        
        # Combine all data into a single DataFrame
        logger.info("Combining data from all tokens...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Make sure timestamp is in the right format
        if 'timestamp' in combined_df.columns and not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # Sort by token and timestamp
        combined_df = combined_df.sort_values(['token', 'timestamp'])
        
        # Save to CSV
        output_file = output_dir / f"memecoin_historical_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        logger.info(f"Saving {len(combined_df)} rows to {output_file}")
        combined_df.to_csv(output_file, index=False)
        
        # Calculate statistics for each token
        token_stats = []
        for token, data in combined_df.groupby('token'):
            # Get price column
            price_col = 'price' if 'price' in data.columns else 'close'
            if price_col not in data.columns:
                continue
                
            # Get metadata
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns and not pd.isna(data['symbol'].iloc[0]) else token[:8]
            name = data['name'].iloc[0] if 'name' in data.columns and not pd.isna(data['name'].iloc[0]) else f"Token {token[:8]}"
            
            # Calculate statistics
            stats = {
                'token': token,
                'symbol': symbol,
                'name': name,
                'data_points': len(data),
                'min_price': data[price_col].min(),
                'max_price': data[price_col].max(),
                'avg_price': data[price_col].mean(),
                'std_dev': data[price_col].std(),
                'start_price': data.iloc[0][price_col] if not data.empty else None,
                'end_price': data.iloc[-1][price_col] if not data.empty else None,
                'current_price': token_metadata.get(token, {}).get('current_price', None)
            }
            
            # Calculate returns
            if stats['start_price'] and stats['end_price']:
                stats['period_return'] = (stats['end_price'] - stats['start_price']) / stats['start_price']
            
            # Calculate volatility
            if len(data) > 1:
                returns = data[price_col].pct_change().dropna()
                stats['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Add price momentum
                stats['price_momentum'] = returns.mean() * 100  # Average daily return as percentage
                
            token_stats.append(stats)
            
            # Save individual token files only for the first few tokens to avoid excessive files
            if len(token_stats) <= 5:
                token_file = output_dir / f"{symbol}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                data.to_csv(token_file, index=False)
                logger.info(f"Saved {len(data)} rows for {symbol} to {token_file}")
        
        # Create token statistics summary file
        stats_df = pd.DataFrame(token_stats)
        
        # Sort by period return (descending)
        if 'period_return' in stats_df.columns:
            stats_df = stats_df.sort_values('period_return', ascending=False)
        
        # Save statistics to CSV
        stats_file = output_dir / f"memecoin_stats_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved token statistics to {stats_file}")
        
        # Create a summary markdown file for easy viewing
        md_file = output_dir / f"memecoin_summary_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.md"
        
        with open(md_file, 'w') as f:
            f.write(f"# Memecoin Historical Data Summary\n\n")
            f.write(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n")
            f.write(f"Total tokens: {len(tokens)}\n")
            f.write(f"Tokens with data: {len(stats_df)}\n")
            f.write(f"Total data points: {len(combined_df)}\n\n")
            
            f.write("## Top 5 Performing Tokens\n\n")
            f.write("| Symbol | Name | Start Price | End Price | Return | Volatility |\n")
            f.write("|--------|------|------------|-----------|--------|------------|\n")
            
            top_tokens = stats_df.head(5)
            for _, row in top_tokens.iterrows():
                f.write(f"| {row['symbol']} | {row['name']} | {row.get('start_price', 0):.6f} | {row.get('end_price', 0):.6f} | {row.get('period_return', 0):.2%} | {row.get('volatility', 0):.2%} |\n")
            
            f.write("\n## Bottom 5 Performing Tokens\n\n")
            f.write("| Symbol | Name | Start Price | End Price | Return | Volatility |\n")
            f.write("|--------|------|------------|-----------|--------|------------|\n")
            
            bottom_tokens = stats_df.tail(5)
            for _, row in bottom_tokens.iterrows():
                f.write(f"| {row['symbol']} | {row['name']} | {row.get('start_price', 0):.6f} | {row.get('end_price', 0):.6f} | {row.get('period_return', 0):.2%} | {row.get('volatility', 0):.2%} |\n")
            
            f.write("\n## Data Sources\n\n")
            f.write("Due to API connectivity issues, synthetic data was generated for most tokens. This data is based on realistic price movements but should be used only for testing and development purposes.\n\n")
            
            f.write("\n## All Tokens\n\n")
            f.write("| Symbol | Name | Min Price | Max Price | Avg Price | Volatility | Return |\n")
            f.write("|--------|------|-----------|-----------|-----------|------------|--------|\n")
            
            for _, row in stats_df.iterrows():
                f.write(f"| {row['symbol']} | {row['name']} | {row.get('min_price', 0):.6f} | {row.get('max_price', 0):.6f} | {row.get('avg_price', 0):.6f} | {row.get('volatility', 0):.2%} | {row.get('period_return', 0):.2%} |\n")
        
        logger.info(f"Created summary markdown file at {md_file}")
        
        logger.info(f"Data collection complete. Files saved to {output_dir}")
        
        # Return statistics
        return {
            "total_tokens": len(tokens),
            "tokens_with_data": len(stats_df),
            "total_data_points": len(combined_df),
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "output_directory": str(output_dir),
            "files_created": [
                output_file.name,
                stats_file.name,
                md_file.name
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for Helius API key
    helius_api_key = os.environ.get('HELIUS_API_KEY')
    if not helius_api_key:
        logger.error("HELIUS_API_KEY not found in environment variables. Please add it to your .env file.")
        logger.error("Sign up for a free API key at https://www.helius.dev/")
        exit(1)
    else:
        logger.info(f"Using Helius API key: {helius_api_key[:6]}...{helius_api_key[-4:]}")
    
    # Run the async function
    results = asyncio.run(fetch_memecoin_historical_data())
    
    if results:
        logger.info("Summary of collected data:")
        for key, value in results.items():
            logger.info(f"{key}: {value}") 