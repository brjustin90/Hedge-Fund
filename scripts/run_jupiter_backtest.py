import asyncio
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
# Use real JupiterCollector instead of mock
from data.collectors.jupiter_collector import JupiterCollector
from backtesting.backtest_engine import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_jupiter_backtest(config: dict):
    """Run backtest for Jupiter trading strategies focusing on Solana memecoins"""
    try:
        # Create output directory for results
        output_dir = Path("backtest_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set date range - extended to a week for more meaningful results
        start_date = datetime(2025, 3, 14)
        end_date = datetime(2025, 3, 21)  # Extended to one week
        
        # Calculate and print estimated backtest duration
        days_to_backtest = (end_date - start_date).days
        estimated_minutes = days_to_backtest * 0.2  # Estimate based on average processing time
        
        logger.info(f"Running backtest from {start_date} to {end_date} ({days_to_backtest} days)")
        logger.info(f"Estimated duration: {estimated_minutes:.1f} minutes")
        
        # Enhance config with ML settings
        if 'backtest' not in config:
            config['backtest'] = {}
        
        # Enable ML strategy
        config['backtest']['use_ml_strategy'] = True
        
        # Add ML config if not present
        if 'ml' not in config:
            config['ml'] = {
                'model_type': 'classification',  # Use classification model (predicts up/down)
                'target_horizon': 6,             # Predict 6 hours ahead
                'prediction_weight': 0.7,        # Weight for ML predictions vs traditional indicators
                'confidence_threshold': 0.65,    # Minimum confidence for signals
                'use_stoploss_prediction': True  # Use ML for exit signals too
            }
            
        # Optimize trading parameters for more trades
        config['trading']['position_size'] = 0.05        # 5% position size (up from 2%)
        config['trading']['max_positions'] = 5           # Allow more positions (up from 3)
        config['trading']['stop_loss'] = 0.07            # Wider stop loss (7%)
        config['trading']['take_profit'] = 0.2           # Higher take profit (20%)
        config['trading']['trailing_stop'] = 0.05        # Add trailing stop (5%)
        config['trading']['vol_threshold'] = 0.03        # Lower volatility threshold for more entries
        config['trading']['entry_confirmation'] = 1      # Require only 1 confirmation
        config['trading']['position_scaling'] = True     # Enable position scaling

        # Create data collector
        logger.info("Initializing data collector...")
        data_collector = JupiterCollector()
        
        # Get tokens to backtest
        # For this example, we'll use pre-selected list to avoid API rate limits
        tokens = await data_collector.get_top_memecoin_tokens(limit=13)
        logger.info(f"Selected {len(tokens)} tokens for backtest")
        
        # Create backtest engine
        logger.info("Initializing backtest engine...")
        backtest_engine = BacktestEngine(config)
        
        # Run backtest
        logger.info("Starting backtest...")
        results = await backtest_engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            data_collector=data_collector,
            tokens=tokens
        )
        
        # Print results
        logger.info("Backtest completed!")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Win rate: {results['win_rate']:.2f}")
        
        # Save results to file
        result_file = output_dir / f"jupiter_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'config': {k: v for k, v in config.items() if k != 'api_keys'},  # Don't save API keys
                'results': results,
                'backtest_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }, f, indent=2, default=str)
            
        logger.info(f"Results saved to {result_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Jupiter backtest: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "backtest_config.yaml"
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Configure logging
    log_level = logging.INFO
    log_config = config.get('logging', {})
    if log_config.get('level') == 'DEBUG':
        log_level = logging.DEBUG
        
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("jupiter_backtest.log")
        ]
    )
    
    # Check for Helius API key
    helius_api_key = os.environ.get('HELIUS_API_KEY')
    if not helius_api_key or helius_api_key == 'your_helius_api_key_here':
        logger.warning("Helius API key not found in environment variables.")
        logger.warning("Set the HELIUS_API_KEY in your .env file for improved data quality.")
        logger.warning("Sign up for a free API key at https://www.helius.dev/")
    else:
        logger.info("Helius API key found. Will use Helius for enhanced price data.")
    
    # Run backtest
    asyncio.run(run_jupiter_backtest(config)) 