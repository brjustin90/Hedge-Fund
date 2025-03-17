import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os
from typing import Dict, List
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.features import FeatureEngineering
from models.ml_strategy import MLStrategy
from models.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """Load configuration from config file"""
    config_path = os.path.join('config', 'backtest_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_price_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess price data from CSV"""
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'])
    
    # Extract token name from filename
    token = os.path.basename(file_path).split(',')[0]
    df['token'] = token
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'token']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {file_path}")
    
    return df

def prepare_data(data_files: List[str]) -> Dict[str, pd.DataFrame]:
    """Prepare data for backtesting"""
    feature_engineering = FeatureEngineering()
    prepared_data = {}
    
    for file_path in data_files:
        try:
            # Load raw data
            df = load_price_data(file_path)
            
            # Create features
            features_df = feature_engineering.create_features(df)
            
            # Store prepared data
            token = df['token'].iloc[0]
            prepared_data[token] = features_df
            
            logger.info(f"Prepared data for {token}: {len(features_df)} rows")
            
        except Exception as e:
            logger.error(f"Error preparing data from {file_path}: {e}")
            continue
    
    return prepared_data

def run_backtest(config: Dict, prepared_data: Dict[str, pd.DataFrame]) -> Dict:
    """Run backtest with prepared data"""
    try:
        # Initialize backtest engine
        engine = BacktestEngine(config)
        
        # Get date range from data
        all_dates = pd.concat([df['timestamp'] for df in prepared_data.values()])
        start_date = all_dates.min()
        end_date = all_dates.max()
        
        # Run backtest
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            data_collector=lambda x, y, z: prepared_data,  # Return prepared data directly
            tokens=list(prepared_data.keys())
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return {}

def analyze_results(results: Dict) -> None:
    """Analyze and print backtest results"""
    if not results:
        logger.error("No results to analyze")
        return
        
    # Print overall metrics
    logger.info("\nBacktest Results:")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    
    # Print trade analysis
    trades = results.get('trades', [])
    if trades:
        profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
        total_trades = len(trades)
        avg_hold_time = sum((t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades) / total_trades
        
        logger.info(f"\nTrade Analysis:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Profitable Trades: {profitable_trades} ({profitable_trades/total_trades:.2%})")
        logger.info(f"Average Hold Time: {avg_hold_time:.1f} hours")
        
        # Analyze exit reasons
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
        logger.info("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            logger.info(f"{reason}: {count} ({count/total_trades:.2%})")

def main():
    """Main function to run 5-minute backtest"""
    # Load configuration
    config = load_config()
    
    # Get data files
    data_dir = 'backtesting'
    data_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.csv') and not f.startswith('.')
    ]
    
    # Prepare data
    logger.info("Preparing data...")
    prepared_data = prepare_data(data_files)
    
    if not prepared_data:
        logger.error("No data prepared, exiting")
        return
    
    # Run backtest
    logger.info("Running backtest...")
    results = run_backtest(config, prepared_data)
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    output_dir = 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'backtest_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main() 