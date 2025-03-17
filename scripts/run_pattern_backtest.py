#!/usr/bin/env python3
"""
Script to run a backtest using the pattern-based trading strategy
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt

sys.path.append(".")

from strategies.pattern_strategy import PatternStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backtest():
    """Run pattern strategy backtest with configured parameters"""
    
    # Ensure output directory exists
    os.makedirs("backtest_results/pattern", exist_ok=True)
    
    # Configure tokens and timeframe
    config = {
        "tokens": [
            "RAYDIUM_PVSSOL_98NOCL.USD",
            "CRYPTO_PWEASEUSD",
            "RAYDIUM_YZYSOL_969WMQ.USD"
        ],
        
        # Adjust the timeframe to include more data points
        "start_date": "2025-03-01 00:00:00",
        "end_date": "2025-03-24 00:00:00",
        "timezone": "US/Eastern",
        "timeframe": "5min",
        
        # Training period (30% of the data)
        "train_test_split": 0.3,
        
        "trading": {
            "initial_capital": 1000.0,
            "base_position_size": 0.25,    # Base position size reduced to 25%
            "max_position_size": 0.40,     # Maximum position size capped at 40%
            "min_position_size": 0.10,     # Minimum position size set to 10%
            "max_positions": 2,            # Maximum 2 positions at a time
            "trading_fee": 0.001,          # 0.1% fee per trade
            "slippage": 0.001,            # 0.1% slippage
            "trailing_stop": 0.020,       # 2% trailing stop
        },
        
        "pattern_strategy": {
            "use_margin": True,
            "max_leverage": 4.0,           # Reduced from 6.0 to 4.0
            
            # Pattern detection parameters
            "breakout_lookback": 12,
            "volatility_lookback": 14,
            "rsi_period": 14,
            "rsi_overbought": 75,          # Increased from 70
            "rsi_oversold": 25,            # Decreased from 30
            
            # MACD parameters
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            
            # Bollinger Band parameters
            "bb_period": 20,
            "bb_std": 2,
            
            # Strategy settings
            "confirmation_needed": 2,       # Increased from 1 to 2
            "min_price_action_score": 1.8,  # Increased from 1.5
            "min_indicator_score": 1.8,     # Increased from 1.5
            "pattern_score_threshold": 3.0,  # Increased from 2.2
            
            # Risk management (more balanced)
            "stop_loss_atr_multiplier": 1.2,
            "take_profit_atr_multiplier": 3.0,
            "trailing_stop_activation": 0.015,
            "trailing_stop_distance": 0.010,
            
            # ML enhancement
            "use_ml": True,
            "ml_confidence_threshold": 0.65,  # Added minimum ML confidence threshold
        }
    }
    
    # Find CSV files in the backtesting directory
    csv_files = [f for f in os.listdir('backtesting') if f.endswith('.csv')]
    
    # Load data for each token
    token_data = {}
    for csv_file in csv_files:
        # Extract token name (handle files with commas in the name)
        token_parts = csv_file.split(',')
        if len(token_parts) > 1:
            token = token_parts[0]  # Take the part before the first comma
        else:
            token = os.path.splitext(csv_file)[0]
            
        if token not in config["tokens"]:
            logger.info(f"Skipping {token} as it's not in the configured tokens list")
            continue
            
        file_path = os.path.join('backtesting', csv_file)
        logger.info(f"Loading data from {file_path}")
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path, parse_dates=['time'])
            
            # Rename columns to match expected format
            df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
            }, inplace=True)
            
            # Add token column
            df['token'] = token
            
            # Add volume column if not present
            if 'volume' not in df.columns:
                # Calculate pseudo-volume based on price range and movement
                df['volume'] = (df['high'] - df['low']) * 1000000  # Simple proxy for volume
            
            # Filter data for the selected date range
            eastern = pytz.timezone(config["timezone"])
            start_date = eastern.localize(datetime.strptime(config["start_date"], "%Y-%m-%d %H:%M:%S"))
            end_date = eastern.localize(datetime.strptime(config["end_date"], "%Y-%m-%d %H:%M:%S"))
            
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df[mask]
            
            if not filtered_df.empty:
                logger.info(f"Loaded data for {token}: {len(filtered_df)} rows")
                token_data[token] = filtered_df
            else:
                logger.warning(f"No data found for {token} in the selected date range")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            continue
            
    if not token_data:
        logger.error("No data available for backtesting")
        return
        
    # Initialize the strategy
    strategy = PatternStrategy(config)
    
    # Prepare data for each token
    prepared_data = {}
    for token, df in token_data.items():
        prepared_df = strategy.prepare_data(df)
        prepared_data[token] = prepared_df
        
    # Generate signals for each token
    signals = {}
    for token, df in prepared_data.items():
        # Get the latest data point
        latest_data = df.iloc[-1:]
        
        # Generate signals
        token_signals = strategy.get_signals(df, {'cash': config['trading']['initial_capital'], 'positions': {}})
        signals[token] = token_signals.get(token, {'signal': 0, 'confidence': 0, 'reasons': []})
        
    # Print signals
    logger.info("Trading signals generated:")
    for token, signal in signals.items():
        if signal['signal'] != 0:
            direction = "LONG" if signal['signal'] > 0 else "SHORT"
            logger.info(f"{token}: {direction} signal with confidence {signal['confidence']:.2f}")
            logger.info(f"Reasons: {', '.join(signal['reasons'])}")
            logger.info(f"Leverage: {signal.get('leverage', 1.0):.1f}x")
            logger.info(f"Stop loss: {signal.get('stop_loss', 0.0):.2%}")
            logger.info(f"Take profit: {signal.get('take_profit', 0.0):.2%}")
            logger.info("---")
        else:
            logger.info(f"{token}: No signal")
            
    # Create a simple report
    report = {
        'initial_capital': config["trading"]["initial_capital"],
        'signals': {token: {'signal': s['signal'], 'confidence': s['confidence']} for token, s in signals.items()},
        'tokens_analyzed': len(token_data),
        'total_signals': sum(1 for s in signals.values() if s['signal'] != 0),
        'long_signals': sum(1 for s in signals.values() if s['signal'] > 0),
        'short_signals': sum(1 for s in signals.values() if s['signal'] < 0),
    }
    
    # Save report
    with open("backtest_results/pattern/report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Calculate potential PnL based on signals
    initial_capital = config["trading"]["initial_capital"]
    final_capital = initial_capital
    pnl = 0.0
    
    for token, signal in signals.items():
        if signal['signal'] != 0:
            # Calculate position size based on configuration and signal confidence
            base_size = config['trading'].get('base_position_size', 0.25)
            confidence_multiplier = 1.0 + (signal['confidence'] - 0.5) * 2 if signal['confidence'] > 0.5 else 0.5
            position_size = min(base_size * confidence_multiplier, config['trading'].get('max_position_size', 0.40))
            position_size = max(position_size, config['trading'].get('min_position_size', 0.10))
            
            # Calculate position value
            position_value = initial_capital * position_size
            
            # Apply leverage
            leverage = signal.get('leverage', 1.0)
            
            # Calculate potential return based on take profit target
            take_profit = signal.get('take_profit', 0.05)  # Default 5% if not specified
            
            # Direction of trade
            direction = 1 if signal['signal'] > 0 else -1
            
            # Calculate trade PnL (simplified)
            trade_pnl = position_value * leverage * take_profit * direction
            
            # Add to total PnL
            pnl += trade_pnl
    
    # Update final capital
    final_capital = initial_capital + pnl
    
    # Add PnL to report
    report['pnl'] = pnl
    report['final_capital'] = final_capital
    report['total_return'] = (final_capital / initial_capital) - 1 if initial_capital > 0 else 0
    
    # Update the report file with PnL information
    with open("backtest_results/pattern/report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Log PnL information
    logger.info(f"Potential PnL: ${pnl:.2f}")
    logger.info(f"Initial capital: ${initial_capital:.2f}")
    logger.info(f"Final capital: ${final_capital:.2f}")
    logger.info(f"Total return: {report['total_return']:.2%}")
    
    # Create a simple chart showing signal strengths
    plt.figure(figsize=(12, 8))
    
    tokens = list(signals.keys())
    signal_strengths = [s['signal'] * s['confidence'] for s in signals.values()]
    
    plt.bar(tokens, signal_strengths)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Trading Signals by Token')
    plt.ylabel('Signal Strength (Direction * Confidence)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig("backtest_results/pattern/signals.png")
    plt.close()
    
    logger.info(f"Backtest completed with {report['total_signals']} signals")
    logger.info(f"Long signals: {report['long_signals']}")
    logger.info(f"Short signals: {report['short_signals']}")
    
    return report

if __name__ == "__main__":
    run_backtest() 