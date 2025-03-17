# Pattern-Based Trading Strategy

This document explains the pattern-based trading strategy implemented in the hedge fund system. The strategy uses technical price patterns and indicators to identify potential long and short trading opportunities with margin.

## Overview

The pattern strategy identifies trading opportunities by analyzing candlestick patterns, support/resistance levels, technical indicators, and trend patterns. It assigns a score to potential long and short signals based on a combination of these factors, and generates trading decisions when the score exceeds a configurable threshold.

Key features:
- Supports both long and short positions
- Uses margin trading with configurable leverage (up to 3x by default)
- Dynamic risk management with ATR-based stop loss and take profit levels
- Trailing stops to lock in profits
- Multiple pattern recognition techniques

## Implemented Patterns

The strategy detects the following patterns:

### Candlestick Patterns
- Hammer pattern (bullish reversal)
- Shooting star pattern (bearish reversal)
- Bullish engulfing pattern
- Bearish engulfing pattern
- Doji pattern (indecision)

### Support and Resistance
- Dynamic support and resistance levels
- Breakouts and breakdowns
- Price action near support/resistance

### Technical Indicators
- RSI (Relative Strength Index) overbought/oversold
- MACD (Moving Average Convergence Divergence) crossovers
- Bollinger Bands price extremes
- ATR (Average True Range) for volatility measurement

### Advanced Patterns
- Bullish and bearish divergences (price vs RSI)
- Trend strength measurement (ADX)
- Moving average crossovers

## Configuration

The strategy can be configured through the config dictionary. Here's an example configuration:

```python
config = {
    'pattern_strategy': {
        'use_margin': True,                # Enable margin trading
        'max_leverage': 3.0,               # Maximum leverage of 3x
        'confirmation_needed': 2,          # Require at least 2 confirming signals
        'pattern_score_threshold': 4,      # Minimum pattern score to generate a signal
        'stop_loss_atr_multiplier': 1.5,   # Stop loss at 1.5 * ATR
        'take_profit_atr_multiplier': 3.0, # Take profit at 3 * ATR
        'trailing_stop_activation': 0.02,  # Activate trailing stop at 2% profit
        'trailing_stop_distance': 0.015,   # 1.5% trailing stop distance
        
        # Technical indicator parameters
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'breakout_lookback': 20,
        'volatility_lookback': 14
    }
}
```

## Signal Generation

The strategy calculates separate scores for long and short signals based on the detected patterns. The scoring system assigns different weights to various patterns:

- Basic patterns (hammer, doji): 1 point
- Significant patterns (engulfing): 1.5 points
- Major patterns (breakouts, divergences): 2 points
- Technical confirmations (RSI extremes, MACD crossovers): 1-1.5 points
- Volume confirmation: 1 point

A signal is generated when the score exceeds the configured threshold (default: 4 points). The confidence of the signal is proportional to how much the score exceeds the threshold.

## Risk Management

The strategy employs several risk management techniques:

1. **Dynamic Stop Loss**: Calculated based on the ATR (Average True Range) to adapt to market volatility
2. **Dynamic Take Profit**: Also based on ATR, with a higher multiplier than stop loss
3. **Trailing Stops**: Activated when profit reaches a certain threshold, then follows the price to lock in profits
4. **Variable Leverage**: Leverage is adjusted based on signal confidence (stronger signals use higher leverage)

## Using the Strategy

To use the pattern strategy:

1. Create an instance of `PatternStrategy` with your configuration:
   ```python
   from strategies.pattern_strategy import PatternStrategy
   
   pattern_strategy = PatternStrategy(config)
   ```

2. Prepare your price data with the required indicators:
   ```python
   prepared_data = pattern_strategy.prepare_data(price_data)
   ```

3. Generate trading signals:
   ```python
   signals = pattern_strategy.get_signals(prepared_data, portfolio)
   ```

4. Update position risk parameters during trading:
   ```python
   # For each active position
   updated_position = pattern_strategy.update_position(position, current_data)
   ```

## Backtesting

A dedicated script is available to backtest the pattern strategy:

```bash
python scripts/run_pattern_backtest.py
```

This will run a backtest using the CSV files in the `backtesting` folder and output the results in the `backtest_results/pattern` directory.

## Performance Optimization

You can optimize the strategy parameters based on historical data using the `optimize_parameters` method:

```python
optimized_params = pattern_strategy.optimize_parameters(historical_data)
```

This will return optimized parameter values that you can use to reconfigure the strategy.

## Next Steps

- Explore combining the pattern strategy with the ML strategy for enhanced performance
- Optimize parameter values using grid search or genetic algorithms
- Add more advanced patterns and technical indicators
- Implement market regime detection to adapt the strategy to different market conditions 