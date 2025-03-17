# Hedge Fund - Solana Memecoin Trading Platform

A trading platform focused on Solana memecoins, leveraging market data for optimal trade execution.

## Features

- Real-time price monitoring of top Solana memecoins
- Volatility-based trade signals
- Advanced position management with trailing stops
- Backtesting engine with detailed performance metrics
- Pattern-based trading strategy with margin support
- Dynamic risk management with ATR-based stop losses

## Setup Instructions

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

1. Copy `.env.example` to `.env`
2. Update configuration values

### Obtaining a Free Helius API Key

The application uses Helius API for retrieving Solana token data. To get your free API key:

1. Visit [Helius.dev](https://www.helius.dev/) and click "Get Started"
2. Create a free account
3. After logging in, go to the Dashboard
4. Create a new API key (the free tier offers 30 RPS with 100K daily requests)
5. Copy your API key to the `.env` file:
   ```
   HELIUS_API_KEY=your_api_key_here
   ```

## Available Strategies

### ML Strategy
Machine learning-based strategy that predicts price movements based on historical patterns.

### Pattern Strategy
Technical analysis strategy that identifies trading opportunities using candlestick patterns, support/resistance levels, and technical indicators. Supports both long and short positions with configurable leverage.

Key features:
- Margin trading with up to 3x leverage
- Dynamic ATR-based stop loss and take profit levels
- Trailing stops to lock in profits
- Multiple pattern recognition techniques

For detailed documentation, see [Pattern Strategy Docs](docs/pattern_strategy.md).

## Running Backtests

To run a backtest with the ML strategy:

```bash
python -m scripts.run_jupiter_backtest
```

To run a backtest with the pattern strategy:

```bash
python -m scripts.run_pattern_backtest
```

These will:
1. Load the historical price data from CSV files in the `backtesting` directory
2. Run the backtest with the configured trading parameters
3. Generate a performance report and charts

## Trading Parameters

The trading strategy uses the following parameters:

- Position size: 5% of portfolio per trade
- Stop loss: Dynamic based on ATR (Average True Range)
- Take profit: Dynamic based on ATR
- Trailing stop: 1.5% to lock in profits
- Maximum positions: 3 at any time
- Maximum leverage: 3x (configurable)

## Results

Backtest results are stored in the `backtest_results/` directory. Performance charts and detailed metrics are available for each strategy. 