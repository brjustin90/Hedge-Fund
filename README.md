# Hedge Fund - Solana Memecoin Trading Platform

A trading platform focused on Solana memecoins, leveraging market data for optimal trade execution.

## Features

- Real-time price monitoring of top Solana memecoins
- Volatility-based trade signals
- Advanced position management with trailing stops
- Backtesting engine with detailed performance metrics

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

## Running Backtests

To run a backtest with the current configuration:

```bash
python -m scripts.run_jupiter_backtest
```

This will:
1. Fetch the top Solana memecoins by market cap
2. Retrieve price history using the Helius API
3. Run the backtest with the configured trading parameters
4. Generate a performance report

## Trading Parameters

The trading strategy uses the following parameters:

- Position size: 5% of portfolio per trade
- Stop loss: 15% from entry
- Take profit: 30% from entry  
- Trailing stop: 10% to lock in profits
- Maximum positions: 10 at any time

## Results

Backtest results are stored in the `backtest_results/` directory. 