import asyncio
import logging
import yaml
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data.collectors.historical_data_fetcher import HistoricalDataFetcher
from backtesting.backtest_engine import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Load configuration
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize data fetcher
    data_fetcher = HistoricalDataFetcher(config)
    await data_fetcher.initialize()
    
    try:
        # Define backtest parameters
        start_date = datetime.now() - timedelta(days=90)  # 90 days of historical data
        end_date = datetime.now()
        
        # Define symbols to test
        symbols = [
            'SOL/USD',    # Solana
            'RAY/USD',    # Raydium
            'SRM/USD',    # Serum
            'MNGO/USD',   # Mango Markets
            'BTC/USD',    # Bitcoin (for market regime)
        ]
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        historical_data = await data_fetcher.fetch_multiple_symbols(symbols, start_date, end_date)
        
        # Save data to file for future use
        data_fetcher.save_to_file(historical_data, 'data/historical_data.json')
        
        # Initialize backtest engine
        backtest = BacktestEngine(config)
        
        # Prepare data for backtesting
        prepared_data = backtest.prepare_data(historical_data)
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtest.run_backtest(prepared_data, start_date, end_date)
        
        # Generate and save performance report
        generate_performance_report(results)
        
        # Plot results
        plot_backtest_results(results)
        
    finally:
        await data_fetcher.close()

def generate_performance_report(results):
    """Generate detailed performance report"""
    # Create reports directory if it doesn't exist
    Path('reports').mkdir(exist_ok=True)
    
    # Basic metrics
    report = {
        'Total Return': f"{results['total_return']*100:.2f}%",
        'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{results['max_drawdown']*100:.2f}%",
        'Total Trades': results['total_trades'],
        'Final Portfolio Value': f"${results['final_portfolio_value']:.2f}"
    }
    
    # Convert trades to DataFrame for analysis
    trades_df = pd.DataFrame(results['trades'])
    if not trades_df.empty:
        trades_df['pnl'] = trades_df.apply(
            lambda x: x['size'] * x['price'] * (-1 if x['side'] == 'buy' else 1),
            axis=1
        )
        
        # Calculate trade statistics
        report.update({
            'Win Rate': f"{(trades_df['pnl'] > 0).mean()*100:.2f}%",
            'Average Win': f"${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}",
            'Average Loss': f"${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}",
            'Profit Factor': f"{abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()):.2f}",
            'Total Fees': f"${trades_df['fees'].sum():.2f}"
        })
        
        # Save trade history
        trades_df.to_csv('reports/trade_history.csv', index=False)
        
    # Save performance metrics
    with open('reports/performance_metrics.txt', 'w') as f:
        for metric, value in report.items():
            f.write(f"{metric}: {value}\n")
            
    logger.info("Performance report generated in reports/performance_metrics.txt")

def plot_backtest_results(results):
    """Plot backtest results"""
    # Create plots directory if it doesn't exist
    Path('reports/plots').mkdir(parents=True, exist_ok=True)
    
    # Convert portfolio history to DataFrame
    portfolio_df = pd.DataFrame(results['portfolio_history'])
    portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
    portfolio_df.set_index('timestamp', inplace=True)
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig('reports/plots/portfolio_value.png')
    plt.close()
    
    # Plot returns distribution
    plt.figure(figsize=(10, 6))
    returns = portfolio_df['portfolio_value'].pct_change()
    sns.histplot(returns.dropna(), kde=True)
    plt.title('Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig('reports/plots/returns_distribution.png')
    plt.close()
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    peak = portfolio_df['portfolio_value'].expanding(min_periods=1).max()
    drawdown = (portfolio_df['portfolio_value'] - peak) / peak
    plt.plot(drawdown.index, drawdown)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.savefig('reports/plots/drawdown.png')
    plt.close()
    
    # Plot trade analysis if trades exist
    trades_df = pd.DataFrame(results['trades'])
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Plot trade sizes
        plt.figure(figsize=(12, 6))
        plt.scatter(trades_df['timestamp'], trades_df['size'],
                   c=trades_df['side'].map({'buy': 'green', 'sell': 'red'}),
                   alpha=0.5)
        plt.title('Trade Sizes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Trade Size')
        plt.grid(True)
        plt.savefig('reports/plots/trade_sizes.png')
        plt.close()
        
    logger.info("Plots generated in reports/plots/")

if __name__ == "__main__":
    asyncio.run(main()) 