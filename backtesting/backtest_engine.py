import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
import asyncio
import os

# Only import what we actually need
from risk.manager import RiskManager
from strategies.ml_strategy import MLStrategy
from models.features import FeatureEngineering

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: Dict):
        self.config = config
        # Get trading parameters from config
        trading_config = config.get('trading', {})
        # Check both backtest and trading config for initial_capital, prioritizing backtest
        backtest_config = config.get('backtest', {})
        self.initial_capital = backtest_config.get('initial_capital', trading_config.get('initial_capital', 10000))
        self.position_size_pct = trading_config.get('position_size', 0.02)
        self.max_positions = trading_config.get('max_positions', 3)
        self.stop_loss_pct = trading_config.get('stop_loss', 0.05)
        self.take_profit_pct = trading_config.get('take_profit', 0.15)
        
        # Advanced parameters for memecoin trading
        self.trailing_stop_pct = trading_config.get('trailing_stop', None)  # Trailing stop as % of price
        self.max_hold_time = trading_config.get('max_hold_time', None)      # Maximum hold time in hours
        self.min_hold_time = trading_config.get('min_hold_time', None)      # Minimum hold time in hours
        self.vol_lookback = trading_config.get('vol_lookback', 24)          # Volatility lookback period in hours
        self.vol_threshold = trading_config.get('vol_threshold', 0.05)      # Volatility threshold for entry
        self.entry_confirmation = trading_config.get('entry_confirmation', 1) # Number of confirmations needed
        self.position_scaling = trading_config.get('position_scaling', False) # Whether to scale positions
        
        # ML strategy
        self.use_ml_strategy = backtest_config.get('use_ml_strategy', True)
        if self.use_ml_strategy:
            self.ml_strategy = MLStrategy(config)
        
        # Get platform fee from config
        platform_config = config.get('platforms', {}).get('jupiter', {})
        self.base_fee = platform_config.get('fee_rate', 0.0035)
        
        # Feature engineering for ML model training
        self.feature_engineering = FeatureEngineering(config)
        
        # Performance tracking
        self.portfolio_value = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.signal_history = {}  # Track signals for entry confirmation
        
        # Create directories for model saving
        os.makedirs('models/saved', exist_ok=True)
        
    def run_backtest(self, token_data: Dict[str, pd.DataFrame], train_end: datetime, test_end: datetime) -> Dict:
        """
        Run a backtest using the provided data and strategy
        
        Args:
            token_data: Dictionary mapping token names to their price data
            train_end: End date for training period (strategy will be trained on data up to this date)
            test_end: End date for testing period
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest")
        
        # Combine data from all tokens
            all_data = []
        for token, df in token_data.items():
            # Add token column if not present
            if 'token' not in df.columns:
                df = df.copy()
                df['token'] = token
            all_data.append(df)
                        
            if not all_data:
            logger.error("No data provided for backtest")
            return {'portfolio_history': []}
                
        # Combine data and sort by timestamp
        combined_data = pd.concat(all_data)
            
            # Check if timestamp is in the index or as a column
        if 'timestamp' not in combined_data.columns and isinstance(combined_data.index, pd.DatetimeIndex):
            # If we have a DatetimeIndex, reset it to make timestamp a column
            combined_data = combined_data.reset_index()
            combined_data.rename(columns={'index': 'timestamp'}, inplace=True)
                
            # Sort by timestamp
        combined_data = combined_data.sort_values('timestamp')
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Timespan: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
        logger.info(f"Unique timestamps: {combined_data['timestamp'].nunique()}")
            
            # Split data into training and testing periods
        train_data = combined_data[combined_data['timestamp'] <= train_end]
        test_data = combined_data[(combined_data['timestamp'] > train_end) & (combined_data['timestamp'] <= test_end)]
        
        if train_data.empty or test_data.empty:
            logger.error("Insufficient data for training or testing")
            return {'portfolio_history': []}
            
        logger.info(f"Training data: {len(train_data)} rows from {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
        logger.info(f"Testing data: {len(test_data)} rows from {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'timestamp': test_data['timestamp'].min(),
            'portfolio_value': self.initial_capital,
            'equity': self.initial_capital,
        }
        
        # Get the strategy from config
        strategy = self.config.get('strategy')
        if not strategy:
            logger.error("No strategy provided in config")
            return {'portfolio_history': []}
        
        # Prepare data using strategy
        prepared_train_data = strategy.prepare_data(train_data)
        
        # Record portfolio history
        portfolio_history = []
        
        # Group test data by timestamp for iteration
        timestamps = test_data['timestamp'].unique()
        timestamps = sorted(timestamps)
        
        # Process each timestamp in the test period
        for i, timestamp in enumerate(timestamps):
            # Get data for current timestamp
            current_data = test_data[test_data['timestamp'] == timestamp].copy()
            
            # Prepare data for this timestamp
            # We need to include some past data for proper indicator calculation
            lookback_data = combined_data[
                (combined_data['timestamp'] <= timestamp) & 
                (combined_data['timestamp'] > timestamp - pd.Timedelta(days=30))
            ]
            prepared_data = strategy.prepare_data(lookback_data)
            current_prepared = prepared_data[prepared_data['timestamp'] == timestamp]
            
            # Update portfolio based on market data
            self._update_portfolio_values(portfolio, current_data)
            
            # Update positions for trailing stops
            self._check_stop_conditions(portfolio, current_data)
            
            # Get signals if we have capacity for new positions
            if len(portfolio['positions']) < self.max_positions:
                signals = strategy.get_signals(prepared_data, portfolio)
                
                # Execute signals if we have available capital
                if portfolio['cash'] > 0:
                    self._execute_signals(portfolio, signals, current_data, timestamp)
                            
                    # Record portfolio state
            snapshot = self._create_portfolio_snapshot(portfolio, timestamp)
            portfolio_history.append(snapshot)
            
            # Log progress periodically
            if i % 100 == 0 or i == len(timestamps) - 1:
                logger.info(f"Processed {i+1}/{len(timestamps)} timestamps - Portfolio value: ${portfolio['portfolio_value']:.2f}")
        
        # Calculate performance metrics
        results = {
            'portfolio_history': portfolio_history,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio['portfolio_value'],
            'total_return': (portfolio['portfolio_value'] / self.initial_capital) - 1,
        }
        
        logger.info(f"Backtest completed - Final portfolio value: ${results['final_portfolio_value']:.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
                
            return results
            
    async def _collect_token_data(self, token: str, start_date: datetime, end_date: datetime, data_collector) -> Optional[pd.DataFrame]:
        """Collect and process data for a single token"""
        try:
            # Get historical data using the data_collector's get_historical_data method
            price_data = await data_collector.get_historical_data(token, start_date, end_date)
            
            # Check if data is already a DataFrame
            if isinstance(price_data, pd.DataFrame):
                if price_data.empty:
                    logger.warning(f"No price data found for {token}")
                    return None
                # Make sure we have the token column
            if 'token' not in price_data.columns:
                price_data['token'] = token
                
                # Add liquidity score if not present
            if 'liquidity_score' not in price_data.columns:
                    price_data['liquidity_score'] = 0.7  # Default liquidity score
            
            return price_data
            else:
                logger.warning(f"Unexpected data type for {token}: {type(price_data)}")
                return None
            
        except Exception as e:
            logger.error(f"Error collecting data for {token}: {e}")
            return None
            
    def _update_positions(self, portfolio: Dict, current_data: pd.DataFrame):
        """Update position values and check stop-loss/take-profit/trailing-stop/time-based exits"""
        positions_to_close = []
        
        current_timestamp = current_data['timestamp'].iloc[0] if not current_data.empty else None
        if current_timestamp is None:
            return
        
        for token, position in portfolio['positions'].items():
            token_data = current_data[current_data['token'] == token]
            if token_data.empty:
                continue
                
            # Get current price - check if 'close' exists, otherwise use 'price'
            if 'close' in token_data.columns:
                current_price = token_data['close'].iloc[0]
            elif 'price' in token_data.columns:
                current_price = token_data['price'].iloc[0]
            else:
                logger.warning(f"No price data found for {token}, skipping position update")
                continue
                
            # Calculate current value and unrealized return
            position['current_price'] = current_price
            position['current_value'] = position['quantity'] * current_price
            unrealized_return = (current_price - position['entry_price']) / position['entry_price']
            
            # Update highest price seen if we have a trailing stop
            if self.trailing_stop_pct is not None:
                if 'highest_price' not in position:
                    position['highest_price'] = position['entry_price']
                else:
                    position['highest_price'] = max(position['highest_price'], current_price)
                
                # Calculate trailing stop price
                trailing_stop_price = position['highest_price'] * (1 - self.trailing_stop_pct)
                
                # Check if price has dropped below trailing stop
                if current_price <= trailing_stop_price and current_price > position['entry_price']:
                    logger.info(f"TRAILING STOP triggered for {token} at {current_price:.6f} (highest: {position['highest_price']:.6f}, trail: {trailing_stop_price:.6f})")
                    positions_to_close.append((token, 'trailing_stop'))
                    continue
            
            # Check time-based exits
            if self.max_hold_time is not None:
                hold_time = (current_timestamp - position['entry_time']).total_seconds() / 3600  # in hours
                if hold_time >= self.max_hold_time:
                    logger.info(f"MAX HOLD TIME reached for {token} after {hold_time:.1f} hours")
                    positions_to_close.append((token, 'max_hold_time'))
                    continue
            
            # Check standard stop-loss and take-profit
            if unrealized_return <= -self.stop_loss_pct:
                positions_to_close.append((token, 'stop_loss'))
                continue
            elif unrealized_return >= self.take_profit_pct:
                positions_to_close.append((token, 'take_profit'))
                continue
            
        # Close positions that hit exit conditions
        for token, exit_reason in positions_to_close:
            token_data = current_data[current_data['token'] == token]
            if not token_data.empty:
                current_price = token_data['close'].iloc[0]
                logger.info(f"CLOSING POSITION for {token} at {current_price:.6f} due to {exit_reason}")
                self._execute_trade(portfolio, token, 'sell', current_price, token_data['timestamp'].iloc[0], 1.0, exit_reason)
            
    def _execute_trade(self, portfolio: Dict, token: str, action: str, price: float, timestamp: datetime, confidence: float, exit_reason: str = None):
        """Execute a trade and update portfolio with position scaling support"""
        try:
            if action == 'buy':
                position_value = portfolio['cash'] * self.position_size_pct * confidence
                
                # Adjust for position scaling if enabled
                if self.position_scaling and token in portfolio['positions']:
                    # This is a scale-in, use half the normal position size
                    position_value = position_value * 0.5
                    logger.info(f"Scaling into existing position for {token} with {position_value:.2f}")
                    
                quantity = position_value / price
                fee = position_value * self.base_fee
                slippage = position_value * 0.001  # Assume 0.1% slippage
                
                if portfolio['cash'] >= (position_value + fee + slippage):
                    # Handle initial position or position scaling
                    if token not in portfolio['positions']:
                        # New position
                        portfolio['positions'][token] = {
                            'quantity': quantity,
                            'entry_price': price,
                            'entry_time': timestamp,
                            'current_value': position_value,
                            'fees_paid': fee,
                            'scale_count': 1  # Track scaling count
                        }
                    else:
                        # Scale into existing position
                        existing_position = portfolio['positions'][token]
                        # Calculate new average entry price
                        total_quantity = existing_position['quantity'] + quantity
                        total_cost = (existing_position['quantity'] * existing_position['entry_price']) + (quantity * price)
                        avg_price = total_cost / total_quantity if total_quantity > 0 else price
                        
                        # Update position
                        existing_position['quantity'] += quantity
                        existing_position['entry_price'] = avg_price
                        existing_position['current_value'] += position_value
                        existing_position['fees_paid'] += fee
                        existing_position['scale_count'] += 1
                    
                    portfolio['cash'] -= (position_value + fee + slippage)
                    
                    # Record trade
                    self.trades.append({
                        'timestamp': timestamp,
                        'token': token,
                        'action': action,
                        'price': price,
                        'size': position_value,
                        'quantity': quantity,
                        'fee': fee,
                        'slippage': slippage,
                        'platform': 'jupiter',
                        'liquidity_score': confidence,
                        'exit_reason': None,
                        'is_scaling': token in portfolio['positions'] and portfolio['positions'][token]['scale_count'] > 1
                    })
                    
            elif action == 'sell' and token in portfolio['positions']:
                position = portfolio['positions'][token]
                
                # Determine sell quantity based on position scaling
                if self.position_scaling and exit_reason not in ['stop_loss', 'max_hold_time']: 
                    # For partial exits, sell half unless it's a stop loss or max hold time
                    sell_quantity = position['quantity'] * 0.5
                    logger.info(f"Scaling out of position for {token}, selling {sell_quantity} units (50%)")
                    
                    # If take_profit triggered, use the scaling approach
                    if exit_reason == 'take_profit':
                        # If this is the first scale out, update the position
                        if position.get('scale_out_count', 0) == 0:
                            position['scale_out_count'] = 1
                        else:
                            # On subsequent scale outs on take profit, sell remaining
                            sell_quantity = position['quantity']
                else:
                    # Full exit for stop loss or if scaling is disabled
                    sell_quantity = position['quantity']
                
                # Calculate values
                position_value = sell_quantity * price
                fee = position_value * self.base_fee
                slippage = position_value * 0.001  # Assume 0.1% slippage
                
                portfolio['cash'] += (position_value - fee - slippage)
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'token': token,
                    'action': action,
                    'price': price,
                    'size': position_value,
                    'quantity': sell_quantity,
                    'fee': fee,
                    'slippage': slippage,
                    'platform': 'jupiter',
                    'liquidity_score': confidence,
                    'exit_reason': exit_reason,
                    'is_scaling': sell_quantity < position['quantity']
                })
                
                # Update or remove position
                if sell_quantity < position['quantity'] and self.position_scaling:
                    # Partial sell - update the position
                    position['quantity'] -= sell_quantity
                    position['current_value'] = position['quantity'] * price
                    position['fees_paid'] += fee
                    position['highest_price'] = price  # Reset highest price after partial take profit
                else:
                    # Complete sell - remove the position
                    del portfolio['positions'][token]
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise
            
    def _calculate_portfolio_value(self, portfolio: Dict, current_data: pd.DataFrame) -> float:
        """Calculate total portfolio value"""
        total_value = portfolio['cash']
        
        for token, position in portfolio['positions'].items():
            token_data = current_data[current_data['token'] == token]
            if not token_data.empty:
                # Get current price - check if 'close' exists, otherwise use 'price'
                if 'close' in token_data.columns:
                    current_price = token_data['close'].iloc[0]
                elif 'price' in token_data.columns:
                    current_price = token_data['price'].iloc[0]
                else:
                    logger.warning(f"No price data found for {token}, using last known price")
                    current_price = position.get('current_price', position['entry_price'])
                
                position_value = position['quantity'] * current_price
                total_value += position_value
                
        return total_value
        
    def generate_report(self, portfolio_history):
        """Generate backtest performance report"""
        try:
            # Convert list of dictionaries to DataFrame for easier analysis
            results = pd.DataFrame(portfolio_history)
            
            # Calculate returns
            results['returns'] = results['portfolio_value'].pct_change().fillna(0)
            results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
            
            # Calculate basic metrics
            initial_capital = results['portfolio_value'].iloc[0]
            final_capital = results['portfolio_value'].iloc[-1]
            total_return = final_capital / initial_capital - 1
            
            total_days = (results['timestamp'].max() - results['timestamp'].min()).total_seconds() / 86400  # Convert to days
            annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
            
            # Risk metrics
            daily_returns = results['returns'].fillna(0)
            volatility = daily_returns.std() * np.sqrt(365)
            sharpe_ratio = (daily_returns.mean() * 365) / volatility if volatility != 0 else 0
            max_drawdown = (results['portfolio_value'] / results['portfolio_value'].cummax() - 1).min()
            
            # Trade statistics
            if 'trades' in results.columns:
                trades = results['trades'].explode().dropna().tolist()
            else:
                trades = []
                
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = -sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades) if losing_trades else float('inf')
            
            report = {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'output_dir': self.config.get('backtest', {}).get('output_dir', 'backtest_results')
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            # Return a default report instead of raising
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'output_dir': self.config.get('backtest', {}).get('output_dir', 'backtest_results')
            }
            
    def plot_results(self, portfolio_history, output_path: str):
        """
        Plot portfolio performance over time
        
        Args:
            portfolio_history: List of dictionaries containing portfolio history
            output_path: Path to save the plot
        """
        try:
            # Convert to DataFrame if it's a list
            if isinstance(portfolio_history, list):
                results = pd.DataFrame(portfolio_history)
            else:
                results = portfolio_history
                
            # If there are no results, create an empty chart
            if results.empty:
                self.plot_empty_chart(output_path)
                return
                
            # Calculate returns if not present
            if 'returns' not in results.columns:
                results['returns'] = results['portfolio_value'].pct_change()
                
            if 'cumulative_returns' not in results.columns:
                results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot portfolio value
            ax1.plot(results.index, results['portfolio_value'], label='Portfolio Value', color='blue')
            ax1.set_title('Portfolio Performance')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot cumulative returns
            ax2.plot(results.index, results['cumulative_returns'] * 100, label='Cumulative Returns (%)', color='green')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Returns (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            # Create a simple error message plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error generating plot: {e}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.savefig(output_path)
            plt.close()
            
    def plot_empty_chart(self, output_path: str):
        """
        Create a chart for when no trades were executed
        
        Args:
            output_path: Path to save the plot
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Add text explaining no trades
            ax.text(0.5, 0.5, "No trades were executed during the backtest period", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16)
            
            # Add a flat line representing no change in portfolio value
            dates = [datetime.now() - timedelta(days=30), datetime.now()]
            values = [self.initial_capital, self.initial_capital]
            ax.plot(dates, values, 'b-')
            
            ax.set_title('Backtest Results - No Trades')
            ax.set_ylabel('Portfolio Value ($)')
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating empty chart: {e}")
            # Create a very simple error message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No trades executed", 
                     horizontalalignment='center', verticalalignment='center')
            plt.savefig(output_path)
            plt.close()
        
    def _update_portfolio_values(self, portfolio: Dict, current_data: pd.DataFrame):
        """Update portfolio value based on current positions"""
        # Start with cash
        portfolio_value = portfolio['cash']
        
        # Add value of each position
        for token, position in portfolio['positions'].items():
            token_data = current_data[current_data['token'] == token]
            if not token_data.empty:
                # Get current price
                current_price = token_data['close'].iloc[0]
                position['current_price'] = current_price
                position['current_value'] = position['quantity'] * current_price
                portfolio_value += position['current_value']
        
        # Update portfolio value
        portfolio['portfolio_value'] = portfolio_value
        portfolio['equity'] = portfolio_value
        
    def _check_stop_conditions(self, portfolio: Dict, current_data: pd.DataFrame):
        """Check if any positions need to be closed due to stop conditions"""
        positions_to_close = []
        
        for token, position in list(portfolio['positions'].items()):
            token_data = current_data[current_data['token'] == token]
            if token_data.empty:
                continue
                
            # Get current price
            current_price = token_data['close'].iloc[0]
            position['current_price'] = current_price
            
            # Calculate unrealized P&L
            entry_price = position['entry_price']
            is_long = position['direction'] == 'long'
            
            if is_long:
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl_pct = (entry_price - current_price) / entry_price
                
            # Check stop loss
            if position.get('stop_loss') and unrealized_pnl_pct <= -position['stop_loss']:
                positions_to_close.append((token, 'stop_loss'))
                continue
                
            # Check take profit
            if position.get('take_profit') and unrealized_pnl_pct >= position['take_profit']:
                positions_to_close.append((token, 'take_profit'))
                continue
                
            # Check trailing stop
            if self.trailing_stop_pct and 'highest_price' in position:
                if is_long:
                    # For long positions, trail below the highest price
                    trailing_stop_price = position['highest_price'] * (1 - self.trailing_stop_pct)
                    if current_price <= trailing_stop_price and current_price > entry_price:
                        positions_to_close.append((token, 'trailing_stop'))
                        continue
                else:
                    # For short positions, trail above the lowest price
                    trailing_stop_price = position['lowest_price'] * (1 + self.trailing_stop_pct)
                    if current_price >= trailing_stop_price and current_price < entry_price:
                        positions_to_close.append((token, 'trailing_stop'))
                        continue
                        
            # Update highest/lowest price seen
            if is_long:
                if 'highest_price' not in position:
                    position['highest_price'] = max(entry_price, current_price)
                else:
                    position['highest_price'] = max(position['highest_price'], current_price)
            else:
                if 'lowest_price' not in position:
                    position['lowest_price'] = min(entry_price, current_price)
                else:
                    position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Close positions that hit exit conditions
        for token, reason in positions_to_close:
            self._close_position(portfolio, token, reason, current_data)
            
    def _execute_signals(self, portfolio: Dict, signals: Dict, current_data: pd.DataFrame, timestamp: datetime):
        """Execute trading signals"""
        for token, signal in signals.items():
            # Skip if we already have a position in this token
            if token in portfolio['positions']:
                continue
                
            # Skip if we've reached max positions
            if len(portfolio['positions']) >= self.max_positions:
                break
                
            # Check if we have a valid signal
            if signal['signal'] == 0:
                continue
                
            # Get token data
            token_data = current_data[current_data['token'] == token]
            if token_data.empty:
                continue
                
            # Get current price
            current_price = token_data['close'].iloc[0]
            
            # Determine position direction
            direction = 'long' if signal['signal'] > 0 else 'short'
            
            # Calculate position size
            position_size = portfolio['cash'] * self.position_size_pct
            
            # Apply leverage if available
            leverage = signal.get('leverage', 1.0)
            position_size *= leverage
            
            # Calculate quantity
            quantity = position_size / current_price
            
            # Open position
            self._open_position(
                portfolio=portfolio,
                token=token,
                direction=direction,
                quantity=quantity,
                price=current_price,
                timestamp=timestamp,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                leverage=leverage
            )
            
    def _open_position(self, portfolio: Dict, token: str, direction: str, quantity: float, 
                      price: float, timestamp: datetime, stop_loss: float = None, 
                      take_profit: float = None, leverage: float = 1.0):
        """Open a new position"""
        # Calculate position cost
        position_cost = quantity * price / leverage  # Adjust for leverage
        
        # Check if we have enough cash
        if position_cost > portfolio['cash']:
            logger.warning(f"Not enough cash to open position in {token}")
            return
            
        # Deduct cost from cash
        portfolio['cash'] -= position_cost
        
        # Create position
        portfolio['positions'][token] = {
            'token': token,
            'direction': direction,
            'quantity': quantity,
            'entry_price': price,
            'entry_time': timestamp,
            'current_price': price,
            'current_value': quantity * price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage,
            'cost': position_cost
        }
        
        logger.info(f"Opened {direction} position in {token}: {quantity:.6f} @ {price:.6f} (leverage: {leverage:.1f}x)")
        
    def _close_position(self, portfolio: Dict, token: str, reason: str, current_data: pd.DataFrame):
        """Close an existing position"""
        if token not in portfolio['positions']:
            return
            
        position = portfolio['positions'][token]
        
        # Get current price
        token_data = current_data[current_data['token'] == token]
        if token_data.empty:
            logger.warning(f"No price data available to close position in {token}")
            return
            
        close_price = token_data['close'].iloc[0]
        
        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']
        direction = position['direction']
        leverage = position.get('leverage', 1.0)
        
        if direction == 'long':
            pnl = (close_price - entry_price) * quantity * leverage
        else:
            pnl = (entry_price - close_price) * quantity * leverage
            
        # Add value back to cash
        position_value = quantity * close_price / leverage  # Adjust for leverage
        portfolio['cash'] += position_value + pnl
        
        # Log the trade
        pnl_pct = pnl / position['cost'] * 100
        logger.info(f"Closed {direction} position in {token}: {quantity:.6f} @ {close_price:.6f} | " +
                   f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
        
        # Remove position
        del portfolio['positions'][token]
        
    def _create_portfolio_snapshot(self, portfolio: Dict, timestamp: datetime) -> Dict:
        """Create a snapshot of the current portfolio state"""
        return {
            'timestamp': timestamp,
            'portfolio_value': portfolio['portfolio_value'],
            'cash': portfolio['cash'],
            'equity': portfolio['equity'],
            'positions': len(portfolio['positions']),
            'position_details': {token: pos.copy() for token, pos in portfolio['positions'].items()}
        }
        
    def _get_market_data(self, current_date: datetime, prepared_data: Dict[str, Dict]) -> Dict:
        """Get current market data for all symbols"""
        market_data = {
            'price': {},
            'volume': {},
            'features': {}
        }
        
        for symbol, data in prepared_data.items():
            # Get current price and volume from each source
            source_data = {}
            for source, df in data['market_data'].items():
                current_data = df[df['timestamp'] == current_date]
                if not current_data.empty:
                    source_data[source] = current_data.iloc[0].to_dict()
                    
            if source_data:
                market_data['price'][symbol] = source_data
                
            # Get current features
            features_df = data['features']
            current_features = features_df[features_df.index == current_date]
            if not current_features.empty:
                market_data['features'][symbol] = current_features.iloc[0]
                
        return market_data
        
    def _get_best_price(self, current_date: datetime, market_data: Dict) -> Optional[float]:
        """Get best available price across all sources"""
        prices = []
        for source_data in market_data.values():
            current_data = source_data[source_data['timestamp'] == current_date]
            if not current_data.empty:
                prices.append(current_data.iloc[0]['close'])
                
        return min(prices) if prices else None
        
    def _generate_signals(self, current_date: datetime, prepared_data: Dict[str, Dict]) -> Dict:
        """Generate trading signals for all symbols"""
        signals = {}
        
        for symbol, data in prepared_data.items():
            try:
                # Get model predictions
                model = self.model_factory.get_model(symbol)
                features = data['features']
                
                if not features.empty:
                    current_features = features[features.index <= current_date]
                    if not current_features.empty:
                        prediction = model.predict(current_features.iloc[-1:])
                        
                        # Get strategy signals
                        strategy_signals = self.strategy_selector.select_strategies(
                            {'price': data['data'], 'btc': prepared_data.get('BTC/USD', {}).get('data', pd.DataFrame())},
                            current_features.iloc[-1:]
                        )
                        
                        # Combine model prediction and strategy signals
                        signals[symbol] = {
                            'signal': prediction[0] * strategy_signals['signal'],
                            'conviction': strategy_signals['conviction'],
                            'metadata': {
                                'model_prediction': prediction[0],
                                'strategy_signals': strategy_signals['strategy_signals']
                            }
                        }
                        
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                
        return signals
        
    def _calculate_positions(self, signals: Dict, market_data: Dict) -> Dict:
        """Calculate position sizes based on signals and risk parameters"""
        try:
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(market_data['price'])
            
            # Get market regime
            market_regime = self.strategy_selector.detect_market_regime(
                market_data.get('btc', pd.DataFrame())
            )
            
            # Calculate volatility regime
            volatility_regime = self._calculate_volatility_regime(market_data['price'])
            
            # Calculate position sizes
            position_sizes = self.risk_manager.calculate_position_sizes(
                signals,
                self.portfolio_value,
                correlation_matrix,
                market_regime,
                volatility_regime
            )
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"Error calculating positions: {e}")
            return {}
            
    def _execute_trades(self, current_date: datetime, position_sizes: Dict, market_data: Dict):
        """Execute trades based on target position sizes"""
        for symbol, target in position_sizes.items():
            try:
                current_position = self.positions.get(symbol, {'size': 0, 'cost_basis': 0})
                
                # Calculate trade size
                trade_size = target['size'] - current_position['size']
                
                if abs(trade_size) > 0:
                    # Get best execution details
                    execution = self.execution_model.get_best_execution(
                        symbol,
                        abs(trade_size),
                        'buy' if trade_size > 0 else 'sell',
                        market_data['price'][symbol]
                    )
                    
                    if execution is None:
                        logger.warning(f"Could not find valid execution for {symbol}")
                        continue
                        
                    trade_cost = abs(trade_size * execution.execution_price * (1 + execution.fee))
                    
                    # Check if we have enough capital
                    if trade_cost <= self.portfolio_value:
                        # Record trade
                        self.trades.append({
                            'timestamp': current_date,
                            'symbol': symbol,
                            'side': 'buy' if trade_size > 0 else 'sell',
                            'size': abs(trade_size),
                            'price': execution.execution_price,
                            'platform': execution.platform,
                            'fee': execution.fee,
                            'slippage': execution.slippage,
                            'cost': trade_cost,
                            'liquidity_score': execution.liquidity_score
                        })
                        
                        # Update position
                        new_size = current_position['size'] + trade_size
                        new_cost = current_position['cost_basis'] + trade_size * execution.execution_price
                        
                        if new_size == 0:
                            self.positions.pop(symbol, None)
                        else:
                            self.positions[symbol] = {
                                'size': new_size,
                                'cost_basis': new_cost,
                                'platform': execution.platform
                            }
                            
                        # Update portfolio value
                        self.portfolio_value -= trade_cost
                        
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                
    def _record_portfolio_state(self, current_date: datetime):
        """Record current portfolio state"""
        self.portfolio_history.append({
            'timestamp': current_date,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'trades': len(self.trades)
        })
        
    def _generate_backtest_results(self) -> Dict:
        """Generate backtest performance metrics"""
        try:
            # Convert portfolio history to DataFrame
            df = pd.DataFrame(self.portfolio_history)
            df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            df['returns'] = df['portfolio_value'].pct_change()
            
            # Calculate metrics
            total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std() if len(df) > 1 else 0
            max_drawdown = self._calculate_max_drawdown(df['portfolio_value'])
            
            # Calculate platform-specific metrics
            platform_metrics = self._calculate_platform_metrics()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.trades),
                'final_portfolio_value': self.portfolio_value,
                'portfolio_history': df.to_dict(orient='records'),
                'trades': self.trades,
                'platform_metrics': platform_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating backtest results: {e}")
            return {}
            
    @staticmethod
    def _calculate_correlation_matrix(price_data: Dict) -> Dict:
        """Calculate correlation matrix between assets"""
        returns = {}
        for symbol, data in price_data.items():
            if isinstance(data, pd.Series):
                returns[symbol] = data.pct_change()
            else:
                returns[symbol] = data['close'].pct_change()
                
        correlation_matrix = {}
        symbols = list(returns.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = returns[symbol1].corr(returns[symbol2])
                correlation_matrix[(symbol1, symbol2)] = correlation
                correlation_matrix[(symbol2, symbol1)] = correlation
                
        return correlation_matrix
        
    @staticmethod
    def _calculate_volatility_regime(price_data: Dict) -> str:
        """Calculate volatility regime"""
        volatilities = []
        
        for data in price_data.values():
            if isinstance(data, pd.Series):
                returns = data.pct_change()
            else:
                returns = data['close'].pct_change()
                
            vol = returns.std() * np.sqrt(252)
            volatilities.append(vol)
            
        avg_vol = np.mean(volatilities)
        
        if avg_vol < 0.2:  # 20% annualized volatility
            return 'low'
        elif avg_vol > 0.4:  # 40% annualized volatility
            return 'high'
        return 'normal'
        
    @staticmethod
    def _calculate_max_drawdown(values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return abs(drawdown.min())
        
    def _calculate_platform_metrics(self) -> Dict:
        """Calculate performance metrics per platform"""
        platform_metrics = {}
        
        # Initialize metrics for each platform
        for platform in ['jupiter', 'raydium', 'openbook']:
            platform_metrics[platform] = {
                'total_trades': 0,
                'total_volume': 0,
                'total_fees': 0,
                'total_slippage': 0,
                'total_liquidity_score': 0,
                'trade_count': 0
            }
            
        # Process trades
        for trade in self.trades:
            platform = trade.get('platform', 'jupiter')  # Default to Jupiter if not specified
            metrics = platform_metrics[platform]
            
            # Update metrics
            metrics['total_trades'] += 1
            metrics['total_volume'] += trade.get('size', 0)
            metrics['total_fees'] += trade.get('fee', 0)
            metrics['total_slippage'] += trade.get('slippage', 0)
            metrics['total_liquidity_score'] += trade.get('liquidity_score', 0)
            metrics['trade_count'] += 1
            
        # Calculate averages
        for platform, metrics in platform_metrics.items():
            if metrics['trade_count'] > 0:
                metrics['avg_slippage'] = metrics['total_slippage'] / metrics['trade_count']
                metrics['avg_liquidity_score'] = metrics['total_liquidity_score'] / metrics['trade_count']
            else:
                metrics['avg_slippage'] = 0
                metrics['avg_liquidity_score'] = 0
                
            # Clean up intermediate values
            del metrics['total_slippage']
            del metrics['total_liquidity_score']
            del metrics['trade_count']
            
        return platform_metrics 