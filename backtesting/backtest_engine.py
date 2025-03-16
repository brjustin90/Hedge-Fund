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
        
    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        data_collector,
        tokens: List[str]
    ) -> Dict:
        """Run backtest on historical data"""
        try:
            logger.info(f"Starting backtest for {len(tokens)} tokens from {start_date} to {end_date}")
            
            results = {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'platform_metrics': {}
            }
            
            if not tokens:
                logger.warning("No tokens provided for backtest")
                return results
                
            # Initialize portfolio
            portfolio = {
                'cash': self.initial_capital,
                'positions': {},
                'total_value': self.initial_capital,
                'history': []
            }
            
            # Initialize signal history
            self.signal_history = {token: {'buy_signals': 0, 'last_signal_time': None} for token in tokens}
            
            # Collect historical data for all tokens - with batch processing
            all_data = []
            token_data_dict = {}  # Dictionary to store data for ML training
            batch_size = 5  # Process tokens in batches to avoid overwhelming API
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i+batch_size]
                batch_tasks = []
                
                # Create tasks for parallel execution
                for token in batch_tokens:
                    task = asyncio.create_task(self._collect_token_data(token, start_date, end_date, data_collector))
                    batch_tasks.append(task)
                
                # Wait for all tasks in batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in data collection: {result}")
                        continue
                    if result is not None:  # Only add valid data
                        all_data.append(result)
                        token = batch_tokens[j]
                        token_data_dict[token] = result
                        
            if not all_data:
                logger.warning("No data collected for any tokens")
                return results
                
            # Combine all data
            logger.info(f"Combining data from {len(all_data)} tokens")
            data = pd.concat(all_data, ignore_index=False)
            
            # Check if timestamp is in the index or as a column
            if 'timestamp' not in data.columns and data.index.name == 'timestamp':
                # Reset index to make timestamp a column
                data = data.reset_index()
            elif 'timestamp' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                # If we have a DatetimeIndex but it's not named 'timestamp'
                data = data.reset_index()
                data.rename(columns={'index': 'timestamp'}, inplace=True)
                
            # Sort by timestamp
            try:
                data = data.sort_values('timestamp')
            except KeyError as e:
                logger.error(f"Error in backtest: {str(e)}")
                return results
                
            logger.info(f"Combined data shape: {data.shape}")
            
            # Train ML models if enabled
            if self.use_ml_strategy:
                logger.info("Training ML models with historical data")
                self.ml_strategy.train_models(token_data_dict)
            
            # Process each timestamp
            logger.info(f"Processing unique timestamps: {data['timestamp'].nunique()}")
            timestamps = data['timestamp'].unique()
            
            # Split data into training and testing periods
            train_end_idx = len(timestamps) // 3  # Use first 1/3 for training
            train_timestamps = timestamps[:train_end_idx]
            test_timestamps = timestamps[train_end_idx:]
            
            logger.info(f"Training period: {train_timestamps[0]} to {train_timestamps[-1]}")
            logger.info(f"Testing period: {test_timestamps[0]} to {test_timestamps[-1]}")
            
            # Process testing period
            for timestamp in test_timestamps:
                try:
                    # Get data for current timestamp
                    group = data[data['timestamp'] == timestamp]
                    
                    # Update existing positions
                    self._update_positions(portfolio, group)
                    
                    # Get ML trading signals if enabled
                    if self.use_ml_strategy:
                        ml_signals = self.ml_strategy.get_signals(group, portfolio)
                        
                        # Execute ML trades
                        for token, signal in ml_signals.items():
                            if signal['action'] == 'buy' and token not in portfolio['positions'] and len(portfolio['positions']) < self.max_positions:
                                logger.info(f"ML BUY signal for {token} at {signal['price']} with confidence {signal['confidence']:.4f}")
                                self._execute_trade(portfolio, token, 'buy', signal['price'], timestamp, signal['confidence'])
                            elif signal['action'] == 'sell' and token in portfolio['positions']:
                                logger.info(f"ML SELL signal for {token} at {signal['price']} with confidence {signal['confidence']:.4f}")
                                self._execute_trade(portfolio, token, 'sell', signal['price'], timestamp, signal['confidence'], 'ml_prediction')
                    
                    # Use traditional strategy as fallback or in addition to ML
                    for token in group['token'].unique():
                        # Skip if we already have a position or if max positions reached
                        if token in portfolio['positions'] or len(portfolio['positions']) >= self.max_positions:
                            continue
                            
                        token_data = group[group['token'] == token]
                        
                        # Skip if not enough data
                        if len(token_data) < 2:
                            continue
                            
                        # Calculate basic signals
                        # Check if 'close' exists, otherwise use 'price'
                        if 'close' in token_data.columns:
                            price = token_data['close'].iloc[-1]
                        elif 'price' in token_data.columns:
                            price = token_data['price'].iloc[-1]
                        else:
                            logger.warning(f"No price data found for {token}, skipping")
                            continue
                        
                        # Calculate price change (handle first row)
                        token_prev_data = data[(data['token'] == token) & (data['timestamp'] < timestamp)].sort_values('timestamp')
                        if len(token_prev_data) >= 1:
                            # Check if 'close' exists, otherwise use 'price'
                            if 'close' in token_prev_data.columns:
                                prev_price = token_prev_data['close'].iloc[-1]
                            elif 'price' in token_prev_data.columns:
                                prev_price = token_prev_data['price'].iloc[-1]
                            else:
                                logger.warning(f"No previous price data found for {token}, using current price")
                                prev_price = price
                                
                            price_change = (price - prev_price) / prev_price
                        else:
                            price_change = 0.0
                            
                        volume = token_data['volume'].iloc[-1] if 'volume' in token_data.columns else 0
                        liquidity_score = token_data['liquidity_score'].iloc[-1] if 'liquidity_score' in token_data.columns else 0.5
                        
                        # Traditional entry conditions - if ML didn't already generate a signal
                        if not self.use_ml_strategy or token not in ml_signals:
                            # Entry conditions with lowered thresholds to generate more trades
                            volatility_ok = abs(price_change) >= self.vol_threshold * 0.7  # Lower threshold
                            price_change_ok = price_change > 0.01  # Lower threshold
                            volume_ok = volume > 50000  # Lower threshold
                            liquidity_ok = liquidity_score > 0.5  # Lower threshold
                            
                            if price_change_ok and volume_ok and liquidity_ok and volatility_ok:
                                # Update signal history
                                if token not in self.signal_history:
                                    self.signal_history[token] = {'buy_signals': 0, 'last_signal_time': None}
                                    
                                # Increment buy signals counter
                                self.signal_history[token]['buy_signals'] += 1
                                self.signal_history[token]['last_signal_time'] = timestamp
                                
                                # Check if we have enough confirmation signals
                                if self.signal_history[token]['buy_signals'] >= self.entry_confirmation:
                                    logger.info(f"Traditional BUY signal for {token} at {price:.6f} (price_change: {price_change:.2%}, volume: {volume:.0f}, liquidity: {liquidity_score:.2f})")
                                    self._execute_trade(portfolio, token, 'buy', price, timestamp, liquidity_score)
                                    # Reset counter after trade
                                    self.signal_history[token]['buy_signals'] = 0
                            
                    # Record portfolio state
                    portfolio_value = self._calculate_portfolio_value(portfolio, group)
                    portfolio['history'].append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': portfolio['cash'],
                        'n_positions': len(portfolio['positions'])
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
            # Calculate final results
            if portfolio['history']:
                logger.info(f"Calculating final results from {len(portfolio['history'])} portfolio snapshots")
                history_df = pd.DataFrame(portfolio['history'])
                
                # Calculate returns
                history_df['returns'] = history_df['portfolio_value'].pct_change().fillna(0)
                
                # Calculate metrics
                total_return = (history_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
                daily_returns = history_df['returns'].fillna(0)
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
                max_drawdown = (history_df['portfolio_value'] / history_df['portfolio_value'].cummax() - 1).min()
                win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
                
                results.update({
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'platform_metrics': self._calculate_platform_metrics()
                })
                
                logger.info(f"Backtest completed successfully with return: {total_return:.2%}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
    async def _collect_token_data(self, token: str, start_date: datetime, end_date: datetime, data_collector) -> Optional[pd.DataFrame]:
        """Collect and process data for a single token"""
        try:
            # Get price history
            price_data = await data_collector.get_price_history(token, start_date, end_date)
            
            # Check if data is already a DataFrame
            if isinstance(price_data, pd.DataFrame):
                if price_data.empty:
                    logger.warning(f"No price data found for {token}")
                    return None
            # Convert dictionary to DataFrame
            elif isinstance(price_data, dict):
                if not price_data or 'prices' not in price_data or not price_data['prices']:
                    logger.warning(f"No price data found for {token}")
                    return None
                
                # Convert the price data from dictionary to DataFrame
                prices = price_data.get('prices', [])
                volumes = price_data.get('total_volumes', [])
                liquidities = price_data.get('liquidity', [])
                
                # Create records
                records = []
                for i, (timestamp, price) in enumerate(prices):
                    # Convert millisecond timestamp to datetime
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    
                    # Get volume if available
                    volume = volumes[i][1] if i < len(volumes) else 0
                    
                    # Get liquidity if available
                    liquidity = liquidities[i][1] if i < len(liquidities) else 1000000
                    
                    records.append({
                        'timestamp': dt,
                        'price': price,
                        'volume': volume,
                        'token': token,
                        'liquidity_score': liquidity / 1000000.0  # Normalize to 0-1 range
                    })
                
                price_data = pd.DataFrame(records)
            else:
                logger.warning(f"Unexpected data type for {token}: {type(price_data)}")
                return None
            
            # Add token info if not already there
            if 'token' not in price_data.columns:
                price_data['token'] = token
                
            # Add liquidity score if not already there
            if 'liquidity_score' not in price_data.columns:
                # Try to get liquidity data
                try:
                    liquidity_data = await data_collector.get_liquidity_data(token)
                    liquidity_score = data_collector.calculate_liquidity_score(liquidity_data)
                    price_data['liquidity_score'] = liquidity_score
                except:
                    # Default liquidity score
                    price_data['liquidity_score'] = 0.7
            
            return price_data
            
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
        
    def generate_report(self, results: pd.DataFrame) -> Dict:
        """Generate performance report"""
        try:
            total_days = (results['timestamp'].max() - results['timestamp'].min()).days
            
            # Calculate metrics
            total_return = results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1
            daily_returns = results['returns'].fillna(0)
            
            # Risk metrics
            volatility = daily_returns.std() * np.sqrt(365)
            sharpe_ratio = (daily_returns.mean() * 365) / volatility if volatility != 0 else 0
            max_drawdown = (results['portfolio_value'] / results['portfolio_value'].cummax() - 1).min()
            
            report = {
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'win_rate': len(daily_returns[daily_returns > 0]) / len(daily_returns),
                'avg_trade_duration': timedelta(days=7),  # Placeholder
                'total_fees': 1000,  # Placeholder
                'avg_slippage': 0.001,  # Placeholder
                'avg_liquidity_score': 0.8  # Placeholder
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
            
    def plot_results(self, results: pd.DataFrame, output_path: str):
        """Plot backtest results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(results['timestamp'], results['portfolio_value'], label='Portfolio Value')
            plt.title('Backtest Results')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot returns
            plt.subplot(2, 1, 2)
            plt.plot(results['timestamp'], results['cumulative_returns'], label='Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise
        
    def _update_portfolio_value(self, current_date: datetime, prepared_data: Dict[str, Dict]):
        """Update portfolio value based on current positions"""
        portfolio_value = self.initial_capital
        
        for symbol, position in self.positions.items():
            if symbol in prepared_data:
                current_price = self._get_best_price(current_date, prepared_data[symbol]['market_data'])
                if current_price is not None:
                    position_value = position['size'] * current_price
                    portfolio_value += position_value - position['cost_basis']
                    
        self.portfolio_value = portfolio_value
        
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