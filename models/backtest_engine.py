import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config['initial_capital']
        self.position_size_pct = config['position_size_pct']
        self.max_positions = config['max_positions']
        self.stop_loss_pct = config['stop_loss_pct']
        self.take_profit_pct = config['take_profit_pct']
        self.trailing_stop_pct = config['trailing_stop_pct']
        self.min_profit_to_trail = config['min_profit_to_trail']
        
        # Risk management
        self.risk_params = config['risk_management']
        self.max_drawdown_pct = self.risk_params['max_drawdown_pct']
        self.daily_loss_limit_pct = self.risk_params['daily_loss_limit_pct']
        
        # Execution parameters
        self.exec_params = config['execution']
        self.entry_timeout = self.exec_params['entry_timeout_bars']
        self.exit_timeout = self.exec_params['exit_timeout_bars']
        self.min_bars_between_trades = self.exec_params['min_bars_between_trades']
        
        # Initialize state
        self.reset_state()
        
    def reset_state(self) -> None:
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = {}  # token -> position info
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        self.max_equity = self.initial_capital
        
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        data_collector: Callable,
        tokens: List[str],
        ml_strategy: Optional[object] = None
    ) -> Dict:
        """Run backtest simulation"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.reset_state()
        
        # Initialize data collection
        current_date = start_date
        while current_date <= end_date:
            # Get data for current date
            daily_data = data_collector(current_date, tokens, self.positions)
            
            if not daily_data:
                current_date += timedelta(days=1)
                continue
                
            # Process each token
            for token, df in daily_data.items():
                if df.empty:
                    continue
                    
                # Get trading signals if ML strategy is provided
                if ml_strategy:
                    signals = ml_strategy.get_signals({token: df})[token]
                    df = pd.concat([df, signals], axis=1)
                
                # Process each bar
                for idx, row in df.iterrows():
                    self._process_bar(token, row)
                    
            # Update daily stats
            daily_pnl = self.capital - self.initial_capital - sum(self.daily_pnl.values())
            self.daily_pnl[current_date.date()] = daily_pnl
            
            # Check daily loss limit
            if daily_pnl < -self.initial_capital * self.daily_loss_limit_pct:
                logger.warning(f"Daily loss limit hit on {current_date.date()}")
                self._close_all_positions(df.iloc[-1])
                
            # Update equity curve
            self.equity_curve.append({
                'timestamp': current_date,
                'equity': self.capital + self._get_positions_value(df.iloc[-1])
            })
            
            # Update maximum equity
            current_equity = self.equity_curve[-1]['equity']
            self.max_equity = max(self.max_equity, current_equity)
            
            # Check maximum drawdown
            drawdown = (self.max_equity - current_equity) / self.max_equity
            if drawdown > self.max_drawdown_pct:
                logger.warning(f"Maximum drawdown limit hit on {current_date.date()}")
                self._close_all_positions(df.iloc[-1])
                
            current_date += timedelta(days=1)
            
        return self._generate_results()
        
    def _process_bar(self, token: str, row: pd.Series) -> None:
        """Process a single price bar"""
        # Check existing position
        if token in self.positions:
            self._handle_position_update(token, row)
        # Check for new position
        elif (
            len(self.positions) < self.max_positions and
            row.get('entry_signal', 0) == 1 and
            self._can_enter_trade(token, row)
        ):
            self._enter_position(token, row)
            
    def _handle_position_update(self, token: str, row: pd.Series) -> None:
        """Update existing position"""
        position = self.positions[token]
        entry_price = position['entry_price']
        current_price = row['close']
        
        # Calculate returns
        unrealized_return = (current_price - entry_price) / entry_price
        
        # Check exit conditions
        exit_triggered = False
        exit_reason = None
        
        # Stop loss
        if unrealized_return < -self.stop_loss_pct:
            exit_triggered = True
            exit_reason = 'stop_loss'
            
        # Take profit
        elif unrealized_return > self.take_profit_pct:
            exit_triggered = True
            exit_reason = 'take_profit'
            
        # Trailing stop
        elif (
            unrealized_return > self.min_profit_to_trail and
            (current_price - position['max_price']) / position['max_price'] < -self.trailing_stop_pct
        ):
            exit_triggered = True
            exit_reason = 'trailing_stop'
            
        # Exit signal
        elif row.get('exit_signal', 0) == 1:
            exit_triggered = True
            exit_reason = 'signal'
            
        # Update maximum price
        if current_price > position['max_price']:
            position['max_price'] = current_price
            
        # Exit position if triggered
        if exit_triggered:
            self._exit_position(token, row, exit_reason)
            
    def _enter_position(self, token: str, row: pd.Series) -> None:
        """Enter a new position"""
        entry_price = row['close']
        position_size = self.capital * self.position_size_pct
        quantity = position_size / entry_price
        
        self.positions[token] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': row['timestamp'],
            'max_price': entry_price
        }
        
        logger.info(f"Entered {token} position at {entry_price}")
        
    def _exit_position(self, token: str, row: pd.Series, reason: str) -> None:
        """Exit an existing position"""
        position = self.positions[token]
        exit_price = row['close']
        
        # Calculate PnL
        pnl = position['quantity'] * (exit_price - position['entry_price'])
        self.capital += pnl
        
        # Record trade
        self.trades.append({
            'token': token,
            'entry_time': position['entry_time'],
            'exit_time': row['timestamp'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'return': (exit_price - position['entry_price']) / position['entry_price'],
            'exit_reason': reason
        })
        
        logger.info(f"Exited {token} position at {exit_price}, PnL: {pnl:.2f}")
        
        # Remove position
        del self.positions[token]
        
    def _close_all_positions(self, row: pd.Series) -> None:
        """Close all open positions"""
        for token in list(self.positions.keys()):
            self._exit_position(token, row, 'risk_management')
            
    def _can_enter_trade(self, token: str, row: pd.Series) -> bool:
        """Check if we can enter a new trade"""
        # Check if we have enough capital
        position_size = self.capital * self.position_size_pct
        if position_size < 100:  # Minimum position size
            return False
            
        # Check correlation with existing positions
        if self.positions:
            # TODO: Implement correlation check
            pass
            
        # Check volatility
        if 'price_volatility_12' in row:
            volatility = row['price_volatility_12']
            if (
                volatility < self.risk_params['min_volatility'] or
                volatility > self.risk_params['max_volatility']
            ):
                return False
                
        return True
        
    def _get_positions_value(self, row: pd.Series) -> float:
        """Calculate total value of open positions"""
        return sum(
            pos['quantity'] * row['close']
            for token, pos in self.positions.items()
        )
        
    def _generate_results(self) -> Dict:
        """Generate backtest results summary"""
        if not self.trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'trades': []
            }
            
        equity_df = pd.DataFrame(self.equity_curve)
        returns = equity_df['equity'].pct_change().dropna()
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        max_drawdown = max(1 - equity_df['equity'] / equity_df['equity'].cummax())
        win_rate = sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_pnl': self.daily_pnl
        } 