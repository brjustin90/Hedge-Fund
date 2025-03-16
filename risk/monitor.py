import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskMonitor:
    def __init__(self, config):
        self.config = config
        self.risk_limits = config['risk']['limits']
        self.drawdown_threshold = config['risk']['drawdown_threshold']
        self.volatility_threshold = config['risk']['volatility_threshold']
        self.correlation_threshold = config['risk']['correlation_threshold']
        self.position_history = {}
        self.portfolio_history = []
        self.alerts = []
        
    def update_position(self, symbol, position_data):
        """Update position tracking for a symbol"""
        if symbol not in self.position_history:
            self.position_history[symbol] = []
        self.position_history[symbol].append({
            'timestamp': datetime.now(),
            **position_data
        })
        
    def update_portfolio(self, portfolio_data):
        """Update portfolio tracking"""
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            **portfolio_data
        })
        
    def calculate_drawdown(self, values):
        """Calculate current drawdown"""
        if not values:
            return 0
        peak = max(values)
        current = values[-1]
        return (peak - current) / peak if peak > 0 else 0
        
    def calculate_volatility(self, returns):
        """Calculate rolling volatility"""
        if len(returns) < 2:
            return 0
        return np.std(returns) * np.sqrt(252)  # Annualized
        
    def check_position_limits(self, positions):
        """Check if any position exceeds size limits"""
        alerts = []
        for symbol, position in positions.items():
            size = position['size']
            if size > self.risk_limits['max_position_size']:
                alerts.append({
                    'type': 'position_size',
                    'severity': 'high',
                    'message': f'Position size for {symbol} exceeds maximum limit',
                    'details': {
                        'symbol': symbol,
                        'current_size': size,
                        'limit': self.risk_limits['max_position_size']
                    }
                })
        return alerts
        
    def check_portfolio_limits(self, portfolio):
        """Check if portfolio metrics exceed limits"""
        alerts = []
        
        # Check total exposure
        total_exposure = sum(p['size'] for p in portfolio.values())
        if total_exposure > self.risk_limits['max_portfolio_size']:
            alerts.append({
                'type': 'portfolio_exposure',
                'severity': 'high',
                'message': 'Total portfolio exposure exceeds maximum limit',
                'details': {
                    'current_exposure': total_exposure,
                    'limit': self.risk_limits['max_portfolio_size']
                }
            })
            
        return alerts
        
    def check_drawdown(self, portfolio_values):
        """Check if drawdown exceeds threshold"""
        drawdown = self.calculate_drawdown(portfolio_values)
        if drawdown > self.drawdown_threshold:
            return [{
                'type': 'drawdown',
                'severity': 'high',
                'message': 'Portfolio drawdown exceeds threshold',
                'details': {
                    'current_drawdown': drawdown,
                    'threshold': self.drawdown_threshold
                }
            }]
        return []
        
    def check_volatility(self, returns):
        """Check if volatility exceeds threshold"""
        volatility = self.calculate_volatility(returns)
        if volatility > self.volatility_threshold:
            return [{
                'type': 'volatility',
                'severity': 'medium',
                'message': 'Portfolio volatility exceeds threshold',
                'details': {
                    'current_volatility': volatility,
                    'threshold': self.volatility_threshold
                }
            }]
        return []
        
    def check_correlation(self, correlation_matrix):
        """Check for high correlations between positions"""
        alerts = []
        for (symbol1, symbol2), correlation in correlation_matrix.items():
            if abs(correlation) > self.correlation_threshold:
                alerts.append({
                    'type': 'correlation',
                    'severity': 'medium',
                    'message': f'High correlation between {symbol1} and {symbol2}',
                    'details': {
                        'symbols': [symbol1, symbol2],
                        'correlation': correlation,
                        'threshold': self.correlation_threshold
                    }
                })
        return alerts
        
    def calculate_risk_metrics(self):
        """Calculate current risk metrics"""
        try:
            if not self.portfolio_history:
                return {}
                
            # Convert history to DataFrame for calculations
            df = pd.DataFrame(self.portfolio_history)
            df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            portfolio_values = df['total_value'].values
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            metrics = {
                'current_drawdown': self.calculate_drawdown(portfolio_values),
                'current_volatility': self.calculate_volatility(returns),
                'max_drawdown': max(self.calculate_drawdown(portfolio_values[:i+1]) for i in range(len(portfolio_values))),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                'total_pnl': portfolio_values[-1] - portfolio_values[0] if len(portfolio_values) > 0 else 0
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def monitor_risks(self, current_positions, portfolio_data, correlation_matrix):
        """Main risk monitoring function"""
        try:
            # Update tracking
            for symbol, position in current_positions.items():
                self.update_position(symbol, position)
            self.update_portfolio(portfolio_data)
            
            # Get portfolio values and returns
            portfolio_values = [h['total_value'] for h in self.portfolio_history]
            returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else []
            
            # Collect all alerts
            alerts = []
            alerts.extend(self.check_position_limits(current_positions))
            alerts.extend(self.check_portfolio_limits(current_positions))
            alerts.extend(self.check_drawdown(portfolio_values))
            alerts.extend(self.check_volatility(returns))
            alerts.extend(self.check_correlation(correlation_matrix))
            
            # Update alerts history
            self.alerts.extend(alerts)
            
            # Calculate current risk metrics
            risk_metrics = self.calculate_risk_metrics()
            
            return {
                'alerts': alerts,
                'risk_metrics': risk_metrics,
                'status': 'critical' if any(a['severity'] == 'high' for a in alerts) else 'warning' if alerts else 'normal'
            }
        except Exception as e:
            logger.error(f"Error monitoring risks: {e}")
            return {
                'alerts': [],
                'risk_metrics': {},
                'status': 'error'
            }
            
    def get_historical_alerts(self, timeframe=None):
        """Get historical alerts, optionally filtered by timeframe"""
        if not timeframe:
            return self.alerts
            
        cutoff = datetime.now() - timeframe
        return [alert for alert in self.alerts if alert.get('timestamp', datetime.now()) >= cutoff]
        
    def get_position_history(self, symbol, timeframe=None):
        """Get position history for a symbol"""
        if symbol not in self.position_history:
            return []
            
        history = self.position_history[symbol]
        if timeframe:
            cutoff = datetime.now() - timeframe
            history = [h for h in history if h['timestamp'] >= cutoff]
            
        return history
        
    def get_portfolio_history(self, timeframe=None):
        """Get portfolio history"""
        history = self.portfolio_history
        if timeframe:
            cutoff = datetime.now() - timeframe
            history = [h for h in history if h['timestamp'] >= cutoff]
            
        return history 