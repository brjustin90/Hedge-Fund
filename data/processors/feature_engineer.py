import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.price_features = PriceFeatureGenerator()
        self.volume_features = VolumeFeatureGenerator()
        self.onchain_features = OnChainFeatureGenerator()
        self.sentiment_features = SentimentFeatureGenerator()
        
    def generate_features(self, raw_data, timeframe):
        """Generate all features from raw data"""
        features = {}
        
        # Generate technical indicators from price data
        if 'price' in raw_data:
            features.update(self.price_features.generate(raw_data['price'], timeframe))
            
        # Generate volume-based features
        if 'volume' in raw_data:
            features.update(self.volume_features.generate(raw_data['volume'], raw_data.get('price')))
            
        # Generate on-chain features if available
        if 'onchain' in raw_data:
            features.update(self.onchain_features.generate(raw_data['onchain']))
            
        # Generate sentiment features if available
        if 'sentiment' in raw_data:
            features.update(self.sentiment_features.generate(raw_data['sentiment']))
            
        return features

class PriceFeatureGenerator:
    def generate(self, price_data, timeframe):
        """Generate features from price data"""
        features = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
                
            # Ensure we have OHLCV data
            if 'open' not in df.columns:
                df['open'] = df['price']
            if 'high' not in df.columns:
                df['high'] = df['price']
            if 'low' not in df.columns:
                df['low'] = df['price']
            if 'close' not in df.columns:
                df['close'] = df['price']
                
            symbol_features = {}
            
            # Basic price features
            symbol_features['returns'] = df['close'].pct_change()
            symbol_features['log_returns'] = np.log1p(symbol_features['returns'])
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                symbol_features[f'returns_{window}'] = df['close'].pct_change(window)
                symbol_features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
                symbol_features[f'ma_{window}'] = df['close'].rolling(window).mean()
                symbol_features[f'ma_ratio_{window}'] = df['close'] / symbol_features[f'ma_{window}']
                
            # Momentum indicators
            symbol_features['rsi_14'] = ta.momentum.RSIIndicator(df['close']).rsi()
            symbol_features['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
            symbol_features['macd'] = ta.trend.MACD(df['close']).macd_diff()
            symbol_features['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
            # Trend indicators
            symbol_features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            symbol_features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # Volatility indicators
            bb = ta.volatility.BollingerBands(df['close'])
            symbol_features['bb_width'] = bb.bollinger_wband()
            symbol_features['bb_ratio'] = (df['close'] - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_mavg())
            
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
            symbol_features['atr'] = atr.average_true_range()
            
            # Support/Resistance
            symbol_features['support'] = df['low'].rolling(20).min()
            symbol_features['resistance'] = df['high'].rolling(20).max()
            symbol_features['price_to_support'] = df['close'] / symbol_features['support']
            symbol_features['price_to_resistance'] = df['close'] / symbol_features['resistance']
            
            # Combine all features for this symbol
            features[symbol] = pd.DataFrame(symbol_features)
            
        return features

class VolumeFeatureGenerator:
    def generate(self, volume_data, price_data=None):
        """Generate features from volume data"""
        features = {}
        
        for symbol, df in volume_data.items():
            if df.empty:
                continue
                
            symbol_features = {}
            
            # Basic volume features
            symbol_features['volume'] = df['volume']
            symbol_features['volume_ma_5'] = df['volume'].rolling(5).mean()
            symbol_features['volume_ma_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratios
            symbol_features['volume_ratio_5'] = df['volume'] / symbol_features['volume_ma_5']
            symbol_features['volume_ratio_20'] = df['volume'] / symbol_features['volume_ma_20']
            
            # If price data is available, calculate price-volume features
            if price_data and symbol in price_data:
                price_df = price_data[symbol]
                
                # On-balance volume
                symbol_features['obv'] = ta.volume.OnBalanceVolumeIndicator(price_df['close'], df['volume']).on_balance_volume()
                
                # Volume-weighted average price
                symbol_features['vwap'] = (price_df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                # Price-volume trend
                symbol_features['pvt'] = ta.volume.VolumePriceTrendIndicator(price_df['close'], df['volume']).volume_price_trend()
                
            # Volume momentum
            symbol_features['volume_momentum'] = df['volume'].pct_change()
            symbol_features['volume_momentum_ma_5'] = symbol_features['volume_momentum'].rolling(5).mean()
            
            # Combine all features for this symbol
            features[symbol] = pd.DataFrame(symbol_features)
            
        return features

class OnChainFeatureGenerator:
    def generate(self, onchain_data):
        """Generate features from on-chain data"""
        features = {}
        
        for symbol, df in onchain_data.items():
            if df.empty:
                continue
                
            symbol_features = {}
            
            # Holder metrics
            if 'holder_count' in df.columns:
                symbol_features['holder_count'] = df['holder_count']
                symbol_features['holder_change'] = df['holder_count'].diff()
                symbol_features['holder_change_pct'] = df['holder_count'].pct_change()
                
            # Transaction metrics
            if 'tx_count' in df.columns:
                symbol_features['tx_count'] = df['tx_count']
                symbol_features['tx_volume'] = df['tx_volume']
                symbol_features['avg_tx_size'] = df['tx_volume'] / df['tx_count']
                
            # Smart money movements
            if 'smart_money_flow' in df.columns:
                symbol_features['smart_money_flow'] = df['smart_money_flow']
                symbol_features['smart_money_ratio'] = df['smart_money_flow'] / df['tx_volume']
                
            # Liquidity metrics
            if 'liquidity' in df.columns:
                symbol_features['liquidity'] = df['liquidity']
                symbol_features['liquidity_change'] = df['liquidity'].diff()
                symbol_features['liquidity_change_pct'] = df['liquidity'].pct_change()
                
            # Combine all features for this symbol
            features[symbol] = pd.DataFrame(symbol_features)
            
        return features

class SentimentFeatureGenerator:
    def generate(self, sentiment_data):
        """Generate features from sentiment data"""
        features = {}
        
        for symbol, df in sentiment_data.items():
            if df.empty:
                continue
                
            symbol_features = {}
            
            # Basic sentiment metrics
            if 'sentiment_score' in df.columns:
                symbol_features['sentiment_score'] = df['sentiment_score']
                symbol_features['sentiment_ma_5'] = df['sentiment_score'].rolling(5).mean()
                symbol_features['sentiment_std_5'] = df['sentiment_score'].rolling(5).std()
                
            # Sentiment momentum
            symbol_features['sentiment_momentum'] = df['sentiment_score'].diff()
            symbol_features['sentiment_acceleration'] = symbol_features['sentiment_momentum'].diff()
            
            # Sentiment extremes
            symbol_features['sentiment_zscore'] = (df['sentiment_score'] - df['sentiment_score'].rolling(20).mean()) / df['sentiment_score'].rolling(20).std()
            
            # Volume-weighted sentiment
            if 'mention_volume' in df.columns:
                symbol_features['weighted_sentiment'] = df['sentiment_score'] * df['mention_volume']
                symbol_features['mention_volume_ma_5'] = df['mention_volume'].rolling(5).mean()
                
            # Combine all features for this symbol
            features[symbol] = pd.DataFrame(symbol_features)
            
        return features 