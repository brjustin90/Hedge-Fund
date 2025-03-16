import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
from datetime import datetime
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from models.features import FeatureEngineering

logger = logging.getLogger(__name__)

class MLPredictor:
    """Machine learning model for predicting memecoin price movements"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_type = self.config.get('model_type', 'classification')  # 'classification' or 'regression'
        self.target_horizon = self.config.get('target_horizon', 6)  # hours ahead to predict
        self.threshold = self.config.get('threshold', 0.6)  # threshold for buy signals
        self.feature_engineering = FeatureEngineering(config)
        self.models = {}  # Dictionary to store trained models for each token
        self.model_dir = self.config.get('model_dir', 'models/saved')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for training or prediction
        
        Args:
            df: Raw price data DataFrame
            
        Returns:
            X: Features DataFrame
            y: Target Series or None if for prediction
        """
        # Create features
        features_df = self.feature_engineering.create_features(df)
        if features_df.empty:
            return pd.DataFrame(), None
            
        # Determine target column
        if self.model_type == 'classification':
            target_col = f'target_up_{self.target_horizon}'
        else:
            target_col = f'future_return_{self.target_horizon}'
            
        # Select features
        selected_features = self.feature_engineering.select_features(features_df, target_col)
        
        # Split features and target
        if target_col in selected_features.columns:
            X = selected_features.drop(columns=[target_col])
            y = selected_features[target_col]
            return X, y
        else:
            logger.warning(f"Target column {target_col} not found in features")
            return pd.DataFrame(), None
            
    def train(self, df: pd.DataFrame, token: str) -> Dict:
        """
        Train a model for a specific token
        
        Args:
            df: DataFrame with historical price data
            token: Token symbol
            
        Returns:
            Dict with training metrics
        """
        logger.info(f"Training ML model for {token}")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        if X.empty or y is None:
            logger.warning(f"Not enough data to train model for {token}")
            return {'success': False, 'error': 'Not enough data'}
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create model pipeline
        if self.model_type == 'classification':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                ))
            ])
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42
                ))
            ])
            
        # Train model
        try:
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            if self.model_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            else:
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                
            # Save model
            self.models[token] = model
            model_path = os.path.join(self.model_dir, f"{token.replace('/', '_')}.joblib")
            joblib.dump(model, model_path)
            
            logger.info(f"Model for {token} trained successfully with metrics: {metrics}")
            return {'success': True, 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"Error training model for {token}: {e}")
            return {'success': False, 'error': str(e)}
            
    def predict(self, df: pd.DataFrame, token: str) -> Dict:
        """
        Make predictions for a token
        
        Args:
            df: DataFrame with recent price data
            token: Token symbol
            
        Returns:
            Dict with prediction results
        """
        # Check if model exists
        if token not in self.models:
            # Try to load model from disk
            model_path = os.path.join(self.model_dir, f"{token.replace('/', '_')}.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[token] = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model for {token}: {e}")
                    return {'signal': 0, 'confidence': 0, 'prediction': 0}
            else:
                logger.warning(f"No model found for {token}")
                return {'signal': 0, 'confidence': 0, 'prediction': 0}
                
        # Preprocess data
        X, _ = self.preprocess_data(df)
        if X.empty:
            logger.warning(f"Could not preprocess data for prediction for {token}")
            return {'signal': 0, 'confidence': 0, 'prediction': 0}
            
        # Get latest data point for prediction
        X_latest = X.iloc[-1:].copy()
        
        # Make prediction
        try:
            model = self.models[token]
            
            if self.model_type == 'classification':
                # Predict probability of price going up
                prob_up = model.predict_proba(X_latest)[0, 1]
                prediction = prob_up
                
                # Generate signal based on probability and threshold
                if prob_up > self.threshold:
                    signal = 1.0  # Buy signal
                    confidence = prob_up
                elif prob_up < (1 - self.threshold):
                    signal = -1.0  # Sell signal
                    confidence = 1 - prob_up
                else:
                    signal = 0  # No clear signal
                    confidence = 0.5
                    
            else:
                # Predict future return
                predicted_return = model.predict(X_latest)[0]
                prediction = predicted_return
                
                # Generate signal based on predicted return
                if predicted_return > 0.05:  # 5% return threshold
                    signal = 1.0
                    confidence = min(predicted_return * 10, 1.0)  # Scale confidence
                elif predicted_return < -0.02:  # -2% return threshold
                    signal = -1.0
                    confidence = min(abs(predicted_return) * 10, 1.0)
                else:
                    signal = 0
                    confidence = 0.5
                    
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {token}: {e}")
            return {'signal': 0, 'confidence': 0, 'prediction': 0}
            
    def batch_train(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train models for multiple tokens
        
        Args:
            data_dict: Dictionary mapping token symbols to price DataFrames
            
        Returns:
            Dict with training results for each token
        """
        results = {}
        for token, df in data_dict.items():
            results[token] = self.train(df, token)
        return results 