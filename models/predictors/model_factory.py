import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from datetime import datetime

logger = logging.getLogger(__name__)

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class ModelFactory:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        
    def create_model(self, model_type: str, input_dim: int) -> object:
        """Create a new model of specified type"""
        try:
            if model_type == 'xgboost':
                return xgb.XGBRegressor(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    objective='reg:squarederror'
                )
            elif model_type == 'lstm':
                return LSTM(
                    input_dim=input_dim,
                    hidden_dim=64,
                    num_layers=2,
                    output_dim=1
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
            
    def get_model(self, token: str, model_type: str, input_dim: int) -> object:
        """Get or create model for a token"""
        model_key = f"{token}_{model_type}"
        if model_key not in self.models:
            self.models[model_key] = self.create_model(model_type, input_dim)
        return self.models[model_key]
        
    def get_scaler(self, token: str) -> StandardScaler:
        """Get or create scaler for a token"""
        if token not in self.scalers:
            self.scalers[token] = StandardScaler()
        return self.scalers[token]
        
    def prepare_data(self, data: pd.DataFrame, token: str, sequence_length: int = 24) -> tuple:
        """Prepare data for model training/prediction"""
        try:
            # Get relevant features
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'liquidity_score'
            ]
            
            # Add technical indicators
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log1p(data['returns'])
            data['volatility'] = data['returns'].rolling(window=24).std()
            
            # Add to features list
            features.extend(['returns', 'log_returns', 'volatility'])
            
            # Get feature data
            X = data[features].fillna(0)
            
            # Scale features
            scaler = self.get_scaler(token)
            X_scaled = scaler.fit_transform(X)
            
            # Prepare sequences for LSTM
            sequences = []
            for i in range(len(X_scaled) - sequence_length):
                sequences.append(X_scaled[i:(i + sequence_length)])
                
            return np.array(sequences), features
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
            
    def save_models(self, path: str):
        """Save all models and scalers"""
        try:
            for model_key, model in self.models.items():
                if isinstance(model, xgb.XGBRegressor):
                    model.save_model(f"{path}/{model_key}.json")
                elif isinstance(model, LSTM):
                    torch.save(model.state_dict(), f"{path}/{model_key}.pth")
                    
            # Save scalers
            for token, scaler in self.scalers.items():
                np.save(f"{path}/{token}_scaler.npy", [scaler.mean_, scaler.scale_])
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
            
    def load_models(self, path: str):
        """Load all models and scalers"""
        try:
            # Implementation depends on how models were saved
            pass
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

class XGBoostModel:
    def __init__(self, params):
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            **params
        }
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            watchlist = [(dtrain, 'train')]
            
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return {}
        return self.model.get_score(importance_type='gain')
        
    def save(self, path):
        """Save model"""
        if self.model is not None:
            self.model.save_model(f"{path}_xgboost.model")
            
    def load(self, path):
        """Load model"""
        self.model = xgb.Booster()
        self.model.load_model(f"{path}_xgboost.model")

class LSTMModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = params.get('input_dim', 32)
        self.hidden_dim = params.get('hidden_dim', 64)
        self.num_layers = params.get('num_layers', 2)
        self.dropout = params.get('dropout', 0.2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        last_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc(last_out)
        return out
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.get('learning_rate', 0.001))
        
        n_epochs = self.params.get('epochs', 100)
        batch_size = self.params.get('batch_size', 32)
        
        for epoch in range(n_epochs):
            self.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # Validation
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val)
                    val_loss = criterion(val_outputs, y_val.unsqueeze(1))
                    
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {total_loss/len(X_train):.4f}, Val Loss: {val_loss.item():.4f}')
                    
    def predict(self, X):
        """Make predictions"""
        self.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self(X)
            
        return predictions.cpu().numpy()
        
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'params': self.params
        }, f"{path}_lstm.pt")
        
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(f"{path}_lstm.pt")
        self.load_state_dict(checkpoint['model_state_dict'])
        self.params = checkpoint['params']

class EnsembleModel:
    def __init__(self, params):
        self.params = params
        self.models = []
        
        # Create models
        for model_type, model_params in params['models']:
            self.models.append((model_type, ModelFactory.create_model(model_type, model_params)))
            
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in ensemble"""
        for model_type, model in self.models:
            logger.info(f"Training {model_type} model...")
            model.train(X_train, y_train, X_val, y_val)
            
    def predict(self, X):
        """Make predictions using ensemble"""
        predictions = []
        
        for model_type, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # Average predictions
        return np.mean(predictions, axis=0)
        
    def save(self, path):
        """Save all models"""
        for model_type, model in self.models:
            model.save(f"{path}_{model_type}")
            
    def load(self, path):
        """Load all models"""
        for model_type, model in self.models:
            model.load(f"{path}_{model_type}") 