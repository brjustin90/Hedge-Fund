import asyncio
import logging
import yaml
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

from data.collectors.jupiter_collector import JupiterCollector
from models.ml_predictor import MLPredictor
from models.features import FeatureEngineering
from strategies.ml_strategy import MLStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_training_data(start_date: datetime, end_date: datetime, tokens: list):
    """
    Collect historical data for training ML models
    
    Args:
        start_date: Start date for data collection
        end_date: End date for data collection
        tokens: List of tokens to collect data for
        
    Returns:
        Dictionary mapping tokens to DataFrames of historical data
    """
    logger.info(f"Collecting training data for {len(tokens)} tokens from {start_date} to {end_date}")
    
    # Initialize data collector
    collector = JupiterCollector()
    
    # Collect data for each token
    token_data_dict = {}
    for token in tqdm(tokens, desc="Collecting token data"):
        try:
            # Get price history for token
            df = await collector.get_price_history(token, start_date, end_date)
            
            if df is not None and not df.empty and len(df) > 24:  # Need at least 24 data points
                token_data_dict[token] = df
                logger.info(f"Collected {len(df)} data points for {token}")
            else:
                logger.warning(f"Not enough data for {token}, skipping")
        except Exception as e:
            logger.error(f"Error collecting data for {token}: {e}")
            
    logger.info(f"Successfully collected data for {len(token_data_dict)}/{len(tokens)} tokens")
    
    return token_data_dict

async def train_and_optimize_models(config: dict):
    """
    Train and optimize ML models for trading
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Load config
        ml_config = config.get('ml', {})
        
        # Create output directory
        output_dir = Path("models/saved")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set date range for training - use longer period for better training
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=30)     # Last 30 days
        
        # Get tokens to train models for
        collector = JupiterCollector()
        tokens = await collector.get_top_memecoin_tokens(limit=15)
        
        # Collect training data
        token_data_dict = await collect_training_data(start_date, end_date, tokens)
        
        if not token_data_dict:
            logger.error("No training data collected, aborting")
            return
            
        # Create ML predictor
        ml_predictor = MLPredictor(ml_config)
        
        # Train models
        logger.info("Training ML models...")
        results = ml_predictor.batch_train(token_data_dict)
        
        # Log results
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"Successfully trained {success_count}/{len(results)} models")
        
        # Log metrics for successful models
        for token, result in results.items():
            if result.get('success', False):
                metrics = result.get('metrics', {})
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Model for {token}: {metrics_str}")
                
        # Perform hyperparameter optimization for one sample token to find best parameters
        if token_data_dict:
            sample_token = next(iter(token_data_dict))
            sample_data = token_data_dict[sample_token]
            
            logger.info(f"Performing hyperparameter optimization using {sample_token}")
            
            # Create feature engineering instance
            feature_eng = FeatureEngineering(ml_config)
            
            # Create features
            features_df = feature_eng.create_features(sample_data)
            
            if not features_df.empty:
                # Grid search for best parameters
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10, 20]
                }
                
                best_params = {}
                best_score = 0
                
                # Simple grid search
                for n_estimators in param_grid['n_estimators']:
                    for max_depth in param_grid['max_depth']:
                        for min_samples_split in param_grid['min_samples_split']:
                            # Update config with new parameters
                            test_config = ml_config.copy()
                            if 'hyperparameters' not in test_config:
                                test_config['hyperparameters'] = {}
                                
                            test_config['hyperparameters'].update({
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split
                            })
                            
                            # Create and train model with these parameters
                            test_predictor = MLPredictor(test_config)
                            test_result = test_predictor.train(sample_data, sample_token)
                            
                            if test_result.get('success', False):
                                metrics = test_result.get('metrics', {})
                                score = metrics.get('f1' if test_config.get('model_type') == 'classification' else 'rmse', 0)
                                
                                # For regression, lower RMSE is better, so negate it
                                if test_config.get('model_type') != 'classification':
                                    score = -score
                                    
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth,
                                        'min_samples_split': min_samples_split
                                    }
                                    
                                logger.info(f"Parameters {test_config['hyperparameters']} - Score: {score:.4f}")
                
                # Log best parameters
                logger.info(f"Best hyperparameters: {best_params} with score {best_score:.4f}")
                
                # Save best parameters to config file
                config_file = Path("config/ml_config.yaml")
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                        
                    if 'ml' in config_data and 'hyperparameters' in config_data['ml']:
                        config_data['ml']['hyperparameters'].update(best_params)
                        
                        with open(config_file, 'w') as f:
                            yaml.dump(config_data, f, default_flow_style=False)
                            
                        logger.info(f"Updated {config_file} with best hyperparameters")
            else:
                logger.warning("Could not create features for hyperparameter optimization")
                
        logger.info("Model training and optimization completed.")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
async def main():
    """Main function to run model training"""
    try:
        # Load configuration
        config_path = Path("config/ml_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
            logger.warning(f"Config file {config_path} not found, using default configuration")
            
        # Train and optimize models
        await train_and_optimize_models(config)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        
if __name__ == "__main__":
    asyncio.run(main()) 