# models/hmm_model.py
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('hmm_trainer', 'hmm_model.log')


def train_hmm_model(market=TRADING_MARKETS[0], n_states=4, model_path='models/hmm_regime_model.pkl'):
    """
    Trains a Gaussian Hidden Markov Model to identify market regimes.

    Args:
        market (str): The market to use for training data (e.g., 'BTC-USD').
        n_states (int): The number of hidden states (regimes) to identify.
        model_path (str): Path to save the trained model.
    """
    logger.info(f"Starting HMM training for {n_states} regimes using data from {market}.")

    # 1. Load Data
    df = read_data(market)
    if df.empty or 'f7_momentum' not in df.columns or 'volatility' not in df.columns:
        logger.error("Data is insufficient for HMM training. Please run the pipeline first.")
        return

    # 2. Prepare Feature Matrix
    # We use F7 momentum and volatility to define the regimes
    features = df[['f7_momentum', 'volatility']].values

    # Handle potential infinite values if any
    if np.any(~np.isfinite(features)):
        logger.warning("Non-finite values found in features. Replacing with 0.")
        features = np.nan_to_num(features)

    # 3. Initialize and Train HMM
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)

    try:
        logger.info("Fitting HMM model...")
        model.fit(features)
    except Exception as e:
        logger.error(f"Error fitting HMM model: {e}", exc_info=True)
        return

    # 4. Save the Trained Model
    logger.info(f"Saving trained HMM model to {model_path}")
    joblib.dump(model, model_path)
    logger.info("✅ HMM model training complete and saved successfully.")


if __name__ == '__main__':
    # Allows for direct execution of the training script
    train_hmm_model()