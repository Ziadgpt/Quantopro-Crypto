# models/predictor.py
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils.logger import setup_logger
from functools import lru_cache  # <-- Import the caching tool

logger = setup_logger('predictor', 'predictor.log')


# --- NEW: Caching helper functions ---
@lru_cache(maxsize=None)
def load_hmm_package(path):
    """Loads the HMM model and scaler from disk, caching the result."""
    logger.info(f"Loading HMM model from disk: {path}")
    return joblib.load(path)


@lru_cache(maxsize=None)
def load_keras_model(path):
    """Loads the Keras model from disk, caching the result."""
    logger.info(f"Loading Keras model from disk: {path}")
    return tf.keras.models.load_model(path)


# --- END NEW ---


def predict_hmm_regime(df, model_path='models/hmm_regime_model.pkl'):
    """
    Predicts the HMM regime for the given data.
    """
    try:
        # --- FIX: Use the cached loader ---
        hmm_package = load_hmm_package(model_path)
        # --- END FIX ---

        model = hmm_package['model']
        scaler = hmm_package['scaler']

        features = df[['f7_momentum', 'volatility']].values
        features = np.nan_to_num(features)
        scaled_features = scaler.transform(features)
        df['hmm_regime'] = model.predict(scaled_features)

        # This log will now only appear once per model, not for every prediction
        # logger.info("Successfully predicted HMM regimes.")
    except FileNotFoundError:
        logger.warning(f"HMM model file not found at {model_path}. Assigning default regime 0.")
        df['hmm_regime'] = 0
    except Exception as e:
        logger.error(f"Error predicting HMM regime: {e}", exc_info=True)
        df['hmm_regime'] = 0
    return df


def predict_lstm_forecast(df, model_path='models/lstm_forecast_model.keras', sequence_length=60):
    try:
        # --- FIX: Use the cached loader ---
        model = load_keras_model(model_path)
        # --- END FIX ---

        features_to_use = ['close', 'f7', 'volatility', 'funding_rate']
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_to_scale = np.nan_to_num(df[features_to_use].values, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_data = scaler.fit_transform(data_to_scale)

        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])
        X = np.array(X)

        if X.shape[0] == 0:
            df['lstm_forecast'] = 0.0
            return df

        predictions_scaled = model.predict(X, verbose=0)

        dummy_array = np.zeros((len(predictions_scaled), len(predictions_scaled[0] * (len(features_to_use) - 1))))
        dummy_array[:, 0] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_array)[:, 0]

        df['lstm_forecast'] = np.nan
        df.iloc[sequence_length:, df.columns.get_loc('lstm_forecast')] = predictions
        df['lstm_forecast'] = df['lstm_forecast'].ffill()
        df['lstm_forecast'].fillna(0, inplace=True)

    except (FileNotFoundError, IOError):
        logger.warning(f"LSTM model not found at {model_path}. Assigning default forecast 0.0.")
        df['lstm_forecast'] = 0.0
    except Exception as e:
        logger.error(f"Error during LSTM prediction: {e}", exc_info=True)
        df['lstm_forecast'] = 0.0
    return df