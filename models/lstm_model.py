# models/lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('lstm_trainer', 'lstm_model.log')


def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_lstm_model(market=TRADING_MARKETS[0], model_path='models/lstm_forecast_model.keras', sequence_length=60):
    logger.info(f"Starting LSTM training using data from {market}.")
    df = read_data(market)
    features_to_use = ['close', 'f7', 'volatility', 'funding_rate']
    if df.empty or not all(f in df.columns for f in features_to_use):
        logger.error("Data is insufficient for LSTM training. Please run the pipeline.")
        return

    # --- FIX: Sanitize data to prevent NaN/inf values before scaling ---
    data = df[features_to_use].values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # --- END FIX ---

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, sequence_length)
    if len(X) == 0:
        logger.error("Not enough data to create sequences for LSTM training.")
        return

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),  # Modern way to specify input shape
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=25),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("Fitting LSTM model...")
    model.fit(X, y, batch_size=32, epochs=50, verbose=0)  # Set verbose=0 to reduce log spam

    model.save(model_path)
    logger.info(f"✅ LSTM model training complete and saved successfully to {model_path}.")