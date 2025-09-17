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
    """Creates sequences of data for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length, 0]  # Predict the next 'close' price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_lstm_model(market=TRADING_MARKETS[0], model_path='models/lstm_forecast_model.keras', sequence_length=60):
    """
    Trains an LSTM model to predict future price movements.

    Args:
        market (str): The market to use for training data.
        model_path (str): Path to save the trained Keras model.
        sequence_length (int): The number of past time steps to use for prediction.
    """
    logger.info(f"Starting LSTM training using data from {market}.")

    # 1. Load and Prepare Data
    df = read_data(market)
    features_to_use = ['close', 'f7', 'volatility', 'funding_rate']
    if df.empty or not all(f in df.columns for f in features_to_use):
        logger.error("Data is insufficient for LSTM training. Please run the pipeline.")
        return

    data = df[features_to_use].values

    # 2. Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 3. Create Sequences
    X, y = create_sequences(scaled_data, sequence_length)
    if len(X) == 0:
        logger.error("Not enough data to create sequences for LSTM training.")
        return

    # 4. Build the LSTM Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=25),
        tf.keras.layers.Dense(units=1)
    ])

    # 5. Compile and Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("Fitting LSTM model...")
    model.fit(X, y, batch_size=32, epochs=50, verbose=1)  # verbose=1 shows progress

    # 6. Save the Trained Model
    logger.info(f"Saving trained LSTM model to {model_path}")
    model.save(model_path)
    logger.info("✅ LSTM model training complete and saved successfully.")


if __name__ == '__main__':
    train_lstm_model()