# models/predictor.py
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils.logger import setup_logger

logger = setup_logger('predictor', 'predictor.log')


def predict_hmm_regime(df, model_path='models/hmm_regime_model.pkl'):
    try:
        model = joblib.load(model_path)
        features = df[['f7_momentum', 'volatility']].values
        features = np.nan_to_num(features)
        df['hmm_regime'] = model.predict(features)
    except FileNotFoundError:
        df['hmm_regime'] = 0
    return df


def predict_lstm_forecast(df, model_path='models/lstm_forecast_model.keras', sequence_length=60):
    try:
        model = tf.keras.models.load_model(model_path)
        features_to_use = ['close', 'f7', 'volatility', 'funding_rate']
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Sanitize data before transforming
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

        dummy_array = np.zeros((len(predictions_scaled), len(features_to_use)))
        dummy_array[:, 0] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_array)[:, 0]

        df['lstm_forecast'] = np.nan
        df.iloc[sequence_length:, df.columns.get_loc('lstm_forecast')] = predictions

        # --- FIX: Replaced deprecated fillna(method=...) with modern .ffill() ---
        df['lstm_forecast'] = df['lstm_forecast'].ffill()
        df['lstm_forecast'].fillna(0, inplace=True)  # Fill any remaining NaNs at the beginning

    except (FileNotFoundError, IOError):
        df['lstm_forecast'] = 0.0
    return df