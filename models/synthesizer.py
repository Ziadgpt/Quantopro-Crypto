# models/synthesizer.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('synthesizer_trainer', 'synthesizer_model.log')


def train_synthesizer_model(market=TRADING_MARKETS[0], model_path='models/synthesizer_model.pkl'):
    """Trains a LightGBM model to synthesize features into a final trading signal."""
    logger.info(f"Starting Synthesizer model training for {market}.")
    df = read_data(market)

    if df.empty or 'hmm_regime' not in df.columns:
        logger.error("Data is insufficient for Synthesizer training. HMM features are missing.")
        return

    # --- FIX: Refined the feature set to remove potential noise like lstm_forecast ---
    base_features = [
        'f7',
        'f7_momentum',
        'bb_percent_b',
        'bb_bandwidth',
        'trend_ema_50',
        'funding_rate',
        'volatility',
        'hmm_regime'
    ]
    # --- END FIX ---

    features = base_features + ['rolling_corr_btc', 'btc_momentum'] if market != 'BTC-USD' else base_features

    df['future_high'] = df['high'].rolling(window=16).max().shift(-16)
    df['target'] = np.where(df['future_high'] > df['close'] * 1.005, 1, 0)

    df.dropna(inplace=True)
    X, y = df[features], df['target']

    if len(X) < 200:
        logger.error(f"Not enough data for {market} for meaningful training. Need at least 200 rows, got {len(X)}.")
        return

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if y_train.value_counts().get(1, 0) == 0:
        logger.error("Training data contains only one class. Cannot train the model.")
        return
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, scale_pos_weight=scale_pos_weight)
    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)

    if len(np.unique(y_test)) > 1:
        logger.info(f"--- Synthesizer Evaluation for {market} ---\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    else:
        logger.warning("Evaluation skipped: y_test contains only one class.")

    joblib.dump({'model': lgbm, 'features': features}, model_path)
    logger.info(f"✅ Synthesizer model for {market} saved successfully to {model_path}.")