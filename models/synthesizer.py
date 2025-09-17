# models/synthesizer_model.py
import pandas as pd, numpy as np, lightgbm as lgb, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('synthesizer_trainer', 'synthesizer_model.log')


def train_synthesizer_model(market=TRADING_MARKETS[0], model_path='models/synthesizer_model.pkl'):
    logger.info(f"Training Synthesizer for {market}.")
    df = read_data(market)
    if df.empty or 'hmm_regime' not in df.columns: return

    base_features = [
        'f7', 'f7_momentum', 'f7_acceleration', 'bb_percent_b', 'bb_bandwidth',
        'trend_ema_50', 'funding_rate', 'volatility', 'sentiment', 'hmm_regime', 'lstm_forecast'
    ]
    features = base_features + ['rolling_corr_btc', 'btc_momentum'] if market != 'BTC-USD' else base_features

    df['future_high'] = df['high'].rolling(window=16).max().shift(-16)
    df['target'] = np.where(df['future_high'] > df['close'] * 1.005, 1, 0)
    df.dropna(inplace=True)
    X, y = df[features], df['target']

    if len(X) < 100: logger.error("Not enough data for training."); return

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, scale_pos_weight=scale_pos_weight)
    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)
    logger.info(f"--- Evaluation ---\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    joblib.dump({'model': lgbm, 'features': features}, model_path)
    logger.info(f"✅ Synthesizer saved to {model_path}.")