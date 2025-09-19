# models/tune_synthesizer.py
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

# Add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('synthesizer_tuner', 'synthesizer_tuner.log')


def run_tuning_study(n_trials=50):
    """
    This function encapsulates the entire tuning process.
    It can now be called from other scripts or a Celery task.
    """
    logger.info(f"Starting hyperparameter tuning study for {n_trials} trials...")

    market = TRADING_MARKETS[0]
    df = read_data(market)

    if df.empty:
        logger.error(f"DataFrame for {market} is empty. Cannot run tuning.")
        return

    base_features = [
        'f7', 'f7_momentum', 'bb_percent_b', 'bb_bandwidth',
        'trend_ema_50', 'funding_rate', 'volatility', 'hmm_regime'
    ]
    features = base_features + ['rolling_corr_btc', 'btc_momentum'] if market != 'BTC-USD' else base_features
    df['future_high'] = df['high'].rolling(window=16).max().shift(-16)
    df['target'] = np.where(df['future_high'] > df['close'] * 1.005, 1, 0)
    df.dropna(inplace=True)
    X, y = df[features], df['target']

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbose': -1
        }
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        params['scale_pos_weight'] = scale_pos_weight
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, pos_label=1)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logger.info("Study finished!")
    logger.info(f"Best trial's F1-score: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters found: {study.best_params}")

    logger.info("Training final model with best hyperparameters...")
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['random_state'] = 42
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    best_params['scale_pos_weight'] = scale_pos_weight
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    model_path = 'models/synthesizer_model.pkl'
    joblib.dump({'model': final_model, 'features': features}, model_path)
    logger.info(f"✅ Final, tuned Synthesizer model saved to {model_path}")


# This allows the script to still be run directly if needed
if __name__ == "__main__":
    run_tuning_study()