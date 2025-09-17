# models/feature_engineering.py
import pandas as pd
import numpy as np
from models.formula_7 import calculate_formula_7


def generate_features(df: pd.DataFrame, df_btc: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generates a comprehensive set of features for the trading model.
    Now includes funding_rate as a pass-through feature.

    Args:
        df (pd.DataFrame): The primary asset's data, must include 'funding_rate'.
        df_btc (pd.DataFrame, optional): Bitcoin's OHLCV data for correlation.

    Returns:
        pd.DataFrame: The DataFrame with all features added.
    """
    if df.empty:
        return df

    # 1. Formula 7 Suite
    df = calculate_formula_7(df, n=20)
    df['f7_momentum'] = df['f7'].diff().fillna(0)
    df['f7_acceleration'] = df['f7_momentum'].diff().fillna(0)

    # 2. Bollinger Bands Suite
    bb_window = 20
    df['bb_sma'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_sma']
    df.drop(columns=['bb_sma', 'bb_std', 'bb_upper', 'bb_lower'], inplace=True)

    # 3. Market Context
    df['trend_ema_50'] = df['close'] / df['close'].ewm(span=50, adjust=False).mean() - 1

    # 4. Correlation Suite (only if df_btc is provided)
    if df_btc is not None and not df_btc.empty:
        # Merge based on timestamp to align data for calculations
        merged = pd.merge_asof(df.sort_values('started_at'),
                               df_btc[['started_at', 'log_returns']].rename(columns={'log_returns': 'log_returns_btc'}),
                               on='started_at')

        df['rolling_corr_btc'] = merged['log_returns'].rolling(window=100).corr(merged['log_returns_btc'])
        df['btc_momentum'] = merged['log_returns_btc'].rolling(window=10).mean()

    # The 'funding_rate' is assumed to be already present and forward-filled in the pipeline
    # We just ensure it's clean
    if 'funding_rate' not in df.columns:
        df['funding_rate'] = 0.0

    # Fill any remaining NaNs that might have been created
    df.fillna(0, inplace=True)

    return df