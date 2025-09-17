# models/formula_7.py
import pandas as pd
import numpy as np


def calculate_formula_7(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Calculates the Formula 7 indicator and its signals.

    Formula: [(Prob High * Avg High Return) - (Prob Low * Avg Low Return)] / n

    Args:
        df (pd.DataFrame): DataFrame with at least a 'close' column.
        n (int): The lookback period for the calculation.

    Returns:
        pd.DataFrame: The original DataFrame with 'f7' and 'f7_signal' columns added.
    """
    if 'close' not in df.columns or df.empty:
        return df  # Return original df if it's empty or doesn't have a close price

    # Calculate returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # Calculate rolling components
    positive_returns = df['returns'].where(df['returns'] > 0, 0)
    negative_returns = df['returns'].where(df['returns'] < 0, 0)

    # ph: Probability of High (fraction of positive candles)
    ph = (positive_returns != 0).rolling(window=n).sum() / n

    # rh: Average High Return (mean of positive returns)
    rh = positive_returns.rolling(window=n).sum() / (positive_returns != 0).rolling(window=n).sum()

    # pl: Probability of Low (fraction of negative candles)
    pl = (negative_returns != 0).rolling(window=n).sum() / n

    # rl: Average Low Return (mean of negative returns)
    # Note: rl will be negative, so we use its absolute value in the spirit of the formula
    rl = negative_returns.rolling(window=n).sum() / (negative_returns != 0).rolling(window=n).sum()

    # Calculate Formula 7
    # We subtract (pl * rl) which is adding a positive number since rl is negative.
    formula_7 = ((ph * rh) - (pl * rl)) / n

    # Normalize the result to be more like a standard oscillator (e.g., by multiplying by 10000 for visibility)
    df['f7'] = formula_7.fillna(0) * 10000

    # Generate Zero-Line Crossover Signal
    # Signal: 1 for bullish crossover, -1 for bearish crossunder, 0 for no change
    df['f7_signal'] = 0
    df.loc[(df['f7'] > 0) & (df['f7'].shift(1) <= 0), 'f7_signal'] = 1  # Bullish crossover
    df.loc[(df['f7'] < 0) & (df['f7'].shift(1) >= 0), 'f7_signal'] = -1  # Bearish crossunder

    # Clean up intermediate columns
    df.drop(columns=['returns'], inplace=True)

    return df