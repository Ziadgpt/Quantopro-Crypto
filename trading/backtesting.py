# trading/backtester.py
import pandas as pd
import numpy as np
import joblib
from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('backtester', 'backtest.log')


def run_backtest(market=TRADING_MARKETS[0], model_path='models/synthesizer_model.pkl'):
    """
    Runs a vector-based backtest of the trading strategy.
    """
    logger.info(f"--- Starting Backtest for {market} ---")

    # 1. Load Model and Data
    try:
        payload = joblib.load(model_path)
        model = payload['model']
        features = payload['features']
    except FileNotFoundError:
        logger.error(f"Model not found at {model_path}. Please train the synthesizer first.")
        return

    df = read_data(market)
    if df.empty or not all(f in df.columns for f in features):
        logger.error("Data is insufficient for backtesting. Run the full pipeline.")
        return

    # 2. Generate Predictions (Conviction Score)
    df['score'] = model.predict_proba(df[features])[:, 1]

    # 3. Trading Logic Simulation
    entry_threshold = 0.70  # Enter a trade if conviction is > 70%
    take_profit = 1.01  # 1% take profit
    stop_loss = 0.995  # 0.5% stop loss

    positions = np.zeros(len(df))
    returns = []

    for i in range(1, len(df)):
        # Entry condition
        if positions[i - 1] == 0 and df['score'].iloc[i] > entry_threshold:
            positions[i] = 1  # Enter long position
            entry_price = df['close'].iloc[i]

            # Simulate holding the position
            for j in range(i + 1, len(df)):
                # Take Profit condition
                if df['high'].iloc[j] >= entry_price * take_profit:
                    returns.append(take_profit - 1)
                    i = j  # Move main loop forward
                    break
                # Stop Loss condition
                elif df['low'].iloc[j] <= entry_price * stop_loss:
                    returns.append(stop_loss - 1)
                    i = j  # Move main loop forward
                    break
                # If neither is hit, continue holding
                positions[j] = 1
            else:  # Loop finished without break (end of data)
                i = len(df)

    # 4. Performance Metrics
    if not returns:
        logger.warning("No trades were executed during the backtest.")
        return

    total_trades = len(returns)
    wins = sum(1 for r in returns if r > 0)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    returns_series = pd.Series(returns)
    cumulative_returns = (1 + returns_series).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100

    logger.info("--- Backtest Results ---")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Return: {total_return:.2f}%")

    # You can add more metrics like Sharpe Ratio, Max Drawdown etc. here


if __name__ == '__main__':
    run_backtest()