# trading/backtesting.py
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.database import read_data
from config.config import TRADING_MARKETS
from utils.logger import setup_logger

logger = setup_logger('backtester', 'backtest.log')


def run_backtest(market=TRADING_MARKETS[0], initial_capital=10000):
    """
    Runs a vector-based backtest on the trained Synthesizer model's signals.
    """
    logger.info(f"Starting backtest for {market} with initial capital ${initial_capital:,.2f}")

    # 1. Load Data and Model
    df = read_data(market)
    try:
        model_package = joblib.load('models/synthesizer_model.pkl')
        model = model_package['model']
        features = model_package['features']
    except FileNotFoundError:
        logger.error("Synthesizer model not found. Please train the model first.")
        return {"error": "Model not found."}

    if df.empty or not all(f in df.columns for f in features):
        logger.error("Data is insufficient for backtesting. Run the pipeline and training first.")
        return {"error": "Insufficient data."}

    # 2. Generate Predictions (Signals)
    X = df[features]
    df['signal'] = model.predict(X)
    df['signal'] = df['signal'].shift(1).fillna(0)  # Shift to prevent lookahead bias

    # --- FIX: Corrected the trade simulation logic ---
    # 3. Simulate Trades
    position = 0
    entry_price = 0
    entry_time = None
    entry_bar = 0
    trades = []
    equity_over_time = [initial_capital]
    capital = initial_capital

    for i, row in df.iterrows():
        # Entry condition: Not in a position and signal is 1
        if position == 0 and row['signal'] == 1:
            position = 1
            entry_price = row['open']  # Enter on the open of the next bar
            entry_time = row['started_at']
            entry_bar = i

        # Exit condition: In a position and 16 bars have passed
        elif position == 1 and i >= entry_bar + 16:
            exit_price = row['open']
            exit_time = row['started_at']

            pnl_pct = (exit_price - entry_price) / entry_price
            capital = capital * (1 + pnl_pct)
            equity_over_time.append(capital)

            trades.append({
                'entry_time': entry_time, 'entry_price': entry_price,
                'exit_time': exit_time, 'exit_price': exit_price,
                'pnl_pct': pnl_pct * 100
            })

            # Reset position state
            position = 0
            entry_price = 0
            entry_time = None
            entry_bar = 0
    # --- END FIX ---

    # 4. Calculate Performance Metrics
    if not trades:
        logger.warning("No trades were executed during the backtest.")
        return {"message": "No trades executed."}

    trades_df = pd.DataFrame(trades)

    # Create an equity curve DataFrame
    equity_series = pd.Series(equity_over_time)
    equity_df = pd.DataFrame({'equity': equity_series})
    # We need to find the timestamps for when equity changed
    trade_exit_indices = [df.index.get_loc(trades_df.iloc[i]['exit_time']) for i in range(len(trades_df))]
    equity_timestamps = [df.iloc[0]['started_at']] + [df.loc[idx, 'started_at'] for idx in trade_exit_indices]
    equity_df['started_at'] = equity_timestamps

    net_profit = capital - initial_capital
    win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
    total_trades = len(trades_df)

    peak = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak) / peak
    max_drawdown = drawdown.min() * 100

    logger.info(f"Backtest complete. Net Profit: ${net_profit:,.2f}, Win Rate: {win_rate:.2f}%")

    return {
        "kpis": {
            "Net Profit ($)": f"${net_profit:,.2f}",
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Max Drawdown (%)": f"{max_drawdown:.2f}%",
            "Total Trades": total_trades
        },
        "equity_curve": equity_df[['started_at', 'equity']],
        "trades_log": trades_df
    }