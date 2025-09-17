# config.py
# This file contains the core configuration constants for the trading bot.

# dYdX v4 Indexer API URL (Corrected: base domain only)
INDEXER_URL = "indexer.dydx.trade"

# List of cryptocurrency markets to trade against USD/USDT
TRADING_MARKETS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "XRP-USD"
]

# Primary candle resolution for high-frequency signals
CANDLE_RESOLUTION = "15MINS"

# Number of days of historical data to fetch for analysis
DATA_DAYS = 30