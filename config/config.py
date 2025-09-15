# config/config.py
# Settings for Crypto Bot 2.0

# Network Settings
INDEXER_URL = "indexer.dydx.trade"  # Mainnet for data, no https://
TESTNET_INDEXER_URL = "indexer.v4testnet.dydx.exchange"  # Testnet, no https://
WEBSOCKET_URL = "wss://indexer.dydx.trade/v4/ws"  # WebSocket for real-time

# Trading Parameters
TRADING_MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
CANDLE_RESOLUTION = "15MINS"  # 15-min trading time frame (dYdX v4 format)
CORRELATION_TIME_FRAMES = ["1HOUR", "4HOUR"]  # For correlation analysis
DATA_DAYS = 30  # Fetch 30 days (~2880 15-min candles)

# Model Parameters
FORMULA_7_WINDOW = 20  # For Formula 7
LSTM_LOOKBACK = 60  # For LSTM
BB_WINDOW = 20  # Bollinger Bands
GARCH_WINDOW = 20  # GARCH

# API & Trading
DYDX_API_KEY = "dydx1d02jk0dxlpgjtgt8fndcju47t0h92982qsvfxs"  # Testnet API key
DYDX_API_SECRET = "end armor emotion such receive solve time neither couch agree six always brown advance saddle matter wisdom allow differ license horse gas cycle midnight"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Optional

# Risk Management
STOP_LOSS_PERCENT = 0.02  # 2%
TAKE_PROFIT_PERCENT = 0.05  # 5%
MAX_POSITION_SIZE = 0.01  # 1% of portfolio