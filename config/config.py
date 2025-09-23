# config/config.py
import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# API Credentials
DYDX_API_KEY = os.getenv("DYDX_API_KEY")
DYDX_API_SECRET = os.getenv("DYDX_API_SECRET")
DYDX_API_PASSPHRASE = os.getenv("DYDX_API_PASSPHRASE")

# Other configs remain the same...
INDEXER_URL = "indexer.dydx.trade"
TRADING_MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]