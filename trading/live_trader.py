# trading/live_trader.py
import asyncio
import sys
import os

# Add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dydx_v4_client.chain.async_client import AsyncClient
from dydx_v4_client.chain.aerial.wallet import LocalWallet
from dydx_v4_client.chain.aerial.private_key import PrivateKey
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.ws.indexer_ws_client import IndexerWsClient
from config.config import DYDX_API_KEY, DYDX_API_SECRET, DYDX_API_PASSPHRASE
from utils.logger import setup_logger

logger = setup_logger('live_trader', 'live_trader.log')

async def main():
    """
    Main function to connect to dYdX v4 WebSocket and stream market data.
    """
    logger.info("Initializing live trading bot...")

    # For now, we only need the IndexerWsClient for market data.
    # The AsyncClient and Wallet are for placing trades, which we will add later.
    indexer_ws_client = IndexerWsClient()

    # Define a callback function to handle incoming messages
    def on_message(message: dict):
        """
        This function is called every time a new message is received from the WebSocket.
        """
        market = message.get('id')
        contents = message.get('contents')
        if contents:
            price = contents.get('price')
            if market and price:
                # For now, we just print the live price.
                # In the future, this is where we'll run our prediction models.
                print(f"LIVE PRICE [{market}]: ${price}")
                logger.info(f"Received price for {market}: {price}")

    # Subscribe to the v4-markets channel for a specific market
    market_to_watch = "BTC-USD"
    logger.info(f"Subscribing to live price feed for {market_to_watch}...")
    await indexer_ws_client.connect()
    await indexer_ws_client.subscribe_to_v4_markets(
        market=market_to_watch,
        callback=on_message
    )

    # Keep the connection alive indefinitely
    logger.info("Connection successful. Streaming live data...")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down live trader.")