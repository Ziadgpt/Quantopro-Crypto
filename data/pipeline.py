# data/pipeline.py
import pandas as pd
import numpy as np
import requests
import asyncio
from datetime import datetime, timedelta
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from config.config import INDEXER_URL, TRADING_MARKETS, CANDLE_RESOLUTION, DATA_DAYS
from data.database import save_data, read_data
from utils.logger import setup_logger

logger = setup_logger('data_pipeline', 'data_pipeline.log')

async def setup_client():
    """Initialize dYdX v4 mainnet client."""
    return IndexerClient(f"https://{INDEXER_URL}")

async def get_available_markets():
    """Asynchronously fetch available markets from dYdX v4 mainnet."""
    try:
        client = await setup_client()
        response = await client.markets.get_perpetual_markets()
        markets = list(response.get('markets', response.get('data', {}).get('markets', {})).keys())
        logger.info(f"Successfully fetched {len(markets)} available markets: {markets}")
        print(f"Available markets: {markets}")
        return markets
    except Exception as e:
        logger.error(f"Error fetching markets: {e}", exc_info=True)
        print(f"Error fetching markets: {e}")
        return []

def fetch_sentiment(market='BTC-USD'):
    """Fetch sentiment from CoinGecko."""
    coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana', 'ADA-USD': 'cardano', 'XRP-USD': 'ripple'}
    coin = coin_map.get(market, market.split('-')[0].lower())
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        sentiment_score = data.get('sentiment_votes_up_percentage', 0.0)
        logger.info(f"Fetched sentiment for {market}: {sentiment_score}")
        print(f"Fetched sentiment for {market}: {sentiment_score}")
        return sentiment_score
    except Exception as e:
        logger.error(f"Error fetching sentiment for {market}: {e}")
        print(f"Error fetching sentiment for {market}: {e}")
        return 0.0

async def fetch_data(market='BTC-USD', timeframe=CANDLE_RESOLUTION, limit=1000, retries=3, delay=2):
    """Asynchronously fetch OHLCV from dYdX v4 mainnet with retries."""
    markets_to_try = [market, market.replace('USD', 'USDT')]
    client = await setup_client()
    for mkt in markets_to_try:
        for attempt in range(retries):
            try:
                logger.info(f"Attempting to fetch data for {mkt} (attempt {attempt + 1}/{retries})")
                to_iso = datetime.utcnow()
                from_iso = to_iso - timedelta(days=DATA_DAYS)
                response = await client.markets.get_perpetual_market_candles(
                    market=mkt,
                    resolution=timeframe,
                    from_iso=from_iso.isoformat() + 'Z',
                    to_iso=to_iso.isoformat() + 'Z',
                    limit=limit
                )
                logger.debug(f"Raw API response for {mkt}: {response}")
                candles = response.get('candles', response.get('data', {}).get('candles', []))
                if not candles:
                    logger.warning(f"No candles returned for {mkt} at {timeframe}. Attempt {attempt + 1}/{retries}.")
                    if attempt < retries - 1:
                        await asyncio.sleep(delay)
                    continue
                df = pd.DataFrame([{
                    'started_at': pd.to_datetime(c.get('startedAt'), errors='coerce'),
                    'open': float(c.get('open', 0)),
                    'high': float(c.get('high', 0)),
                    'low': float(c.get('low', 0)),
                    'close': float(c.get('close', 0)),
                    'base_token_volume': float(c.get('baseTokenVolume', 0))
                } for c in candles])
                df.dropna(subset=['started_at'], inplace=True)
                if df.empty:
                    logger.warning(f"DataFrame empty after processing candles for {mkt}.")
                    continue
                df.sort_values('started_at', inplace=True, ignore_index=True)
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
                df['volatility'] = df['log_returns'].rolling(window=20).std().fillna(0)
                df['sentiment'] = fetch_sentiment(market)
                df_temp = df.set_index('started_at')
                agg_rules = {
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                    'base_token_volume': 'sum', 'log_returns': 'sum',
                    'volatility': 'mean', 'sentiment': 'last'
                }
                df_1h = df_temp.resample('h').agg(agg_rules).dropna(subset=['open']).reset_index()
                df_4h = df_temp.resample('4h').agg(agg_rules).dropna(subset=['open']).reset_index()
                df = df.reset_index()
                logger.info(f"Fetched and processed {len(df)} 15-min candles for {mkt}")
                print(f"Fetched {len(df)} 15-min candles for {mkt}")
                return df, df_1h, df_4h
            except Exception as e:
                logger.error(f"Error fetching {mkt} (attempt {attempt + 1}/{retries}): {e}", exc_info=True)
                print(f"Error fetching {mkt} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue
        logger.error(f"Failed to fetch data for {mkt} after {retries} attempts")
        print(f"Failed to fetch data for {mkt} after {retries} attempts")
    logger.error(f"Failed to fetch data for {market}")
    print(f"Failed to fetch data for {market}")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

async def process_market(market):
    """Helper coroutine to fetch, process, and save data for a single market."""
    try:
        logger.info(f"Processing market {market}")
        df, df_1h, df_4h = await fetch_data(market)
        if not df.empty:
            logger.info(f"Saving data for {market}, {market}_1H, and {market}_4H")
            save_data(df, market)
            save_data(df_1h, f"{market}_1H")
            save_data(df_4h, f"{market}_4H")
        else:
            logger.warning(f"Received empty DataFrame for {market}, skipping save.")
            print(f"Received empty DataFrame for {market}, skipping save.")
    except Exception as e:
        logger.error(f"Unhandled error while processing market {market}: {e}", exc_info=True)
        print(f"Unhandled error while processing market {market}: {e}")

async def fetch_all_data(markets=TRADING_MARKETS):
    """Concurrently fetch and save data for all specified markets."""
    available_markets = await get_available_markets()
    tasks = []
    for market in markets:
        if market in available_markets or market.replace('USD', 'USDT') in available_markets:
            tasks.append(process_market(market))
        else:
            logger.warning(f"Market {market} is not available on dYdX. Skipping.")
            print(f"Market {market} is not available on dYdX. Skipping.")
    await asyncio.gather(*tasks)
    logger.info("--- Starting Post-Save Validation ---")
    for market in markets:
        df = read_data(market)
        logger.info(f"Validation: Found {len(df)} rows in DB for {market}")
        print(f"Validation: Found {len(df)} rows in DB for {market}")

def main():
    """Main function to run the asynchronous data fetching pipeline."""
    try:
        asyncio.run(fetch_all_data())
    except Exception as e:
        logger.critical(f"The data pipeline failed to run: {e}", exc_info=True)
        print(f"The data pipeline failed to run: {e}")

if __name__ == "__main__":
    main()