# data/pipeline.py
import pandas as pd
import numpy as np
import requests
import asyncio
from datetime import datetime, timedelta
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from config.config import INDEXER_URL, TRADING_MARKETS, DATA_DAYS
from data.database import save_data, read_data
from utils.logger import setup_logger

logger = setup_logger('data_pipeline', 'data_pipeline.log')

async def setup_client():
    """Initialize dYdX v4 mainnet client."""
    return IndexerClient(f"https://{INDEXER_URL}")

async def get_available_markets():
    """Fetch markets from dYdX v4 mainnet."""
    try:
        client = await setup_client()
        response = await client.markets.get_perpetual_markets()
        markets = list(response['markets'].keys())
        logger.info(f"Available markets: {markets}")
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
        sentiment_score = data.get('sentiment_votes_up_percentage', 0)
        logger.info(f"Fetched sentiment for {market}: {sentiment_score}")
        print(f"Fetched sentiment for {market}: {sentiment_score}")
        return sentiment_score
    except Exception as e:
        logger.error(f"Error fetching sentiment for {market}: {e}")
        print(f"Error fetching sentiment for {market}: {e}")
        return 0

async def fetch_data(market='BTC-USD', limit=1000, retries=3, delay=2):
    """Fetch OHLCV from dYdX v4 mainnet with retries."""
    available_markets = await get_available_markets()
    if market not in available_markets:
        logger.warning(f"Market {market} not in available markets")
        print(f"Market {market} not in available markets")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    client = await setup_client()
    for attempt in range(retries):
        try:
            logger.info(f"Attempting to fetch data for {market} (attempt {attempt + 1}/{retries})")
            to_iso = datetime.utcnow()
            from_iso = to_iso - timedelta(days=DATA_DAYS)
            response = await client.markets.get_perpetual_market_candles(
                market=market,
                resolution="15MINS",
                from_iso=from_iso.isoformat() + 'Z',
                to_iso=to_iso.isoformat() + 'Z',
                limit=limit
            )
            logger.debug(f"Raw API response for {market}: {response}")
            candles = response.get('candles', [])
            if not candles:
                logger.warning(f"No candles for {market} at 15MINS")
                print(f"No candles for {market} at 15MINS")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            df = pd.DataFrame([{
                'started_at': pd.to_datetime(candle.get('startedAt', pd.NaT), errors='coerce'),
                'open': float(candle.get('open', 0)),
                'high': float(candle.get('high', 0)),
                'low': float(candle.get('low', 0)),
                'close': float(candle.get('close', 0)),
                'base_token_volume': float(candle.get('baseTokenVolume', 0))
            } for candle in candles])
            if df['started_at'].isna().all():
                logger.error(f"No valid 'startedAt' for {market}")
                print(f"No valid 'startedAt' for {market}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            df = df.sort_values('started_at').reset_index(drop=True)
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            df['volatility'] = df['log_returns'].rolling(window=20).std().fillna(0)
            df['sentiment'] = fetch_sentiment(market)
            df.set_index('started_at', inplace=True)
            df_1h = df.resample('h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'base_token_volume': 'sum', 'log_returns': 'sum', 'volatility': 'mean', 'sentiment': 'last'
            }).reset_index()
            df_4h = df.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'base_token_volume': 'sum', 'log_returns': 'sum', 'volatility': 'mean', 'sentiment': 'last'
            }).reset_index()
            df = df.reset_index()
            logger.info(f"Fetched {len(df)} 15-min candles for {market}")
            print(f"Fetched {len(df)} 15-min candles for {market}")
            return df, df_1h, df_4h
        except Exception as e:
            logger.error(f"Error fetching {market} (attempt {attempt + 1}/{retries}): {e}", exc_info=True)
            print(f"Error fetching {market} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            continue
    logger.error(f"Failed to fetch data for {market} after {retries} attempts")
    print(f"Failed to fetch data for {market} after {retries} attempts")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

async def fetch_all_data(markets=TRADING_MARKETS):
    """Fetch and save data for all markets."""
    available_markets = await get_available_markets()
    for market in markets:
        try:
            logger.info(f"Processing market {market}")
            if market in available_markets:
                df, df_1h, df_4h = await fetch_data(market)
                if not df.empty:
                    logger.info(f"Attempting to save data for {market}")
                    save_data(df, market)
                    save_data(df_1h, f"{market}_1H")
                    save_data(df_4h, f"{market}_4H")
                else:
                    logger.warning(f"Empty DataFrame for {market}, skipping save")
                    print(f"Empty DataFrame for {market}, skipping save")
            else:
                logger.warning(f"Market {market} not available")
                print(f"Market {market} not available")
        except Exception as e:
            logger.error(f"Error processing {market} in fetch_all_data: {e}", exc_info=True)
            print(f"Error processing {market}: {e}")
            continue
    # Validate saves
    for market in markets:
        df = read_data(market)
        logger.info(f"Post-save validation: {len(df)} rows in DB for {market}")
        print(f"Post-save validation: {len(df)} rows in DB for {market}")

def main():
    """Main function to run async data fetching."""
    asyncio.run(fetch_all_data())

if __name__ == "__main__":
    main()