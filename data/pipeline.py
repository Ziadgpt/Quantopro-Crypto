# data/pipeline.py
import pandas as pd
import numpy as np
import requests
import asyncio
from datetime import datetime, timedelta, timezone

from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from config.config import INDEXER_URL, TRADING_MARKETS, CANDLE_RESOLUTION, DATA_DAYS
from data.database import save_data, read_data
from utils.logger import setup_logger
from models.feature_engineering import generate_features

logger = setup_logger('data_pipeline', 'data_pipeline.log')


async def setup_client():
    """Initialize dYdX v4 mainnet client."""
    return IndexerClient(host=f"https://{INDEXER_URL}")


async def get_available_markets():
    """Asynchronously fetch available markets from dYdX v4 mainnet."""
    try:
        client = await setup_client()
        response = await client.markets.get_perpetual_markets()
        markets = list(response.get('markets', {}).keys())
        return markets
    except Exception as e:
        logger.error(f"Error fetching markets: {e}", exc_info=True)
        return []


async def fetch_ohlcv(market='BTC-USD', timeframe=CANDLE_RESOLUTION, limit=1000):
    """Asynchronously fetch OHLCV data."""
    client = await setup_client()
    markets_to_try = [market, market.replace('USD', 'USDT')]
    to_iso, from_iso = datetime.utcnow(), datetime.utcnow() - timedelta(days=DATA_DAYS)

    for mkt in markets_to_try:
        try:
            response = await client.markets.get_perpetual_market_candles(
                market=mkt, resolution=timeframe, from_iso=from_iso.isoformat() + 'Z', to_iso=to_iso.isoformat() + 'Z',
                limit=limit
            )
            candles = response.get('candles', [])
            if not candles: continue

            df = pd.DataFrame(candles)[['startedAt', 'open', 'high', 'low', 'close', 'baseTokenVolume']]
            df.columns = ['started_at', 'open', 'high', 'low', 'close', 'base_token_volume']
            df['started_at'] = pd.to_datetime(df['started_at'])
            df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
            df.sort_values('started_at', inplace=True, ignore_index=True)
            return df
        except Exception:
            continue
    return pd.DataFrame()


async def fetch_funding_rates(market='BTC-USD'):
    """Asynchronously fetch historical funding rates using a direct HTTP request."""
    markets_to_try = [market, market.replace('USD', 'USDT')]
    from_iso_str = (datetime.utcnow() - timedelta(days=DATA_DAYS)).isoformat() + 'Z'

    for mkt in markets_to_try:
        url = f"https://{INDEXER_URL}/v4/historicalFunding/{mkt}"
        params = {'effectiveAtOrAfter': from_iso_str}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=10))
            response.raise_for_status()

            data = response.json()
            rates = data.get('historicalFunding', [])

            if not rates:
                logger.warning(f"No funding rates returned for {mkt} from direct API call.")
                continue

            df = pd.DataFrame(rates)[['effectiveAt', 'rate']]
            df.columns = ['started_at', 'funding_rate']
            df['started_at'] = pd.to_datetime(df['started_at'])
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
            df.dropna(inplace=True)
            df.sort_values('started_at', inplace=True, ignore_index=True)
            logger.info(f"✅ Successfully fetched {len(df)} funding rate entries for {mkt} via direct API call.")
            return df

        except requests.exceptions.RequestException as e:
            logger.warning(f"Direct API call failed for funding rates for {mkt}: {e}")

    logger.error(f"Failed to fetch funding rates for primary market {market} after all attempts.")
    return pd.DataFrame()


def fetch_sentiment(market='BTC-USD'):
    """Synchronous function to fetch sentiment from CoinGecko."""
    coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana', 'ADA-USD': 'cardano',
                'XRP-USD': 'ripple'}
    coin_id = coin_map.get(market, market.split('-')[0].lower())
    url = f"https://api.gecko.com/api/v3/coins/{coin_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get('sentiment_votes_up_percentage', 0.0)
    except requests.exceptions.RequestException:
        return 0.0


async def process_and_save_market(market, df_btc=None):
    """Fetches all data, merges, generates features, and saves."""
    try:
        logger.info(f"Processing market: {market}")

        ohlcv_task = fetch_ohlcv(market)
        funding_task = fetch_funding_rates(market)
        df_ohlcv, df_funding = await asyncio.gather(ohlcv_task, funding_task)

        if df_ohlcv.empty:
            logger.warning(f"No OHLCV data for {market}, skipping.")
            return

        if not df_funding.empty:
            df_15m = pd.merge_asof(df_ohlcv, df_funding, on='started_at')
            # --- CODE CLEANUP: Replaced deprecated fillna(method=...) with modern .ffill() ---
            df_15m['funding_rate'] = df_15m['funding_rate'].ffill()
        else:
            df_15m = df_ohlcv
            df_15m['funding_rate'] = 0.0

        df_15m.fillna(0, inplace=True)

        df_15m['sentiment'] = fetch_sentiment(market)
        df_15m['log_returns'] = np.log(df_15m['close'] / df_15m['close'].shift(1)).fillna(0)
        df_15m['volatility'] = df_15m['log_returns'].rolling(window=20).std().fillna(0)

        df_temp = df_15m.set_index('started_at')
        agg_rules = {col: 'last' for col in df_15m.columns if col != 'started_at'}
        agg_rules.update({'open': 'first', 'high': 'max', 'low': 'min', 'base_token_volume': 'sum'})

        df_1h = df_temp.resample('1h').agg(agg_rules).dropna(subset=['open']).reset_index()
        df_4h = df_temp.resample('4h').agg(agg_rules).dropna(subset=['open']).reset_index()

        df_15m = generate_features(df_15m, df_btc)
        df_1h = generate_features(df_1h, df_btc)
        df_4h = generate_features(df_4h, df_btc)

        save_data(df_15m, market)
        save_data(df_1h, f"{market}_1H")
        save_data(df_4h, f"{market}_4H")
        logger.info(f"Successfully processed and saved all timeframes for {market}")

    except Exception as e:
        logger.error(f"Unhandled error processing market {market}: {e}", exc_info=True)


async def run_pipeline(markets=TRADING_MARKETS):
    """Main pipeline execution function."""
    logger.info("--- Starting Data Pipeline ---")

    btc_market = "BTC-USD"
    await process_and_save_market(btc_market)
    df_btc = read_data(btc_market)
    if df_btc.empty:
        logger.error("Failed to process BTC-USD. Correlation features will be skipped.")

    altcoin_markets = [m for m in markets if m != btc_market]
    if altcoin_markets:
        tasks = [process_and_save_market(market, df_btc) for market in altcoin_markets]
        await asyncio.gather(*tasks)

    logger.info("--- Data Pipeline Finished ---")


if __name__ == "__main__":
    asyncio.run(run_pipeline())