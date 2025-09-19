# data/pipeline.py
import pandas as pd, numpy as np, requests, asyncio
from datetime import datetime, timedelta
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from config.config import INDEXER_URL, TRADING_MARKETS, CANDLE_RESOLUTION
from data.database import save_data, read_data
from utils.logger import setup_logger
from models.feature_engineering import generate_features
from models.predictor import predict_hmm_regime, predict_lstm_forecast

logger = setup_logger('data_pipeline', 'data_pipeline.log')


async def setup_client():
    return IndexerClient(host=f"https://{INDEXER_URL}")


async def get_available_markets():
    """Asynchronously fetch available markets from dYdX v4 mainnet."""
    try:
        client = await setup_client()
        response = await client.markets.get_perpetual_markets()

        # --- FIX: Handle both object and dict response types from the client ---
        try:
            # Assumes the response is an object with a .data attribute
            markets = list(response.data['markets'].keys())
        except AttributeError:
            # Assumes the response is a raw dictionary
            markets = list(response['markets'].keys())
        # --- END FIX ---

        logger.info(f"Successfully fetched {len(markets)} available markets.")
        return markets
    except Exception as e:
        logger.error(f"Error fetching markets: {e}", exc_info=True)
        return []


async def fetch_ohlcv(market='BTC-USD', timeframe=CANDLE_RESOLUTION, total_candles=3000):
    # This function and the rest of the file remain the same...
    # (The rest of the file is omitted for brevity but should remain as it was)
    client, markets_to_try = await setup_client(), [market, market.replace('USD', 'USDT')]
    for mkt in markets_to_try:
        all_candles_list, to_iso_str = [], datetime.utcnow().isoformat() + "Z"
        try:
            while len(all_candles_list) < total_candles:
                limit = min(total_candles - len(all_candles_list), 1000)
                response = await client.markets.get_perpetual_market_candles(market=mkt, resolution=timeframe,
                                                                             to_iso=to_iso_str, limit=limit)
                candles = response.get('candles', [])
                if not candles: break
                all_candles_list.extend(candles)
                to_iso_str = min(c['startedAt'] for c in candles)
            if all_candles_list:
                df = pd.DataFrame(all_candles_list)[['startedAt', 'open', 'high', 'low', 'close', 'baseTokenVolume']]
                df.columns = ['started_at', 'open', 'high', 'low', 'close', 'base_token_volume']
                df['started_at'] = pd.to_datetime(df['started_at'])
                df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce')
                df.drop_duplicates(subset=['started_at'], inplace=True);
                df.sort_values('started_at', inplace=True, ignore_index=True)
                logger.info(f"✅ Fetched {len(df)} total candles for {mkt}")
                return df
        except Exception as e:
            logger.warning(f"Could not fetch OHLCV for {mkt}: {e}")
    return pd.DataFrame()


async def fetch_funding_rates(market='BTC-USD'):
    markets_to_try = [market, market.replace('USD', 'USDT')]
    from_iso_str = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'
    for mkt in markets_to_try:
        url = f"https://{INDEXER_URL}/v4/historicalFunding/{mkt}";
        params = {'effectiveAtOrAfter': from_iso_str}
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=10))
            response.raise_for_status()
            rates = response.json().get('historicalFunding', [])
            if not rates: continue
            df = pd.DataFrame(rates)[['effectiveAt', 'rate']]
            df.columns = ['started_at', 'funding_rate'];
            df['started_at'] = pd.to_datetime(df['started_at'])
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
            df.dropna(inplace=True);
            df.sort_values('started_at', inplace=True, ignore_index=True)
            logger.info(f"✅ Fetched {len(df)} funding rate entries for {mkt}.")
            return df
        except requests.exceptions.RequestException as e:
            logger.warning(f"API call failed for funding rates for {mkt}: {e}")
    return pd.DataFrame()


def fetch_sentiment(market='BTC-USD'):
    coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana', 'ADA-USD': 'cardano',
                'XRP-USD': 'ripple'}
    url = f"https://api.coingecko.com/api/v3/coins/{coin_map.get(market)}"
    try:
        r = requests.get(url, timeout=10);
        r.raise_for_status();
        return r.json().get('sentiment_votes_up_percentage', 0.0)
    except:
        return 0.0


async def process_and_save_market(market, df_btc=None):
    logger.info(f"Processing market: {market}")
    ohlcv_task, funding_task = fetch_ohlcv(market), fetch_funding_rates(market)
    df_ohlcv, df_funding = await asyncio.gather(ohlcv_task, funding_task)
    if df_ohlcv.empty: return

    df_15m = pd.merge_asof(df_ohlcv, df_funding, on='started_at') if not df_funding.empty else df_ohlcv
    df_15m['funding_rate'] = df_15m.get('funding_rate', 0).ffill().fillna(0)
    df_15m.fillna(0, inplace=True)
    df_15m['sentiment'] = fetch_sentiment(market)
    df_15m['log_returns'] = np.log(df_15m['close'] / df_15m['close'].shift(1)).fillna(0)
    df_15m['volatility'] = df_15m['log_returns'].rolling(window=20).std().fillna(0)

    df_temp = df_15m.set_index('started_at')
    agg_rules = {c: 'last' for c in df_15m.columns if c != 'started_at'}
    agg_rules.update({'open': 'first', 'high': 'max', 'low': 'min', 'base_token_volume': 'sum'})
    df_1h = df_temp.resample('1h').agg(agg_rules).dropna(subset=['open']).reset_index()
    df_4h = df_temp.resample('4h').agg(agg_rules).dropna(subset=['open']).reset_index()

    df_15m = predict_lstm_forecast(predict_hmm_regime(generate_features(df_15m, df_btc)))
    df_1h = predict_lstm_forecast(predict_hmm_regime(generate_features(df_1h, df_btc)))
    df_4h = predict_lstm_forecast(predict_hmm_regime(generate_features(df_4h, df_btc)))

    save_data(df_15m, market);
    save_data(df_1h, f"{market}_1H");
    save_data(df_4h, f"{market}_4H")
    logger.info(f"Successfully processed and saved all timeframes for {market}")


async def run_pipeline(markets=TRADING_MARKETS):
    logger.info("--- Starting Prediction Pipeline ---")
    available_markets = await get_available_markets()

    # Ensure we only process markets that are available
    markets_to_process = [m for m in markets if m in available_markets]

    # Process BTC first if it's in the list
    if "BTC-USD" in markets_to_process:
        await process_and_save_market("BTC-USD")
        df_btc = read_data("BTC-USD")
    else:
        df_btc = None  # No BTC data to correlate against

    # Process altcoins concurrently
    altcoins = [m for m in markets_to_process if m != "BTC-USD"]
    if altcoins:
        await asyncio.gather(*[process_and_save_market(m, df_btc) for m in altcoins])

    logger.info("--- Prediction Pipeline Finished ---")


if __name__ == "__main__":
    asyncio.run(run_pipeline())