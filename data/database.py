# data/database.py
import pandas as pd
import sqlite3
import threading
from utils.logger import setup_logger

logger = setup_logger('database', 'database.log')
db_lock = threading.Lock()

def get_connection(db_name='crypto_data.db'):
    """Get SQLite connection with serialized access."""
    with db_lock:
        logger.debug(f"Opening SQLite connection to {db_name}")
        conn = sqlite3.connect(db_name, isolation_level=None)
        return conn

def read_data(market='BTC-USD', db_name='crypto_data.db'):
    """Read data from SQLite."""
    try:
        with db_lock:
            conn = get_connection(db_name)
            table_name = market.replace('-', '_') + '_data'
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
        if df.empty or 'started_at' not in df.columns:
            logger.error(f"No data or missing 'started_at' for {market}")
            return pd.DataFrame()
        logger.info(f"Read {len(df)} rows for {market}")
        return df
    except Exception as e:
        logger.error(f"Error reading {market}: {e}", exc_info=True)
        return pd.DataFrame()

def save_data(df, market='BTC-USD', db_name='crypto_data.db'):
    """Save data to SQLite, preserving signals."""
    if df.empty:
        logger.warning(f"No data to save for {market}: DataFrame is empty")
        print(f"No data to save for {market}: DataFrame is empty")
        return
    if 'started_at' not in df.columns:
        logger.warning(f"No data to save for {market}: missing 'started_at' column")
        print(f"No data to save for {market}: missing 'started_at' column")
        return
    logger.debug(f"Saving {len(df)} rows for {market}, dtypes: {df.dtypes.to_dict()}, head: {df.head(2).to_dict()}")
    try:
        with db_lock:
            conn = get_connection(db_name)
            table_name = market.replace('-', '_') + '_data'
            logger.debug(f"Checking for existing table: {table_name}")
            table_check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if table_name not in table_check['name'].values:
                logger.debug(f"Table {table_name} does not exist, creating new")
            existing_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1", conn) if table_name in table_check['name'].values else pd.DataFrame()
            if not existing_df.empty:
                signal_columns = [col for col in existing_df.columns if col in ['f7_signal', 'regime', 'hmm_signal', 'lstm_signal', 'bb_signal', 'garch_signal', 'correlation_signal', 'ensemble_signal']]
                logger.debug(f"Existing signal columns: {signal_columns}")
                if signal_columns:
                    existing_signals = pd.read_sql(f"SELECT started_at, {', '.join(signal_columns)} FROM {table_name}", conn)
                    df = df.merge(existing_signals, on='started_at', how='left', suffixes=('', '_existing'))
                    for col in signal_columns:
                        df[col] = df[f"{col}_existing"].fillna(df[col])
                        df = df.drop(f"{col}_existing", axis=1)
            logger.debug(f"Executing to_sql for {market}")
            df.to_sql(table_name, conn, if_exists='replace', index=False, dtype={'started_at': 'TEXT'})
            logger.debug(f"to_sql completed, committing")
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved {len(df)} rows for {market} to {db_name}")
            print(f"Successfully saved {len(df)} rows for {market} to {db_name}")
    except Exception as e:
        logger.error(f"FATAL ERROR saving {market} to {db_name}: {e}", exc_info=True)
        print(f"FATAL ERROR saving {market} to {db_name}: {e}")