# data/database.py
import pandas as pd
import sqlite3
import threading
from utils.logger import setup_logger
import re

logger = setup_logger('database', 'database.log')
db_lock = threading.Lock()

def get_connection(db_name='crypto_data.db'):
    """Get a thread-safe SQLite connection."""
    with db_lock:
        logger.debug(f"Opening SQLite connection to {db_name}")
        conn = sqlite3.connect(db_name, timeout=10)
        return conn

def get_market_list(db_name='crypto_data.db'):
    """Get a list of available markets from the database tables."""
    market_list = []
    try:
        with db_lock:
            conn = get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            table_pattern = re.compile(r'^([A-Z0-9]+_[A-Z]+)_data$')
            for table in tables:
                match = table_pattern.match(table[0])
                if match:
                    market_list.append(match.group(1).replace('_', '-'))
    except Exception as e:
        logger.error(f"Error fetching market list from DB: {e}", exc_info=True)
    return market_list

def read_data(market='BTC-USD', db_name='crypto_data.db'):
    """Read data from SQLite and parse 'started_at' column to datetime."""
    table_name = market.replace('-', '_') + '_data'
    try:
        with db_lock:
            conn = get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone() is None:
                logger.warning(f"Table '{table_name}' does not exist. Cannot read data.")
                conn.close()
                return pd.DataFrame()
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
            conn.close()
        if df.empty:
            logger.warning(f"No data read from table '{table_name}'.")
            return pd.DataFrame()
        logger.info(f"Read {len(df)} rows for {market}")
        return df
    except Exception as e:
        logger.error(f"Error reading data for {market}: {e}", exc_info=True)
        return pd.DataFrame()

def save_data(df, market='BTC-USD', db_name='crypto_data.db'):
    """Save data to SQLite using a safe upsert strategy."""
    if df.empty or 'started_at' not in df.columns:
        logger.warning(f"DataFrame for {market} is empty or missing 'started_at'. Skipping save.")
        print(f"DataFrame for {market} is empty or missing 'started_at'. Skipping save.")
        return
    table_name = market.replace('-', '_') + '_data'
    logger.debug(f"Saving {len(df)} rows for {market}, dtypes: {df.dtypes.to_dict()}, head: {df.head(2).to_dict()}")
    try:
        with db_lock:
            conn = get_connection(db_name)
            try:
                existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
                logger.debug(f"Found {len(existing_df)} existing rows in {table_name}.")
            except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
                existing_df = pd.DataFrame()
                logger.debug(f"Table {table_name} does not exist. Creating new.")
            combined_df = pd.concat([existing_df, df])
            combined_df.drop_duplicates(subset=['started_at'], keep='last', inplace=True)
            combined_df.sort_values('started_at', inplace=True, ignore_index=True)
            logger.debug(f"Executing to_sql for {market}")
            combined_df.to_sql(table_name, conn, if_exists='replace', index=False, dtype={'started_at': 'TEXT'})
            logger.debug(f"to_sql completed, committing")
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved {len(combined_df)} total rows for {market} to {table_name}")
            print(f"Successfully saved {len(combined_df)} total rows for {market} to {table_name}")
    except Exception as e:
        logger.error(f"FATAL ERROR saving {market} to {db_name}: {e}", exc_info=True)
        print(f"FATAL ERROR saving {market} to {db_name}: {e}")