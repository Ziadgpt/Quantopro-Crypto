# data/database.py
import pandas as pd
import sqlite3
import threading
import re
from utils.logger import setup_logger

logger = setup_logger('database', 'database.log')
db_lock = threading.Lock()  # Ensures thread-safe database operations


def get_connection(db_name='crypto_data.db'):
    """Get a SQLite connection. The lock must be acquired before calling this."""
    return sqlite3.connect(db_name, timeout=10)


def save_data(df, market='BTC-USD', db_name='crypto_data.db'):
    """
    Save DataFrame to SQLite using a thread-safe upsert strategy.
    FIX: Sets dtype={'started_at': 'TEXT'} to prevent silent save failures.
    """
    if df.empty or 'started_at' not in df.columns:
        logger.warning(f"DataFrame for {market} is empty or missing 'started_at' column. Skipping save.")
        return

    table_name = market.replace('-', '_') + '_data'
    logger.info(f"Attempting to save {len(df)} rows to table '{table_name}'...")

    with db_lock:
        try:
            conn = get_connection(db_name)
            # Read existing data if table exists
            try:
                existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
            except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
                existing_df = pd.DataFrame()  # Table doesn't exist yet

            # Upsert logic: combine, remove duplicates, and sort
            combined_df = pd.concat([existing_df, df])
            combined_df.drop_duplicates(subset=['started_at'], keep='last', inplace=True)
            combined_df.sort_values('started_at', inplace=True, ignore_index=True)

            # Write data to SQL, specifying the datetime type as TEXT
            combined_df.to_sql(
                table_name,
                conn,
                if_exists='replace',
                index=False,
                dtype={'started_at': 'TEXT'}  # CRITICAL FIX for datetime handling
            )
            conn.close()
            logger.info(f"✅ Successfully saved {len(combined_df)} total rows for {market} to {table_name}")
        except Exception as e:
            logger.error(f"❌ FATAL ERROR saving {market} to {db_name}: {e}", exc_info=True)


def read_data(market='BTC-USD', db_name='crypto_data.db'):
    """Read data from SQLite. pandas' parse_dates converts the TEXT back to datetime."""
    table_name = market.replace('-', '_') + '_data'
    with db_lock:
        try:
            conn = get_connection(db_name)
            # Check if table exists before trying to read
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone() is None:
                logger.warning(f"Table '{table_name}' does not exist. Cannot read data.")
                conn.close()
                return pd.DataFrame()

            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
            conn.close()
            logger.info(f"Read {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error reading data for {market}: {e}", exc_info=True)
            return pd.DataFrame()


def get_market_list(db_name='crypto_data.db'):
    """Get a list of available base markets (e.g., 'BTC-USD') from the database tables."""
    with db_lock:
        try:
            conn = get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Extract base market names (e.g., from 'BTC_USD_data', 'BTC_USD_data_1H')
            market_set = set()
            for table in tables:
                # This regex captures the base market name like 'BTC_USD'
                match = re.match(r'^([A-Z0-9]+_[A-Z]+)_data', table)
                if match:
                    market_set.add(match.group(1).replace('_', '-'))

            return sorted(list(market_set))
        except Exception as e:
            logger.error(f"Error fetching market list from DB: {e}", exc_info=True)
            return []