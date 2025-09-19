# data/database.py
import pandas as pd
import sqlite3
import threading
from utils.logger import setup_logger
import re

# Setup logger for the database module
logger = setup_logger('database', 'database.log')

# A single lock to ensure all database operations are thread-safe.
# SQLite can have issues with concurrent writes from multiple threads.
db_lock = threading.Lock()


def get_connection(db_name='crypto_data.db'):
    """
    Establishes a connection to the SQLite database.
    Note: This function itself is not thread-safe. The caller must
    use the global db_lock to ensure safe database access.
    """
    logger.debug(f"Attempting to establish SQLite connection to {db_name}")
    # Set a timeout to prevent indefinite waits if the DB is locked by another process.
    conn = sqlite3.connect(db_name, timeout=10)
    return conn


def get_market_list(db_name='crypto_data.db'):
    """
    Scans the database and returns a list of available markets
    by looking for tables with the '_data' suffix.
    """
    market_list = []
    logger.info("Fetching list of available markets from the database.")
    try:
        # Acquire lock to safely interact with the database.
        with db_lock:
            conn = get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()

        # Regex to identify primary market data tables (e.g., BTC_USD_data)
        # and format them into the standard market name (e.g., BTC-USD).
        table_pattern = re.compile(r'^([A-Z0-9]+_[A-Z]+)_data$')
        for table in tables:
            match = table_pattern.match(table[0])
            if match:
                market_list.append(match.group(1).replace('_', '-'))
        logger.info(f"Found {len(market_list)} markets in the database.")
    except Exception as e:
        logger.error(f"Error fetching market list from DB: {e}", exc_info=True)
    return market_list


def read_data(market='BTC-USD', db_name='crypto_data.db'):
    """
    Reads all data for a specific market from its table in the SQLite database.
    The 'started_at' column is automatically parsed into datetime objects.
    """
    table_name = market.replace('-', '_') + '_data'
    logger.info(f"Reading data for market '{market}' from table '{table_name}'.")
    try:
        # Acquire lock for the duration of the read operation.
        with db_lock:
            conn = get_connection(db_name)
            cursor = conn.cursor()
            # Check if the table exists before attempting to read from it.
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone() is None:
                logger.warning(f"Table '{table_name}' does not exist. Cannot read data.")
                conn.close()
                return pd.DataFrame()

            # Use pandas read_sql with parse_dates for automatic datetime conversion.
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
            conn.close()

        if df.empty:
            logger.warning(f"No data was read from table '{table_name}'. The table might be empty.")
            return pd.DataFrame()

        logger.info(f"Successfully read {len(df)} rows for {market}.")
        return df
    except Exception as e:
        logger.error(f"Error reading data for {market}: {e}", exc_info=True)
        return pd.DataFrame()


def save_data(df, market='BTC-USD', db_name='crypto_data.db'):
    """
    Saves a DataFrame to the database using a safe "upsert" strategy.
    It combines the new data with existing data, removes duplicates, and
    replaces the table, ensuring no historical data is lost.
    """
    if df.empty or 'started_at' not in df.columns:
        logger.warning(f"DataFrame for {market} is empty or missing 'started_at' column. Skipping save.")
        return

    table_name = market.replace('-', '_') + '_data'
    logger.info(f"Starting save operation for {len(df)} new/updated rows for market '{market}'.")

    try:
        # Acquire lock to prevent race conditions during the read-modify-write operation.
        with db_lock:
            conn = get_connection(db_name)

            # 1. Read existing data from the database.
            try:
                existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['started_at'])
                logger.debug(f"Found {len(existing_df)} existing rows in table '{table_name}'.")
            except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
                # This is not an error; it just means the table doesn't exist yet.
                existing_df = pd.DataFrame()
                logger.info(f"Table '{table_name}' does not exist. A new one will be created.")

            # 2. Combine old and new data.
            combined_df = pd.concat([existing_df, df])

            # 3. Drop duplicates based on the timestamp, keeping the newest entry.
            combined_df.drop_duplicates(subset=['started_at'], keep='last', inplace=True)

            # 4. Sort by timestamp to maintain chronological order in the table.
            combined_df.sort_values('started_at', inplace=True, ignore_index=True)

            # 5. Write the consolidated data back to the table, replacing the old content.
            # Explicitly set dtype for 'started_at' to TEXT for cross-platform consistency.
            combined_df.to_sql(table_name, conn, if_exists='replace', index=False, dtype={'started_at': 'TEXT'})

            conn.commit()
            conn.close()
            logger.info(f"Successfully saved {len(combined_df)} total rows for '{market}' to table '{table_name}'.")

    except Exception as e:
        logger.error(f"A fatal error occurred while saving data for {market}: {e}", exc_info=True)