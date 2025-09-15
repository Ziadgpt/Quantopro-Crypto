# utils/logger.py
import logging
import os

def setup_logger(name, log_file):
    """Set up logger for Crypto Bot 2.0."""
    log_path = os.path.join(os.getcwd(), log_file)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(name)
    return logger