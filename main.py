# main.py
import argparse
import asyncio
import subprocess
import sys
import os

from data.pipeline import run_pipeline
from utils.logger import setup_logger
# Import the new training functions
from models.hmm_model import train_hmm_model
from models.lstm_model import train_lstm_model

logger = setup_logger('main', 'main.log')


def launch_dashboard():
    """Launches the Streamlit dashboard."""
    dashboard_path = os.path.join("ui", "dashboard.py")
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard file not found: {dashboard_path}")
        return
    logger.info("Launching Streamlit dashboard...")
    command = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    subprocess.run(command)


def main():
    """Main function to parse arguments and control the bot."""
    parser = argparse.ArgumentParser(description="Crypto Bot 2.0 - Main Controller")

    parser.add_argument('--pipeline', action='store_true', help="Run the data fetching and processing pipeline.")
    parser.add_argument('--dashboard', action='store_true', help="Launch the Streamlit dashboard.")
    parser.add_argument('--train', type=str, choices=['hmm', 'lstm', 'all'],
                        help="Train a model: 'hmm', 'lstm', or 'all'.")

    args = parser.parse_args()

    if args.pipeline:
        logger.info("Starting data pipeline from main controller.")
        asyncio.run(run_pipeline())
        logger.info("Data pipeline finished.")

    elif args.dashboard:
        launch_dashboard()

    elif args.train:
        if args.train in ['hmm', 'all']:
            logger.info("Starting HMM model training...")
            train_hmm_model()
            logger.info("HMM model training finished.")
        if args.train in ['lstm', 'all']:
            logger.info("Starting LSTM model training...")
            train_lstm_model()
            logger.info("LSTM model training finished.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()