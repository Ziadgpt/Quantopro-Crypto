# main.py
import argparse, asyncio, subprocess, sys, os
from data.pipeline import run_pipeline
from utils.logger import setup_logger
from models.hmm_model import train_hmm_model
from models.lstm_model import train_lstm_model
from models.synthesizer import train_synthesizer_model
from trading.backtesting import run_backtest

logger = setup_logger('main', 'main.log')

def launch_dashboard():
    dashboard_path = os.path.join("ui", "dashboard.py")
    command = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Crypto Bot 2.0 - Main Controller")
    parser.add_argument('--pipeline', action='store_true', help="Run the full data pipeline.")
    parser.add_argument('--dashboard', action='store_true', help="Launch the dashboard.")
    parser.add_argument('--train', type=str, choices=['hmm', 'lstm', 'synthesizer', 'all'], help="Train a model.")
    parser.add_argument('--backtest', action='store_true', help="Run a backtest.")
    args = parser.parse_args()

    if args.pipeline: asyncio.run(run_pipeline())
    elif args.dashboard: launch_dashboard()
    elif args.train:
        if args.train in ['hmm', 'all']: train_hmm_model()
        if args.train in ['lstm', 'all']: train_lstm_model()
        if args.train in ['synthesizer', 'all']: train_synthesizer_model()
    elif args.backtest: run_backtest()
    else: parser.print_help()

if __name__ == "__main__": main()