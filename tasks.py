# tasks.py
import asyncio
from celery import Celery
from celery.utils.log import get_task_logger

# Import your existing functions
from data.pipeline import run_pipeline
from models.hmm_model import train_hmm_model
from models.lstm_model import train_lstm_model
from models.synthesizer import train_synthesizer_model
from trading.backtesting import run_backtest
from models.tune_synthesizer import run_tuning_study # We'll create this function next

# Initialize logger for Celery tasks
logger = get_task_logger(__name__)

# Configure Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@celery_app.task(name='tasks.run_pipeline_task')
def run_pipeline_task():
    """Celery task to run the data pipeline."""
    logger.info("Celery task started: Running data pipeline...")
    try:
        asyncio.run(run_pipeline())
        result = "Pipeline completed successfully."
        logger.info(result)
        return result
    except Exception as e:
        error_message = f"Pipeline failed: {e}"
        logger.error(error_message, exc_info=True)
        return error_message

@celery_app.task(name='tasks.train_all_models_task')
def train_all_models_task():
    """Celery task to train all models."""
    logger.info("Celery task started: Training all models...")
    try:
        train_hmm_model()
        train_lstm_model()
        train_synthesizer_model()
        result = "All models trained successfully."
        logger.info(result)
        return result
    except Exception as e:
        error_message = f"Model training failed: {e}"
        logger.error(error_message, exc_info=True)
        return error_message

# --- NEW TASKS ADDED BELOW ---

@celery_app.task(name='tasks.run_tuning_task')
def run_tuning_task():
    """Celery task to run hyperparameter tuning."""
    logger.info("Celery task started: Running hyperparameter tuning...")
    try:
        # We need to refactor tune_synthesizer.py to have a callable function
        run_tuning_study()
        result = "Hyperparameter tuning completed successfully."
        logger.info(result)
        return result
    except Exception as e:
        error_message = f"Tuning failed: {e}"
        logger.error(error_message, exc_info=True)
        return error_message

@celery_app.task(name='tasks.run_backtest_task')
def run_backtest_task():
    """Celery task to run a backtest."""
    logger.info("Celery task started: Running backtest...")
    try:
        # Assuming run_backtest() returns a dict or string summary
        results = run_backtest()
        logger.info("Backtest completed successfully.")
        return results
    except Exception as e:
        error_message = f"Backtest failed: {e}"
        logger.error(error_message, exc_info=True)
        return error_message