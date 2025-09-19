# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from celery.result import AsyncResult

# Add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- FIX: Import the 'celery_app' instance itself, along with the tasks ---
from tasks import celery_app, run_pipeline_task, train_all_models_task, run_tuning_task, run_backtest_task
# --- END FIX ---
from data.database import read_data, get_market_list
import logging

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")


# (Helper functions for display remain the same)
def display_kpis(kpis):
    st.subheader("📈 Key Performance Indicators")
    cols = st.columns(len(kpis))
    for i, (metric, value) in enumerate(kpis.items()):
        cols[i].metric(label=metric, value=value)


def display_equity_curve(equity_df):
    st.subheader("💰 Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df['started_at'], y=equity_df['equity'], mode='lines', name='Equity'))
    fig.update_layout(title="Portfolio Value Over Time", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def display_trades_log(trades_df):
    st.subheader("📋 Trades Log")
    st.dataframe(trades_df)


# (Session state initialization remains the same)
if 'backtest_task_id' not in st.session_state:
    st.session_state.backtest_task_id = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

st.title("Crypto Bot 2.0 Dashboard")

# (Sidebar and button logic remain the same)
st.sidebar.title("🛠️ Control Panel")
st.sidebar.info("Tasks run in the background. Check your Celery worker terminal for progress.")
if st.sidebar.button("Run Data Pipeline"):
    task = run_pipeline_task.delay()
    st.sidebar.success(f"Pipeline task started! (ID: {task.id})")
if st.sidebar.button("Train All Models"):
    task = train_all_models_task.delay()
    st.sidebar.success(f"Model training started! (ID: {task.id})")
if st.sidebar.button("Tune Synthesizer Model"):
    task = run_tuning_task.delay()
    st.sidebar.success(f"Hyperparameter tuning started! (ID: {task.id})")
if st.sidebar.button("Run Backtest"):
    task = run_backtest_task.delay()
    st.session_state.backtest_task_id = task.id
    st.session_state.backtest_results = None
    st.sidebar.success(f"Backtest started! (ID: {task.id})")
    st.rerun()

# --- Main Page Content ---
markets = get_market_list()

if not markets:
    st.error("No data found in database. Run the 'Data Pipeline' from the Control Panel.")
else:
    market = st.selectbox("Select Market", sorted(markets))

    # --- Backtest Results Section ---
    if st.session_state.backtest_task_id:
        # --- FIX: Pass the celery_app instance when checking the result ---
        result = AsyncResult(st.session_state.backtest_task_id, app=celery_app)
        # --- END FIX ---
        if result.ready():
            if result.successful():
                st.session_state.backtest_results = result.result
                st.session_state.backtest_task_id = None
            else:
                st.error("Backtest task failed. Check the Celery worker log for details.")
                st.session_state.backtest_task_id = None
        else:
            st.info("🔄 Backtest in progress, please wait...")

    if st.session_state.backtest_results:
        results_data = st.session_state.backtest_results
        if "error" in results_data:
            st.error(f"Backtest Error: {results_data['error']}")
        elif "message" in results_data:
            st.warning(results_data['message'])
        else:
            display_kpis(results_data['kpis'])
            display_equity_curve(results_data['equity_curve'])
            display_trades_log(results_data['trades_log'])
            st.divider()

    # (Market data section remains the same)
    if market:
        df = read_data(market)
        if not df.empty:
            st.subheader(f"{market} Market Data")
            fig = go.Figure(data=[
                go.Candlestick(x=df['started_at'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(title=f"{market} Candlestick Chart", xaxis_rangeslider_visible=False,
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)