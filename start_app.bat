@echo off
ECHO ============================================
ECHO  Starting Quantopro-Crypto Services
ECHO ============================================

:: This script assumes you have already started Redis inside WSL.
:: To start it manually, open an Ubuntu terminal and run: sudo service redis-server start

:: Step 1: Activate Virtual Environment and Start Celery Worker in a new window
ECHO [+] Launching Celery Worker...
start "Celery Worker" cmd /k "cd /d %~dp0 && .\.venv\Scripts\activate && celery -A tasks worker -P gevent -c 8 --loglevel=info"

:: Give the worker 5 seconds to initialize before starting the dashboard
timeout /t 5 >nul

:: Step 2: Activate Virtual Environment and Start Streamlit Dashboard in another new window
ECHO [+] Launching Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k "cd /d %~dp0 && .\.venv\Scripts\activate && python main.py --dashboard"

ECHO [+] Launch commands sent! Check the new windows for status.

:: This window will close in 3 seconds.
timeout /t 3 >nul