@echo off
ECHO Starting background services for Crypto Bot 2.0...

:: Step 1: Start the Redis server inside WSL (no new window for this)
ECHO [+] Starting Redis Server in WSL...
wsl -e sudo service redis-server start

:: Check if Redis is running by pinging it
wsl -e redis-cli ping
IF ERRORLEVEL 1 (
    ECHO [!] Redis did not start correctly. Please check WSL.
    pause
    exit /b
)
ECHO [+] Redis is running.

:: Give Redis a moment to initialize
timeout /t 2 >nul

:: Step 2: Start the Celery Worker in a new terminal window
ECHO [+] Starting Celery Worker...
start "Celery Worker" cmd /k "cd /d %~dp0 && .\.venv\Scripts\activate && celery -A tasks worker -P gevent -c 8 --loglevel=info"

:: Give the worker a moment to initialize
timeout /t 5 >nul

:: Step 3: Start the Streamlit Dashboard in another new window
ECHO [+] Starting Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k "cd /d %~dp0 && .\.venv\Scripts\activate && python main.py --dashboard"

ECHO All services have been launched!