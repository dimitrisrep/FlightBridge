@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo FlightBridge: the local Python environment was not found.
    echo Open the project instructions to create .venv first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import flightbridge, streamlit" >nul 2>&1
if errorlevel 1 (
    echo FlightBridge: the project is not installed in the local environment.
    echo Run these commands from this folder:
    echo .\.venv\Scripts\python.exe -m pip install -r requirements.lock
    echo .\.venv\Scripts\python.exe -m pip install --no-deps -e .
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run demo.py --server.address 127.0.0.1 %*
if errorlevel 1 pause
endlocal
