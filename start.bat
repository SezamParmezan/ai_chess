@echo off
cd /d "%~dp0app"

if not exist ".venv" (
    echo Setting up for the first time...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

echo Starting AI Chess...
start http://127.0.0.1:8000
uvicorn api.main:chess --host 127.0.0.1 --port 8000

pause