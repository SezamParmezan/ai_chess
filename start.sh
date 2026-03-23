#!/bin/bash
cd "$(dirname "$0")/app"

if [ ! -d ".venv" ]; then
    echo "Setting up for the first time..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "Starting AI Chess..."
open http://127.0.0.1:8000 2>/dev/null || xdg-open http://127.0.0.1:8000
uvicorn api.main:chess --host 127.0.0.1 --port 8000