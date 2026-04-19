cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/app"
if [ ! -d ".venv" ]; then
    echo "Setting up..."
    python3 -m venv .venv
    .venv/bin/pip install -r ../requirements.txt
fi
echo "Starting AI Chess..."
open http://127.0.0.1:8000 2>/dev/null || xdg-open http://127.0.0.1:8000
..venv/bin/uvicorn api.main:chess --host 127.0.0.1 --port 8000
EOF