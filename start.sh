cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/app"
if [ ! -d ".venv" ]; then
    echo "Setting up..."
    python3 -m venv .venv
    .venv/bin/pip install -r ../requirements.txt
fi

echo "Starting AI Chess..."
.venv/bin/uvicorn api.main:chess --host 127.0.0.1 --port 8000 &

until curl -s http://127.0.0.1:8000 > /dev/null 2>&1; do
    sleep 1
done

open http://127.0.0.1:8000 2>/dev/null || xdg-open http://127.0.0.1:8000
wait
EOF