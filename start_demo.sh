#!/bin/bash
set -e

echo "Starting MemoOS demo..."

# 1. Check/start Ollama
if ! pgrep -x "ollama" > /dev/null; then
  echo "Starting Ollama..."
  ollama serve > /tmp/ollama.log 2>&1 &
  sleep 2
else
  echo "Ollama already running."
fi

# 2. Confirm the model is available
if ! ollama list | grep -q "memoos-model"; then
  echo "ERROR: memoos-model not found in Ollama. Run 'ollama create memoos-model -f Modelfile' first."
  exit 1
fi
echo "Model memoos-model is available."

# 3. Start the FastAPI server in the background
echo "Starting API server..."
uvicorn api:app > /tmp/memoos_api.log 2>&1 &
API_PID=$!
sleep 2

# 4. Health check
if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
  echo "API server is healthy."
else
  echo "ERROR: API server did not start correctly. Check /tmp/memoos_api.log"
  exit 1
fi

echo ""
echo "Everything is running. Opening the demo page..."
open test_college_site.html

echo ""
echo "To stop everything later, run: kill $API_PID && pkill ollama"
