#!/bin/bash
# Start MemoOS: the memory layer, its dashboard, and the model that
# distils terminal sessions into memories.
#
# Run it from anywhere, with or without the venv activated:
#
#     ./start_demo.sh
#
set -e

# Everything below is relative to the project, not to wherever you
# happened to be standing when you ran this.
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Starting MemoOS..."

# 0. Find an interpreter that has the dependencies.
#
#    Preferring ./venv over PATH is what lets this run from a plain
#    shell. `source venv/bin/activate` only edits PATH for the shell that
#    runs it, so a script invoked from a fresh terminal would otherwise
#    reach a system python with no uvicorn and fail three steps later,
#    at the health check, where the cause is no longer obvious.
if [ -x "venv/bin/python" ]; then
  PYTHON="$PWD/venv/bin/python"
  echo "Using project venv."
else
  PYTHON="$(command -v python3 || command -v python || true)"
  [ -n "$PYTHON" ] || { echo "ERROR: no python found on PATH."; exit 1; }
  echo "No ./venv — using $PYTHON."
fi

# Fail here, with the fix, rather than at the health check with a log path.
if ! "$PYTHON" -c "import uvicorn, fastapi" 2>/dev/null; then
  echo "ERROR: uvicorn/fastapi are not installed for $PYTHON."
  echo "       Run: $PYTHON -m pip install -r requirements.txt"
  exit 1
fi

# 1. Ollama — needed for extraction (distilling) and embeddings.
if ! command -v ollama > /dev/null; then
  echo "ERROR: ollama is not installed. See https://ollama.com/download"
  exit 1
fi
if ! pgrep -x "ollama" > /dev/null; then
  echo "Starting Ollama..."
  ollama serve > /tmp/ollama.log 2>&1 &
  sleep 2
else
  echo "Ollama already running."
fi

# 2. Confirm the extraction model is present. This is the only model
#    MemoOS still uses — there is no chat model any more.
EXTRACT_MODEL="${MEMOOS_EXTRACT_MODEL:-mistral:latest}"
if ! ollama list | grep -q "${EXTRACT_MODEL%%:*}"; then
  echo "ERROR: extraction model '$EXTRACT_MODEL' not found."
  echo "       Run: ollama pull $EXTRACT_MODEL"
  exit 1
fi
echo "Extraction model $EXTRACT_MODEL is available."

# 3. Start the API, which also serves the dashboard.
echo "Starting API server..."
"$PYTHON" -m uvicorn api:app > /tmp/memoos_api.log 2>&1 &
API_PID=$!

# 4. Health check. Uvicorn does not always finish binding within a fixed
#    pause, and a single early probe reports a failure for a server that
#    is merely still starting. Poll instead, and give up only once the
#    process itself is gone or the wait has genuinely run long.
for _ in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
    HEALTHY=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if [ -n "$HEALTHY" ]; then
  echo "API server is healthy."
else
  echo "ERROR: API server did not start correctly. Check /tmp/memoos_api.log"
  kill "$API_PID" 2>/dev/null
  exit 1
fi

echo ""
echo "Dashboard: http://127.0.0.1:8000/"
open http://127.0.0.1:8000/

echo ""
echo "To attach memory to your terminal:  $PYTHON memoos_cli.py install"
echo "To stop everything later:           kill $API_PID && pkill ollama"
