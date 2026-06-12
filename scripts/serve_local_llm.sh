#!/bin/bash
# Serve an open-weights model locally for the LLM advisor.
#   bash scripts/serve_local_llm.sh qwen3.5:9b-bf16
# Requires Ollama (https://ollama.com). For hybrid reasoning models (e.g.
# Qwen3.5) the no-think proxy keeps responses fast and deterministic-format;
# point OPENAI_BASE_URL at http://127.0.0.1:11435/v1. For plain instruct
# models use Ollama directly (http://127.0.0.1:11434/v1).
set -e

command -v ollama >/dev/null 2>&1 || {
    echo "ERROR: ollama is not installed (https://ollama.com)" >&2
    exit 1
}

MODEL=${1:-qwen3.5:9b-bf16}
ollama serve >/tmp/ollama_serve.log 2>&1 &

# Wait for the Ollama API instead of a blind sleep.
for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
done
curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 || {
    echo "ERROR: ollama did not come up on :11434 (see /tmp/ollama_serve.log)" >&2
    exit 1
}

ollama pull "$MODEL"

python3 "$(dirname "$0")/nothink_proxy.py" --port 11435 &
PROXY_PID=$!
sleep 1
if curl -sf http://127.0.0.1:11435/v1/models >/dev/null 2>&1; then
    echo "model $MODEL ready; proxy on :11435 (think disabled, pid $PROXY_PID), ollama on :11434"
else
    echo "ERROR: proxy on :11435 did not answer /v1/models — check that port 11435 is free" >&2
    exit 1
fi
