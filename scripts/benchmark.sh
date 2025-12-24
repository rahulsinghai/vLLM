#!/usr/bin/env bash
#
# vLLM Benchmark Script
#
# Runs throughput and latency benchmarks against a running vLLM server
# using vLLM's built-in benchmarking tool and curl-based latency checks.
#
# Prerequisites:
#   - vLLM server running on $VLLM_URL (default: http://localhost:8000)
#   - Python packages: vllm, aiohttp
#
# Usage:
#   chmod +x scripts/benchmark.sh
#   ./scripts/benchmark.sh

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-10}"

echo "=============================================="
echo " vLLM Benchmark Suite"
echo "=============================================="
echo " Server:        $VLLM_URL"
echo " Model:         $MODEL"
echo " Num prompts:   $NUM_PROMPTS"
echo " Request rate:  $REQUEST_RATE req/s"
echo "=============================================="

# --- 1. Health Check ---
echo ""
echo "[1/4] Health check..."
if curl -sf "$VLLM_URL/health" > /dev/null; then
    echo "  ✓ Server is healthy"
else
    echo "  ✗ Server is not responding at $VLLM_URL"
    exit 1
fi

# --- 2. Single Request Latency ---
echo ""
echo "[2/4] Single request latency (cold)..."
time curl -s "$VLLM_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"Hello, world!\",
        \"max_tokens\": 64,
        \"temperature\": 0.0
    }" > /dev/null

# --- 3. Throughput Benchmark (ShareGPT dataset) ---
echo ""
echo "[3/4] Throughput benchmark (ShareGPT-style, $NUM_PROMPTS prompts @ $REQUEST_RATE req/s)..."
echo "  Running vLLM benchmark tool..."

python -m vllm.entrypoints.openai.api_server_benchmark \
    --backend vllm \
    --base-url "$VLLM_URL" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    2>&1 | tee /tmp/vllm-benchmark-results.txt || {
    echo "  Note: Built-in benchmark tool not available. Falling back to curl-based test."

    echo ""
    echo "  Running parallel curl benchmark ($NUM_PROMPTS requests)..."
    seq 1 "$NUM_PROMPTS" | xargs -P 20 -I{} \
        curl -s -o /dev/null -w "%{time_total}\n" \
        "$VLLM_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"Generate a short summary of request number {}\",
            \"max_tokens\": 64,
            \"temperature\": 0.0
        }" > /tmp/vllm-latencies.txt

    echo ""
    echo "  Latency stats (seconds):"
    awk '{
        sum += $1; count++; vals[count] = $1
    } END {
        asort(vals);
        printf "    Min:    %.3f\n", vals[1];
        printf "    Median: %.3f\n", vals[int(count/2)];
        printf "    p95:    %.3f\n", vals[int(count*0.95)];
        printf "    p99:    %.3f\n", vals[int(count*0.99)];
        printf "    Max:    %.3f\n", vals[count];
        printf "    Avg:    %.3f\n", sum/count;
    }' /tmp/vllm-latencies.txt
}

# --- 4. Streaming TTFT ---
echo ""
echo "[4/4] Streaming Time-to-First-Token..."
python3 -c "
import time, requests

url = '${VLLM_URL}/v1/completions'
data = {
    'model': '${MODEL}',
    'prompt': 'Explain the theory of relativity in simple terms.',
    'max_tokens': 128,
    'temperature': 0.0,
    'stream': True,
}

start = time.perf_counter()
resp = requests.post(url, json=data, stream=True)
first_chunk = True
for line in resp.iter_lines():
    if first_chunk and line:
        ttft = (time.perf_counter() - start) * 1000
        print(f'  TTFT: {ttft:.1f} ms')
        first_chunk = False
total = (time.perf_counter() - start) * 1000
print(f'  Total: {total:.1f} ms')
"

echo ""
echo "=============================================="
echo " Benchmark complete!"
echo "=============================================="
