# Production Dockerfile for vLLM inference server
#
# Build:
#   docker build -t vllm-server:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
#     vllm-server:latest

FROM vllm/vllm-openai:latest

# --- Environment ---
ENV VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
ENV VLLM_PORT=8000
ENV VLLM_HOST=0.0.0.0
ENV VLLM_GPU_MEMORY_UTILIZATION=0.90
ENV VLLM_MAX_MODEL_LEN=8192
ENV VLLM_MAX_NUM_SEQS=128
ENV VLLM_DTYPE=auto

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${VLLM_PORT}/health || exit 1

EXPOSE ${VLLM_PORT}

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "${VLLM_MODEL}", \
     "--host", "${VLLM_HOST}", \
     "--port", "${VLLM_PORT}", \
     "--gpu-memory-utilization", "${VLLM_GPU_MEMORY_UTILIZATION}", \
     "--max-model-len", "${VLLM_MAX_MODEL_LEN}", \
     "--max-num-seqs", "${VLLM_MAX_NUM_SEQS}", \
     "--dtype", "${VLLM_DTYPE}"]
