#!/usr/bin/env python3
"""
Offline batch inference using vLLM's Python engine (no server needed).

This is the fastest way to process large datasets — vLLM handles continuous
batching and PagedAttention automatically under the hood.

Prerequisites:
    pip install vllm

Usage:
    python batch_generate.py
"""

import time
from vllm import LLM, SamplingParams


MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

PROMPTS = [
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a Python decorator that retries a function up to 3 times on exception.",
    "Translate to SQL: Find all users who signed up in the last 30 days and made at least 2 purchases.",
    "Summarize the CAP theorem in 3 bullet points.",
    "Write a haiku about distributed systems.",
    "What are the pros and cons of microservices vs monolith architectures?",
    "Explain the concept of attention in transformers to a 10-year-old.",
    "Generate a Dockerfile for a Python FastAPI application.",
]

SAMPLING_PARAMS = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=256,
    stop=["\n\n\n"],  # stop on triple newline
)


def main():
    print(f"Loading model: {MODEL}")
    print(f"Prompts to process: {len(PROMPTS)}\n")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
        # tensor_parallel_size=1,  # increase for multi-GPU
    )

    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - start

    total_tokens = 0
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        total_tokens += num_tokens

        print(f"{'='*70}")
        print(f"Prompt {i+1}: {output.prompt[:80]}...")
        print(f"Tokens: {num_tokens}")
        print(f"{'─'*70}")
        print(gen_text.strip())
        print()

    print(f"{'='*70}")
    print(f"Total prompts:  {len(PROMPTS)}")
    print(f"Total tokens:   {total_tokens}")
    print(f"Wall time:      {elapsed:.2f}s")
    print(f"Throughput:     {total_tokens / elapsed:.1f} tokens/sec")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
