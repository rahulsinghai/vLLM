#!/usr/bin/env python3
"""
Multi-LoRA adapter serving example.

Shows how to query a vLLM server that serves multiple LoRA fine-tuned
adapters on top of a single base model.

Server setup:
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --enable-lora \
        --lora-modules \
            sql-expert=./lora-adapters/sql-expert \
            summarizer=./lora-adapters/summarizer \
        --max-loras 4 \
        --max-lora-rank 64 \
        --port 8000

Usage:
    python lora_serving.py
"""

import os

from openai import OpenAI

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


def query_model(model_name: str, prompt: str) -> str:
    """Query a specific model or LoRA adapter by name."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content


def main():
    # List available models (base + LoRA adapters)
    models = client.models.list()
    print("Available models/adapters:")
    for m in models.data:
        print(f"  - {m.id}")
    print()

    # Query the base model
    print("[Base Model]")
    print(query_model(BASE_MODEL, "What is a LoRA adapter?"))
    print()

    # Query LoRA adapter: sql-expert
    print("[LoRA: sql-expert]")
    print(query_model("sql-expert", "Convert to SQL: Find the top 10 customers by total order value."))
    print()

    # Query LoRA adapter: summarizer
    print("[LoRA: summarizer]")
    print(query_model("summarizer", "Summarize: The transformer architecture uses self-attention mechanisms to process sequences in parallel, unlike RNNs which process tokens sequentially."))
    print()


if __name__ == "__main__":
    main()
