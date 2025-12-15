#!/usr/bin/env python3
"""
Streaming chat completion example.

Demonstrates real-time token streaming using Server-Sent Events (SSE)
through the OpenAI SDK against a vLLM server.

Prerequisites:
    pip install openai
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000

Usage:
    python stream_chat.py
    python stream_chat.py "Your custom prompt here"
"""

import os
import sys
import time

from openai import OpenAI

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


def stream_chat(prompt: str):
    """Stream a chat completion and measure time-to-first-token (TTFT)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": prompt},
    ]

    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        max_tokens=512,
        temperature=0.7,
    )

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(delta, end="", flush=True)
            token_count += 1

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else 0
    total = end - start

    print(f"\n\n{'─'*60}")
    print(f"TTFT:             {ttft*1000:.1f} ms")
    print(f"Total time:       {total*1000:.1f} ms")
    print(f"Tokens generated: {token_count}")
    print(f"Tokens/sec:       {token_count / total:.1f}")
    print(f"{'─'*60}\n")


def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Explain how a GPU executes a matrix multiplication in 5 sentences."
    stream_chat(prompt)


if __name__ == "__main__":
    main()
