#!/usr/bin/env python3
"""
Chat completion example using the OpenAI Python SDK against a vLLM server.

Prerequisites:
    pip install openai
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000

Usage:
    python chat_completion.py
"""

import os

from openai import OpenAI

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "not-needed")
MODEL = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def list_models():
    """List all models available on the server."""
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(f"  - {m.id}")
    return models


def chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 512):
    """Send a chat completion request."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def main():
    list_models()
    print()

    # Single-turn
    answer = chat([
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "What is PagedAttention and why does it matter for LLM serving?"},
    ])
    print(f"[Single-turn]\n{answer}\n")

    # Multi-turn conversation
    history = [
        {"role": "system", "content": "You are an expert Python developer."},
        {"role": "user", "content": "Write a Python function to compute Fibonacci numbers using memoization."},
    ]
    reply_1 = chat(history)
    print(f"[Multi-turn: Round 1]\n{reply_1}\n")

    history.append({"role": "assistant", "content": reply_1})
    history.append({"role": "user", "content": "Now convert it to an iterative version."})
    reply_2 = chat(history)
    print(f"[Multi-turn: Round 2]\n{reply_2}\n")


if __name__ == "__main__":
    main()
