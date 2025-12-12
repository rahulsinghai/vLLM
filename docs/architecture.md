# vLLM Architecture Deep Dive

This document covers the internals of vLLM — how requests flow through the system, how PagedAttention manages memory, and how continuous batching maximizes GPU utilization.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "API Layer"
        FASTAPI["FastAPI Server<br/>(OpenAI-compatible)"]
        TOK["Tokenizer<br/>(HuggingFace)"]
    end

    subgraph "Scheduling Layer"
        SCHED["Scheduler"]
        WAIT["Waiting Queue"]
        RUN["Running Queue"]
        SWAP["Swapped Queue"]
    end

    subgraph "Execution Layer"
        EXEC["Model Executor"]
        TP_DRIVER["TP Driver"]
        WORKER_0["Worker 0<br/>(GPU:0)"]
        WORKER_1["Worker 1<br/>(GPU:1)"]
        WORKER_N["Worker N<br/>(GPU:N)"]
    end

    subgraph "Memory Layer"
        BM["Block Manager"]
        GPU_POOL["GPU KV Block Pool"]
        CPU_POOL["CPU KV Block Pool"]
        SWAP_MGR["Swap Manager"]
    end

    subgraph "Model Layer"
        ATTN["PagedAttention Kernels"]
        MODEL["Transformer Model<br/>(weights)"]
        SAMPLER["Sampler"]
    end

    FASTAPI --> TOK --> SCHED
    SCHED --> WAIT & RUN & SWAP
    SCHED --> EXEC
    EXEC --> TP_DRIVER
    TP_DRIVER --> WORKER_0 & WORKER_1 & WORKER_N
    WORKER_0 --> BM
    BM --> GPU_POOL & CPU_POOL
    BM --> SWAP_MGR
    WORKER_0 --> ATTN --> MODEL --> SAMPLER
```

---

## PagedAttention — The Core Innovation

Traditional attention mechanisms allocate a **contiguous** block of GPU memory for the entire KV cache of each sequence. This leads to:

- **Internal fragmentation** — allocated but unused memory within a sequence's block
- **External fragmentation** — small gaps between allocations that can't be used
- **Reservation waste** — pre-allocating for `max_seq_len` even if the sequence is short

### How PagedAttention Fixes This

PagedAttention borrows the concept of **virtual memory paging** from operating systems:

```mermaid
graph TB
    subgraph "Traditional Attention"
        SEQ1_T["Sequence 1<br/>████████░░░░░░░░<br/>(50% wasted)"]
        SEQ2_T["Sequence 2<br/>████████████░░░░<br/>(25% wasted)"]
        FRAG["Fragmented gap<br/>░░░░ unusable"]
    end

    subgraph "PagedAttention"
        direction TB
        BT["Block Table<br/>(per sequence)"]

        subgraph "GPU Block Pool"
            B0["Block 0"]
            B1["Block 1"]
            B2["Block 2"]
            B3["Block 3"]
            B4["Block 4"]
            B5["Block 5"]
        end

        BT -->|"Seq 1: [0,2,4]"| B0 & B2 & B4
        BT -->|"Seq 2: [1,3,5]"| B1 & B3 & B5
    end

    style FRAG fill:#ef4444,color:#fff
    style B0 fill:#3b82f6,color:#fff
    style B2 fill:#3b82f6,color:#fff
    style B4 fill:#3b82f6,color:#fff
    style B1 fill:#10b981,color:#fff
    style B3 fill:#10b981,color:#fff
    style B5 fill:#10b981,color:#fff
```

### Memory Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Free : Server starts
    Free --> Allocated : Scheduler assigns to sequence
    Allocated --> Active : Used in attention computation
    Active --> Allocated : Waiting for next decode step
    Allocated --> Swapped : GPU pressure → move to CPU
    Swapped --> Allocated : GPU space available → swap back
    Allocated --> Free : Sequence completes
    Swapped --> Free : Sequence preempted / cancelled
```

Key properties:
- Blocks are a fixed size (default: 16 tokens)
- Blocks are **non-contiguous** — a sequence's KV cache can be scattered across GPU memory
- **Block table** (like a page table) maps logical blocks → physical blocks
- **Copy-on-write** for beam search / parallel sampling — shared prefixes share physical blocks

---

## Continuous Batching

Traditional (static) batching waits for all sequences in a batch to finish before admitting new ones. This wastes GPU compute while short sequences wait for long ones.

### Static vs Continuous Batching

```mermaid
gantt
    title Static Batching (Wasteful)
    dateFormat X
    axisFormat %s

    section Batch 1
    Seq A (short)  : 0, 3
    Seq B (long)   : 0, 8
    GPU idle (A done, waiting for B) : 3, 8

    section Batch 2
    Seq C : 8, 12
    Seq D : 8, 14
```

```mermaid
gantt
    title Continuous Batching (vLLM)
    dateFormat X
    axisFormat %s

    section GPU Always Busy
    Seq A        : 0, 3
    Seq C (fills A's slot) : 3, 7
    Seq B        : 0, 8
    Seq D (fills B's slot) : 8, 12
    Seq E (fills C's slot) : 7, 11
```

### Scheduler State Machine

```mermaid
stateDiagram-v2
    [*] --> WAITING : New request arrives
    WAITING --> RUNNING : Scheduler admits to batch<br/>(GPU blocks allocated)
    RUNNING --> RUNNING : Decode step complete<br/>(generate next token)
    RUNNING --> SWAPPED : GPU memory pressure<br/>(KV cache → CPU)
    SWAPPED --> RUNNING : GPU space freed<br/>(CPU → GPU swap-in)
    RUNNING --> FINISHED : EOS token / max_tokens
    SWAPPED --> FINISHED : Timeout / cancelled
    WAITING --> FINISHED : Rejected (queue full)
    FINISHED --> [*]
```

### Scheduling Policy

vLLM's scheduler runs on every decode step:

1. **Try to swap in** — bring back swapped sequences if GPU blocks available
2. **Try to schedule waiting** — admit new sequences from the waiting queue
3. **Preempt if needed** — if neither works, swap out the lowest-priority running sequence

Priority is determined by arrival time (FCFS) by default, but can be customized.

---

## Tensor Parallelism

For models too large for a single GPU, vLLM splits the model across GPUs using **tensor parallelism (TP)**.

```mermaid
graph TB
    subgraph "Tensor Parallel (TP=4)"
        INPUT["Input Tokens"]
        EMB["Embedding<br/>(replicated)"]

        subgraph "Each Transformer Layer"
            direction LR
            GPU0["GPU 0<br/>Attention heads 0-7<br/>MLP slice 0"]
            GPU1["GPU 1<br/>Attention heads 8-15<br/>MLP slice 1"]
            GPU2["GPU 2<br/>Attention heads 16-23<br/>MLP slice 2"]
            GPU3["GPU 3<br/>Attention heads 24-31<br/>MLP slice 3"]
        end

        AR["All-Reduce<br/>(NCCL)"]
        LM_HEAD["LM Head<br/>(replicated)"]
        OUTPUT["Output Logits"]
    end

    INPUT --> EMB --> GPU0 & GPU1 & GPU2 & GPU3
    GPU0 & GPU1 & GPU2 & GPU3 --> AR --> LM_HEAD --> OUTPUT
```

### TP Sizing Guide

| Model Size | TP Size | GPU Type | Notes |
|---|---|---|---|
| 7B | 1 | A100 40GB / L40S | Fits on single GPU |
| 7B (quantized) | 1 | RTX 4090 24GB | AWQ/GPTQ |
| 13B | 1-2 | A100 80GB | 1 GPU if fp16 fits |
| 34B | 2 | A100 80GB | |
| 70B | 4 | A100 80GB | |
| 70B | 2 | H100 80GB | Higher bandwidth |
| 405B | 8 | H100 80GB | Full node |

---

## Prefix Caching

When many requests share the same system prompt (common in production), vLLM can **cache and reuse** the KV blocks for the shared prefix.

```mermaid
graph LR
    subgraph "Without Prefix Caching"
        R1_A["Req 1: System prompt + User A<br/>Compute ALL tokens"]
        R2_A["Req 2: System prompt + User B<br/>Compute ALL tokens AGAIN"]
    end

    subgraph "With Prefix Caching"
        CACHE["Cached KV blocks<br/>for system prompt"]
        R1_B["Req 1: Reuse cache + User A"]
        R2_B["Req 2: Reuse cache + User B"]
        CACHE --> R1_B & R2_B
    end

    style CACHE fill:#10b981,color:#fff
```

Enable with `--enable-prefix-caching`. Most impactful when:
- System prompts are long (RAG context, instructions)
- Many concurrent users share the same system prompt
- Multi-turn conversations (prior turns are the shared prefix)

---

## Speculative Decoding

Speculative decoding uses a small **draft model** to predict multiple tokens ahead, then the large **target model** verifies them in a single forward pass.

```mermaid
sequenceDiagram
    participant Draft as Draft Model (small)
    participant Target as Target Model (large)
    participant Output

    Draft->>Draft: Generate K draft tokens<br/>[t1, t2, t3, t4, t5]
    Draft->>Target: Send input + K draft tokens
    Target->>Target: Single forward pass<br/>verify all K tokens
    Target->>Output: Accept [t1, t2, t3] ✓<br/>Reject [t4, t5] ✗<br/>+ correction token t4'
    Note over Output: Net gain: 3 tokens<br/>in 1 target forward pass
```

Enable with:
```shell
vllm serve large-model \
    --speculative-model small-draft-model \
    --num-speculative-tokens 5
```

---

## Summary: Data Flow End-to-End

```mermaid
flowchart LR
    A["HTTP Request"] --> B["FastAPI"]
    B --> C["Tokenizer"]
    C --> D["Scheduler<br/>(waiting → running)"]
    D --> E["Block Manager<br/>(allocate KV blocks)"]
    E --> F["Model Executor"]
    F --> G["PagedAttention<br/>Forward Pass"]
    G --> H["Sampler<br/>(top-p, top-k, temp)"]
    H --> I{"EOS or<br/>max_tokens?"}
    I -->|No| D
    I -->|Yes| J["Detokenize"]
    J --> K["HTTP Response"]

    style A fill:#3b82f6,color:#fff
    style K fill:#10b981,color:#fff
```
