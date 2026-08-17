# Legroom — Context Compression Proxy

Reduce LLM context tokens while preserving meaning. A Python-based alternative to [headroom](https://github.com/headroomlabs-ai/headroom).

## Features

- **SmartCrusher** — JSON array deduplication and summarization
- **CacheAligner** — Detects UUIDs, timestamps, JWTs that bust KV cache
- **ThinkingCompactor** — Strips `<think>...</think>` reasoning blocks
- **ContentRouter** — Dispatches to the best compressor (JSON, logs, search, code)
- **Cross-Turn Dedup** — Replaces identical spans across messages with in-context pointers
- **Recursive JSON** — Finds and compresses nested JSON in payloads
- **Lossless Compaction** — ANSI stripping, run collapse, search heading compression
- **Adaptive Sizer** — Kneedle algorithm auto-determines optimal compression depth
- **ML Compressor** — Kompress-v2-base ONNX model for token-level retention scoring
- **CCR (Compression Cache Retrieval)** — Reversible compression with retrieval tool

## Installation

```bash
pip install legroom
pip install legroom[ml]  # ML features (onnxruntime, tokenizers)
pip install legroom[dev]  # Testing
```

## Usage

```python
from legroom import compress, CompressConfig

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

result = compress(messages, model="gpt-4o")
print(f"Tokens: {result.tokens_before} -> {result.tokens_after} (saved {result.tokens_saved})")
```

### With protection for recent messages

```python
config = CompressConfig(protect_recent=2)
result = compress(messages, model="gpt-4o", config=config)
```

## CLI

```bash
echo '[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]' | legroom
```

## Architecture

```
Phase 1: OutputShaper (verbosity steering)
Phase 2: CacheAligner (volatile content detection)
Phase 2.5: CrossTurnDedup (identical span dedup)
Phase 3: CompressPhase
  ├─ Lossless compaction (ANSI, runs, headings)
  ├─ ContentRouter (SmartCrusher, LogCompressor, etc.)
  └─ Recursive JSON routing (nested JSON compression)
Phase 4: ThinkingCompactor (reasoning block removal)
Phase 5: CCR Tool Injection (retrieval tool + instructions)
```

## Testing

```bash
pytest tests/ -v
```

## Model Integration

Legroom supports the [Kompress-v2-base](https://huggingface.co/chopratejas/kompress-v2-base) model for ML-based token-level retention:

```python
from legroom import MLTextCompressor

compressor = MLTextCompressor(
    model_path="models/kompress-v2-base/onnx/kompress-fp32.onnx",
    tokenizer_path="models/kompress-v2-base/tokenizer.json",
)
```

## License

MIT
