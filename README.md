# Legroom

**Context compression for LLM agents.** Reduce token usage on every turn without losing the information the model actually needs — a Python-native alternative to [headroom](https://github.com/headroomlabs-ai/headroom), built as a library first and a proxy second.

## Proof

Measured with [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) against realistic agent traces (tool-call JSON, log dumps, repeated file reads, grep output) — not synthetic best cases:

| Trace | Before | After | Saved |
|---|---:|---:|---:|
| Coding-agent file reads (stale re-reads) | 548 | 412 | **24.8%** |
| Log dump (retries, repeated lines) | 1139 | 856 | **24.8%** |
| JSON tool results (paginated records) | 871 | 766 | 12.1% |
| Grep results | 457 | 425 | 7.0% |
| **Total** | **3015** | **2459** | **18.4%** |

Run it yourself: `python benchmarks/run_benchmark.py`. It auto-detects a `headroom` install for a side-by-side comparison column; without one it just reports legroom's own numbers rather than guessing.

## How it compresses

- **SmartCrusher** — JSON array deduplication and summarization, with SimHash-based near-duplicate detection to auto-size how much to keep
- **CacheAligner** — Detects UUIDs, timestamps, JWTs that bust KV cache
- **ThinkingCompactor** — Strips `<think>...</think>` reasoning blocks
- **ContentRouter** — Dispatches to the best compressor (JSON, logs, search, code)
- **Cross-Turn Dedup** — Replaces identical spans across messages with in-context pointers
- **Read Lifecycle** — Compresses stale or superseded file reads once a later edit or re-read supersedes them
- **Recursive JSON** — Finds and compresses nested JSON in payloads
- **Lossless Compaction** — ANSI stripping, run collapse, search heading compression
- **Adaptive Sizer** — Kneedle-on-bigram-coverage + real SimHash clustering auto-determines optimal compression depth
- **ML Compressor** — Kompress-v2-base ONNX model for token-level retention scoring
- **CCR (Compression Cache Retrieval)** — Reversible compression with a retrieval tool, so compressed content can be pulled back on demand instead of being lost

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

### Proxy Server

Legroom includes a FastAPI reverse proxy that compresses context on the fly and serves a live dashboard:

```bash
# Start proxy (binds safely to 127.0.0.1:8888)
# Compressed requests are forwarded to 127.0.0.1:8080 (your OpenAI-compatible server)
export OPENAI_API_KEY=sk-your-key
legroom proxy
```

Then open **http://localhost:8888/** to see the dashboard with:

- Real-time compression stats (tokens before/after, ratio)
- Request history with per-transform breakdowns
- Token savings chart (last 30 requests)
- Read lifecycle and CCR statistics
- Live WebSocket/SSE updates

#### Proxy configuration

```bash
export OPENAI_API_KEY=sk-your-key  # or use --api-key flag
legroom proxy --port 8888 --target http://127.0.0.1:8080 --mode token
```

#### Using the proxy as a drop-in replacement

Point your OpenAI client at the proxy:

```python
from openai import OpenAI

# Point to proxy at 127.0.0.1:8080 (your OpenAI-compatible server)
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="any-key")
# All requests are compressed by the proxy before being forwarded
response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

Or with `httpx`:

```python
import httpx

async with httpx.AsyncClient(base_url="http://127.0.0.1:8080/v1") as client:
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
```

#### Proxy options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind (proxy dashboard + API) |
| `--port` | `8888` | Port to bind (proxy dashboard + API) |
| `--target` | `http://127.0.0.1:8080/v1/chat/completions` | Target LLM API URL (or env `LEGROOM_TARGET_URL`) |
| `--api-key` | env var | API key (or env `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LEGROOM_API_KEY`) |
| `--no-compress` | false | Disable context compression |
| `--mode` | `token` | `token` compresses full history; `cache` freezes prior items and compresses only the live item |

The proxy compresses `POST /v1/chat/completions` and message-shaped
`POST /v1/responses` requests. Other paths, methods, and bodies are forwarded
byte-for-byte, including binary and non-JSON payloads.

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key for OpenAI (fallback for `--api-key`) |
| `ANTHROPIC_API_KEY` | — | API key for Anthropic (fallback) |
| `LEGROOM_API_KEY` | — | Generic API key (fallback) |
| `LEGROOM_TARGET_URL` | `http://127.0.0.1:8080/v1/chat/completions` | Target LLM API URL |

#### Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/stats` | Aggregate compression stats |
| `GET /api/history?limit=50&offset=0` | Recent requests |
| `GET /api/read-lifecycle` | Read lifecycle statistics |
| `GET /api/ccr` | CCR store statistics |
| `GET /livez` | Process liveness |
| `GET /readyz` | HTTP-client readiness |
| `GET /metrics` | Prometheus request, error, cache, latency, and token metrics |
| `GET /ws/events` | WebSocket live events |
| `GET /api/events` | SSE fallback live events |

#### Architecture

The proxy listens on `127.0.0.1:8888` by default and forwards compressed traffic to your OpenAI-compatible server at `127.0.0.1:8080`:

```
Client → Proxy (127.0.0.1:8888) → OpenAI Server (127.0.0.1:8080)
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

## Benchmarks

```bash
python benchmarks/run_benchmark.py           # table output
python benchmarks/run_benchmark.py --json     # machine-readable
python benchmarks/run_benchmark.py --model claude-3-5-sonnet
```

Fixtures live in [`benchmarks/fixtures/`](benchmarks/fixtures/) as plain `{"description", "messages"}` JSON — drop in your own traces to benchmark against your actual workload. Each run reports tokens before/after, compression ratio, latency, and which transforms fired per trace.

## Model Integration

Legroom supports the [Kompress-v2-base](https://huggingface.co/chopratejas/kompress-v2-base) model for ML-based token-level retention. Download the model files:

```bash
mkdir -p models/kompress-v2-base/{onnx,adapter}
wget -O models/kompress-v2-base/onnx/kompress-fp32.onnx \
    "https://huggingface.co/chopratejas/kompress-v2-base/resolve/main/onnx/kompress-fp32.onnx"
wget -O models/kompress-v2-base/tokenizer.json \
    "https://huggingface.co/chopratejas/kompress-v2-base/resolve/main/tokenizer.json"
```

Then use in code (paths default to the above if omitted):

```python
from legroom import MLTextCompressor

compressor = MLTextCompressor(
    model_path="models/kompress-v2-base/onnx/kompress-fp32.onnx",
    tokenizer_path="models/kompress-v2-base/tokenizer.json",
)
```

| Model file | Size | Description |
|------------|------|-------------|
| `onnx/kompress-fp32.onnx` | 572 MB | Kompress-v2-base ONNX model (FP32) |
| `tokenizer.json` | 3.5 MB | HuggingFace tokenizer vocab |

### Enabling it in the pipeline (opt-in)

`compress()` never uses the ML compressor unless you ask for it — it's lossy
(drops low-score tokens from plain text) and needs the optional deps and
model files above, so it stays off by default:

```python
from legroom import compress, CompressConfig

config = CompressConfig(
    ml_compress_enabled=True,
    # optional overrides — otherwise uses the paths under models/kompress-v2-base/
    ml_model_path="models/kompress-v2-base/onnx/kompress-fp32.onnx",
    ml_tokenizer_path="models/kompress-v2-base/tokenizer.json",
    retention_threshold=0.5,       # higher = keep more tokens
    min_compression_ratio=0.1,     # floor on how much must be dropped to accept
)
result = compress(messages, model="gpt-4o", config=config)
```

If `legroom[ml]` isn't installed or the model files aren't present, this
degrades gracefully: a warning is logged once at startup (or compression
silently falls back per-message) and plain lossless text compression is
used instead — it never raises or blocks the rest of the pipeline.

## License

MIT
