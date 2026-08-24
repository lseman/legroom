# Legroom

<p align="center">
  <img src="assets/logo.svg" alt="Legroom" width="280">
</p>

<p align="center">
  <a href="https://pypi.org/project/legroom/"><img src="https://img.shields.io/pypi/v/legroom.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/legroom/"><img src="https://img.shields.io/pypi/pyversions/legroom.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/legroom/"><img src="https://img.shields.io/pypi/dm/legroom.svg" alt="PyPI downloads"></a>
  <a href="https://github.com/seman/legroom/blob/main/LICENSE"><img src="https://img.shields.io/github/license/seman/legroom.svg" alt="License"></a>
</p>

**Context compression for LLM agents.** Reduce token usage on every turn without losing the information the model actually needs — a Python-native alternative to [headroom](https://github.com/headroomlabs-ai/headroom), built as a library first and a proxy second.

## Evaluation

Legroom ships a versioned evaluation suite built from realistic agent traces:
tool-call JSON, log dumps, repeated file reads, and search output. It compares
Legroom with identity, recent-window, and head/tail truncation baselines and
reports:

- token reduction and the aggregate quality–token Pareto frontier;
- expected-fact task success and structural preservation invariants;
- p50/p95 latency and peak traced memory;
- per-fixture, per-strategy results in Markdown or machine-readable JSON.

Run it yourself with `python benchmarks/run_benchmark.py`, or produce a stable
artifact with `python benchmarks/run_benchmark.py --json`. The suite manifest
is [`benchmarks/suite-v1.json`](benchmarks/suite-v1.json); results are measured
on the current checkout rather than copied from a separate installation.

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

Model profiles are explicit presets so they never silently overwrite caller
configuration. Set `CompressConfig(use_model_profile=True)` when you want the
selected model's preset values for `protect_recent`, compression threshold, and
adaptive-size bias.

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
| `--provider-cache` | `off` | Provider prompt-cache policy: `off`, `implicit`, or `explicit` |
| `--prompt-cache-key` | derived | Stable explicit-cache key; caller-supplied request fields take precedence |
| `--prompt-cache-ttl` | provider default | OpenAI extended prompt-cache retention (`24h`) |
| `--shadow-mode` | false | Evaluate compression and potential savings without changing outbound context |
| `--uncached-input-price` | `0` | USD per million uncached input tokens |
| `--cache-write-price` | `0` | USD per million cache-write input tokens |
| `--cache-read-price` | `0` | USD per million cache-read input tokens |

The proxy compresses `POST /v1/chat/completions` and message-shaped
`POST /v1/responses` requests. Other paths, methods, and bodies are forwarded
byte-for-byte, including binary and non-JSON payloads.

For cache-sensitive production traffic, start in shadow mode and preserve the
stable prefix while measuring actual provider cache reads:

```bash
legroom proxy --mode cache --provider-cache explicit \
  --prompt-cache-ttl 24h --shadow-mode \
  --uncached-input-price 2.50 --cache-write-price 3.00 \
  --cache-read-price 0.25
```

Pricing is deliberately supplied by the operator because provider and model
rates change independently of Legroom releases. Once the quality and savings
metrics meet your release gate, remove `--shadow-mode` to mutate outbound
requests. Legroom never replaces prompt-cache fields already supplied by the
caller.

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
| `GET /api/stats` | Compression, provider-cache, shadow, and calibration stats |
| `GET /api/history?limit=50&offset=0` | Recent requests |
| `GET /api/read-lifecycle` | Read lifecycle statistics |
| `GET /api/ccr` | CCR store statistics |
| `GET /livez` | Process liveness |
| `GET /readyz` | HTTP-client readiness |
| `GET /metrics` | Prometheus request, phase, cache-token, cost, shadow, latency, and token metrics |
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

Provider requests first pass through lossless OpenAI Chat Completions or
Responses adapters into a typed, provider-neutral conversation IR. Unknown
provider fields and opaque content blocks round-trip unchanged. Each enabled
pipeline phase then follows the same `analyze → propose → validate → apply`
contract. Results include `metadata["phase_reports"]` with phase status, token
delta, protected spans, reversibility, latency, confidence, failures, and
phase-specific metadata.

The IR assigns provenance and compression-risk labels at the provider boundary.
System/developer instructions, structured tool calls, opaque provider data, and
the current user turn are restored after every phase if a transform touches
them. A rolling calibration controller can disable phases whose validated
success or downstream-quality score falls below configured gates; per-request
quality failures roll back the whole candidate. Shadow mode exercises the same
pipeline and reports potential savings without changing the request sent
upstream.

Token counts include protocol framing, roles, tool calls, structured content,
and tool identifiers. They are still estimates: providers may use private wire
serialization and media-token accounting.

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

Fixtures live in [`benchmarks/fixtures/`](benchmarks/fixtures/) as plain `{"description", "messages"}` JSON — drop in your own traces to benchmark against your actual workload. Each run reports tokens before/after, compression ratio, latency, memory, invariants, and task-success scores per trace. The Python harness also accepts a typed callable evaluator, so repository tests, model graders, or complete agent tasks can produce executable pass/fail and score evidence instead of relying only on retained terms.

Task-success markers and suite membership live in the versioned suite manifest,
separate from trace payloads. This makes corpus changes reviewable and prevents
silent benchmark drift.

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
