"""CLI interface for legroom — compression proxy and dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .provider_cache import Backend, CacheMode, CachePricing


def compress_stdin(
    input_file: str = "-",
    model: str = "gpt-4o",
    output_file: str = "-",
    protect_recent: int = 0,
    no_optimize: bool = False,
    no_ccr: bool = False,
) -> None:
    """Compress messages from stdin (or file) and output result."""
    from ..runtime.compress import compress
    from ..runtime.config import CompressConfig

    input_text = sys.stdin.read() if input_file == "-" else Path(input_file).read_text()
    try:
        messages = json.loads(input_text)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    config = CompressConfig(
        optimize=not no_optimize,
        ccr_enabled=not no_ccr,
        protect_recent=protect_recent,
    )

    result = compress(messages, model=model, config=config)

    output = {
        "messages": result.messages,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "transforms_applied": result.transforms_applied,
        "warnings": result.warnings,
    }

    output_text = json.dumps(output, indent=2) + "\n"
    if output_file == "-":
        sys.stdout.write(output_text)
    else:
        Path(output_file).write_text(output_text)


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 8888,
    target: str = "https://api.openai.com/v1/chat/completions",
    api_key: str | None = None,
    compress_context: bool = True,
    mode: str = "token",
    provider_cache_mode: CacheMode = "off",
    provider_cache_key: str | None = None,
    provider_cache_ttl: str | None = None,
    backend: Backend = "openai",
    shadow_mode: bool = False,
    uncached_input_price: float = 0.0,
    cache_write_price: float = 0.0,
    cache_read_price: float = 0.0,
) -> None:
    """Start the FastAPI proxy server with dashboard."""
    import uvicorn

    from ..proxy.proxy_server import LegroomProxy

    proxy = LegroomProxy(
        target_url=target,
        api_key=api_key,
        compress_context=compress_context,
        mode=mode,
        provider_cache_mode=provider_cache_mode,
        provider_cache_key=provider_cache_key,
        provider_cache_ttl=provider_cache_ttl,
        backend=backend,
        shadow_mode=shadow_mode,
        cache_pricing=CachePricing(
            uncached_input=uncached_input_price,
            cache_write=cache_write_price,
            cache_read=cache_read_price,
        ),
    )

    key_display = "(set via --api-key or OPENAI_API_KEY)" if proxy.api_key else "(not set — requests will fail)"
    print(f"Starting Legroom proxy on http://{host}:{port}")
    print(f"Dashboard: http://{host}:{port}/")
    print(f"Forwarding to: {target}")
    print(f"API key: {key_display}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        proxy.app,
        host=host,
        port=port,
        log_level="info",
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Legroom — Context compression proxy",
        prog="legroom",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command (default)
    compress_parser = subparsers.add_parser("compress", help="Compress messages from stdin")
    compress_parser.add_argument("input", nargs="?", default="-", help="Input file (default: stdin)")
    compress_parser.add_argument("-o", "--output", default="-", help="Output file (default: stdout)")
    compress_parser.add_argument("--model", default="gpt-4o", help="Model for token counting")
    compress_parser.add_argument("--protect-recent", type=int, default=0, help="Number of recent messages to protect")
    compress_parser.add_argument("--no-optimize", action="store_true", help="Disable compression")
    compress_parser.add_argument("--no-ccr", action="store_true", help="Disable CCR injection")

    # Proxy command
    proxy_parser = subparsers.add_parser("proxy", help="Start the FastAPI proxy server with dashboard")
    proxy_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    proxy_parser.add_argument("--port", type=int, default=8888, help="Port to bind (default: 8888)")
    proxy_parser.add_argument("--target", default=None, help="Target LLM API URL (default: http://127.0.0.1:8080/v1/chat/completions or env LEGROOM_TARGET_URL)")
    proxy_parser.add_argument("--api-key", default=None, help="API key for target service (or use OPENAI_API_KEY env var)")
    proxy_parser.add_argument("--no-compress", action="store_true", help="Disable context compression")
    proxy_parser.add_argument(
        "--mode", choices=("token", "cache"), default="token",
        help="Compression mode: token rewrites history; cache freezes prior items",
    )
    proxy_parser.add_argument(
        "--uncached-input-price", type=float, default=0.0, help="USD per million tokens"
    )
    proxy_parser.add_argument(
        "--cache-write-price", type=float, default=0.0, help="USD per million tokens"
    )
    proxy_parser.add_argument(
        "--cache-read-price", type=float, default=0.0, help="USD per million tokens"
    )
    proxy_parser.add_argument(
        "--backend",
        choices=("openai", "llama_cpp"),
        default="openai",
        help=(
            "Target inference backend. 'llama_cpp' targets a llama.cpp server: "
            "provider-cache controls switch to id_slot/cache_prompt, KV-cache "
            "prefix-pointer rewriting is disabled (it would break the byte-"
            "identical prefix match llama.cpp's real KV cache needs), and "
            "cache alignment (stripping volatile UUIDs/timestamps) turns on "
            "by default to keep prompt prefixes stable across turns."
        ),
    )
    proxy_parser.add_argument(
        "--provider-cache",
        choices=("off", "implicit", "explicit"),
        default="off",
        help="Add provider prompt-cache controls without overriding caller fields",
    )
    proxy_parser.add_argument("--prompt-cache-key", default=None)
    proxy_parser.add_argument(
        "--prompt-cache-ttl",
        choices=("24h",),
        default=None,
        help="OpenAI extended prompt-cache retention (openai backend only)",
    )
    proxy_parser.add_argument(
        "--shadow-mode",
        action="store_true",
        help="Measure compression and quality without mutating outbound context",
    )

    args = parser.parse_args()

    if args.command == "proxy" or (not args.command and "--host" in sys.argv):
        run_proxy(
            host=args.host,
            port=args.port,
            target=args.target,
            api_key=args.api_key,
            compress_context=not getattr(args, "no_compress", False),
            mode=args.mode,
            provider_cache_mode=args.provider_cache,
            provider_cache_key=args.prompt_cache_key,
            provider_cache_ttl=args.prompt_cache_ttl,
            backend=args.backend,
            shadow_mode=args.shadow_mode,
            uncached_input_price=args.uncached_input_price,
            cache_write_price=args.cache_write_price,
            cache_read_price=args.cache_read_price,
        )
    else:
        # Default: compress mode (for backwards compatibility)
        if not args.command:
            # No subcommand given, try to detect if this is proxy mode
            if any(arg.startswith("--") for arg in sys.argv[1:]):
                # Has flags but no command — might be proxy mode
                args.command = "proxy"
            else:
                args.command = "compress"

        if args.command == "compress":
            compress_stdin(
                input_file=args.input,
                model=args.model,
                output_file=args.output,
                protect_recent=args.protect_recent,
                no_optimize=args.no_optimize,
                no_ccr=args.no_ccr,
            )


if __name__ == "__main__":
    main()
