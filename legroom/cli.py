"""CLI interface for legroom."""

from __future__ import annotations

import argparse
import json
import sys

from .compress import compress
from .config import CompressConfig


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Compress LLM context")
    parser.add_argument("input", nargs="?", default="-", help="Input file (default: stdin)")
    parser.add_argument("-o", "--output", default="-", help="Output file (default: stdout)")
    parser.add_argument("--model", default="gpt-4o", help="Model for token counting")
    parser.add_argument("--protect-recent", type=int, default=0, help="Number of recent messages to protect")
    parser.add_argument("--no-optimize", action="store_true", help="Disable compression")
    parser.add_argument("--no-ccr", action="store_true", help="Disable CCR injection")
    args = parser.parse_args()

    # Read input
    input_text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    messages = json.loads(input_text)

    # Compress
    config = CompressConfig(
        optimize=not args.no_optimize,
        ccr_enabled=not args.no_ccr,
        protect_recent=args.protect_recent,
    )

    result = compress(messages, model=args.model, config=config)

    # Output
    output = {
        "messages": result.messages,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "transforms_applied": result.transforms_applied,
        "warnings": result.warnings,
    }

    output_text = json.dumps(output, indent=2)
    if args.output == "-":
        print(output_text)
    else:
        with open(args.output, "w") as f:
            f.write(output_text)


if __name__ == "__main__":
    main()
