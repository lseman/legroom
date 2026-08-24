"""Marker resolution — parses and resolves CCR compression markers."""

from __future__ import annotations

import re
from typing import Any

_CCR_PATTERNS = [
    # Pattern 1: [N items compressed. hash=abc123]
    re.compile(r"\[(\d+) items? compressed\. hash=([a-f0-9]+)\]"),
    # Pattern 2: <<ccr:abc123>>
    re.compile(r"<<ccr:([a-f0-9]+)>>"),
    # Pattern 3: read-lifecycle marker
    re.compile(r"Retrieve original: hash=([a-f0-9]+)"),
]


def parse_markers(text: str) -> list[str]:
    """Extract CCR hash keys from text."""
    hashes = []
    for pattern in _CCR_PATTERNS:
        for match in pattern.finditer(text):
            hash_key = match.group(1)
            if hash_key:
                hashes.append(hash_key)
    return list(set(hashes))


def resolve_marker(
    hash_key: str,
    store: Any,
) -> str | None:
    """Resolve a CCR hash key to original content."""
    return store.retrieve(hash_key)


def create_resolution_prompt(hashes: list[str]) -> str:
    """Create a prompt explaining how to retrieve compressed content."""
    hash_list = ", ".join(f"`{h}`" for h in hashes[:10])
    more = "" if len(hashes) <= 10 else f" (and {len(hashes) - 10} more)"

    prompt = f"""Some tool outputs have been compressed to reduce context size. If you need
the full uncompressed data, you can retrieve it using the `ccr_retrieve` tool.

**How to retrieve:**
- Call `ccr_retrieve(hash="<hash>")` to get the full original content back

**Available hashes:** {hash_list}{more}

Look for markers like `[N items compressed. hash=abc123]` or `<<ccr:abc123>>`
in tool results to find the hash for each compressed output.
"""
    return prompt
