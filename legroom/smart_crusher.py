"""SmartCrusher — intelligent JSON compression with deduplication."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any
from collections import defaultdict

from .compressor_registry import CompressInput, CompressOutput


@dataclass
class SmartCrusherConfig:
    """Configuration for SmartCrusher."""

    max_items: int = 50
    adaptive_sizing: bool = False
    size_bias: float = 1.0
    protect_recent: int = 0
    min_group_size: int = 3


class SmartCrusher:
    """Compresses arrays of items with deduplication and summarization."""

    def __init__(self, config: SmartCrusherConfig | None = None) -> None:
        self.config = config or SmartCrusherConfig()

    def compress(self, content: str, source_hint: str = "unknown") -> CompressOutput:
        """Compress JSON content."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                data = self._crush_array(data, {})
                compressed = json.dumps(data, indent=2)
                tokens_before = len(content) // 4
                tokens_after = len(compressed) // 4
                return CompressOutput(
                    compressed=compressed,
                    original_token_count=tokens_before,
                    compressed_token_count=tokens_after,
                    strategy="smart_crusher",
                )
        except (json.JSONDecodeError, ValueError):
            pass
        # Return unchanged
        return CompressOutput(
            compressed=content,
            original_token_count=len(content) // 4,
            compressed_token_count=len(content) // 4,
            strategy="smart_crusher",
        )

    def _crush_array(self, items: list, metadata: dict) -> list:
        """Compress an array of items."""
        if len(items) <= 1:
            return items

        # Protect recent items (rolling window)
        protect_recent = metadata.get("protect_recent", self.config.protect_recent)
        recent = items[-protect_recent:] if protect_recent > 0 else []
        to_compress = items[:-protect_recent] if protect_recent > 0 else items

        # Find and group structurally similar items
        groups = self._find_structural_groups(to_compress)

        result = []
        for group in groups:
            if len(group) >= self.config.min_group_size:
                result.append(self._summarize_group(group))
            else:
                result.extend(group)

        result.extend(recent)
        return result

    def _find_structural_groups(self, items: list) -> list[list]:
        """Group items by structural fingerprint."""
        groups: dict[str, list] = defaultdict(list)
        for item in items:
            if isinstance(item, dict):
                key = self._structural_key(item)
                groups[key].append(item)
            else:
                groups[str(type(item))].append(item)
        return list(groups.values())

    def _structural_key(self, item: dict) -> str:
        """Create a structural fingerprint for an item."""
        parts = []
        for key in sorted(item.keys()):
            val = item[key]
            parts.append(f"{key}:{type(val).__name__}")
        return "|".join(parts)

    def _find_varying_fields(self, items: list) -> list[str]:
        """Find which fields vary across a group."""
        if not items or not all(isinstance(i, dict) for i in items):
            return []

        all_keys = set()
        for item in items:
            all_keys.update(item.keys())

        varying = []
        for key in all_keys:
            values = set()
            for item in items:
                if key in item:
                    val = item[key]
                    values.add(json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else str(val))
                    if len(values) > 1:
                        varying.append(key)
                        break

        return varying

    def _summarize_group(self, group: list) -> dict:
        """Create a summary of a group of similar items."""
        if not group:
            return {}

        # Find varying fields
        varying = self._find_varying_fields(group)

        # Sample the first item for fixed fields
        sample = group[0]
        summary = {
            **{k: v for k, v in sample.items() if k not in varying},
            "_compressed_summary": f"{len(group)} structurally similar items (varying: {', '.join(varying)}). Sample shown.",
            "_content": json.dumps(sample),
        }

        return summary
