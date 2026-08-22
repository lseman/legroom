"""SmartCrusher — intelligent JSON compression with field-level deduplication."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

from .adaptive_sizer import compute_optimal_k
from .compressor_registry import CompressInput, CompressOutput
from ..query_relevance import query_relevance
from ..tokenizer import count_tokens


@dataclass
class SmartCrusherConfig:
    """Configuration for SmartCrusher."""

    max_items: int = 50
    adaptive_sizing: bool = False
    size_bias: float = 1.0
    protect_recent: int = 0
    min_group_size: int = 3
    # New: field-level compression settings
    compress_varying_fields: bool = True
    max_varying_field_chars: int = 40  # Truncate varying fields beyond this
    max_varying_samples: int = 5  # Cap on deduplicated sample values kept per varying field
    # Query-aware retention: items scoring >= this against the current
    # query's terms are kept as full items instead of folded into a
    # group summary. 0.0 (default query_terms=set()) is a no-op, so this
    # is inert unless a query is actually passed in.
    query_relevance_threshold: float = 0.3


class SmartCrusher:
    """Compresses arrays of items with deduplication and field-level summarization.

    Field-level deduplication:
    - Fixed fields (same value across all items): shown once in summary
    - Varying fields: replaced with compact statistics (min/max/unique_count)
    - Large varying fields (strings > max_varying_field_chars): truncated with ellipsis
    """

    def __init__(self, config: SmartCrusherConfig | None = None) -> None:
        self.config = config or SmartCrusherConfig()

    def compress(
        self,
        content: str,
        source_hint: str = "unknown",
        model: str = "gpt-4o",
        query_terms: set[str] | None = None,
    ) -> CompressOutput:
        """Compress JSON content.

        ``query_terms`` (from :func:`legroom.query_relevance.latest_query_terms`)
        biases retention toward items relevant to the current turn's question —
        pass None or an empty set to get the original query-agnostic behavior.
        """
        try:
            data = json.loads(content)
            if isinstance(data, list):
                data = self._crush_array(data, {}, query_terms or set())
                compressed = json.dumps(data, separators=(",", ":"))
                tokens_before = count_tokens(content, model)
                tokens_after = count_tokens(compressed, model)
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
            original_token_count=count_tokens(content, model),
            compressed_token_count=count_tokens(content, model),
            strategy="smart_crusher",
        )

    def _crush_array(self, items: list, metadata: dict, query_terms: set[str] | None = None) -> list:
        """Compress an array of items."""
        if len(items) <= 1:
            return items

        # Protect recent items (rolling window)
        protect_recent = metadata.get("protect_recent", self.config.protect_recent)
        recent = items[-protect_recent:] if protect_recent > 0 else []
        to_compress = items[:-protect_recent] if protect_recent > 0 else items

        # Query-aware retention: pull out items relevant to the current
        # query so they survive as full items instead of being folded
        # into a group summary. No-op when query_terms is empty.
        relevant: list = []
        if query_terms:
            still_to_compress = []
            for item in to_compress:
                item_text = item if isinstance(item, str) else json.dumps(item)
                if query_relevance(item_text, query_terms) >= self.config.query_relevance_threshold:
                    relevant.append(item)
                else:
                    still_to_compress.append(item)
            to_compress = still_to_compress

        # Find and group structurally similar items
        groups = self._find_structural_groups(to_compress)

        result = []
        for group in groups:
            if len(group) < self.config.min_group_size:
                result.extend(group)
                continue

            if self.config.adaptive_sizing:
                # Let compute_optimal_k gauge how many items in this group
                # are genuinely distinct (via SimHash near-dup detection +
                # Kneedle knee-finding). A summary is only worth its fixed
                # overhead when the group collapses to well under its size;
                # otherwise the items are diverse enough that summarizing
                # would just throw away information for little savings.
                serialized = [json.dumps(item, sort_keys=True) for item in group]
                k = compute_optimal_k(serialized, bias=self.config.size_bias)
                if k >= len(group) * 0.75:
                    result.extend(group)
                    continue

            result.append(self._summarize_group(group))

        result.extend(relevant)
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

    def _compute_field_stats(self, group: list, field_name: str) -> dict[str, Any]:
        """Compute statistics for a varying field across a group.

        Returns a compact representation instead of the full data:
        - For numbers: {"type": "number", "min": ..., "max": ..., "unique": N}
        - For strings: {"type": "string", "sample": ..., "unique": N, "avg_len": N}
        - For bools: {"type": "bool", "true_count": N, "false_count": N}
        - For nested objects/arrays: {"type": "complex", "unique": N}
        """
        values = []
        for item in group:
            if isinstance(item, dict) and field_name in item:
                values.append(item[field_name])

        if not values:
            return {"type": "missing"}

        unique_values: set[str] = set()
        for v in values:
            unique_values.add(json.dumps(v, sort_keys=True))

        n = len(values)
        unique_count = len(unique_values)

        # Number stats
        numbers = [v for v in values if isinstance(v, (int, float))]
        if numbers:
            return {
                "type": "number",
                "count": n,
                "unique": unique_count,
                "min": min(numbers),
                "max": max(numbers),
                "avg": round(sum(numbers) / n, 3),
            }

        # Bool stats
        bools = [v for v in values if isinstance(v, bool)]
        if bools:
            return {
                "type": "bool",
                "count": n,
                "true_count": sum(1 for v in bools if v),
                "false_count": n - sum(1 for v in bools if v),
                "unique": unique_count,
            }

        # String stats
        strings = [v for v in values if isinstance(v, str)]
        if strings:
            avg_len = sum(len(s) for s in strings) / len(strings)
            # Sample representative strings
            samples = []
            seen = set()
            for s in strings:
                if s not in seen and len(samples) < 2:
                    seen.add(s)
                    samples.append(s)
            return {
                "type": "string",
                "count": n,
                "unique": unique_count,
                "avg_len": round(avg_len),
                "sample": samples[0] if samples else None,
            }

        # Complex (dict/list)
        return {
            "type": "complex",
            "count": n,
            "unique": unique_count,
            "sample_keys": list(values[0].keys()) if isinstance(values[0], dict) else None,
        }

    def _compress_field_value(self, value: Any, field_name: str, group_size: int) -> Any:
        """Compress a single varying field value.

        Truncates long strings and replaces large objects with compact repr.
        """
        if isinstance(value, str):
            max_chars = self.config.max_varying_field_chars
            if len(value) > max_chars:
                return f"{value[:max_chars // 2]}...{value[-max_chars // 3:]}"
            return value
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (dict, list)):
            # Large nested object — replace with summary
            return {
                "_truncated": True,
                "type": "dict" if isinstance(value, dict) else "list",
                "keys_count": len(value) if isinstance(value, dict) else len(value),
                "summary": f"{group_size} items with {len(value)} keys/elements",
            }
        elif value is None:
            return None
        else:
            return str(value)

    def _summarize_group(self, group: list) -> dict:
        """Create a summary of a group of similar items with field-level dedup.

        Fixed fields are preserved as-is. Varying fields are replaced with
        compact statistics and truncated samples, dramatically reducing output size.
        """
        if not group:
            return {}

        varying = self._find_varying_fields(group)
        fixed_fields = {k: v for k, v in group[0].items() if k not in varying}

        summary: dict[str, Any] = dict(fixed_fields)

        # Add varying field summaries
        field_stats = {}
        compressed_varying = {}
        for vf in varying:
            # Compute stats for the field across the group
            stats = self._compute_field_stats(group, vf)
            field_stats[vf] = stats

            # Sample deduplicated values instead of every raw occurrence —
            # low-cardinality fields (e.g. status) don't need N repeats of
            # the same value, and _field_stats already covers the shape of
            # high-cardinality ones (e.g. ids, timestamps).
            seen: dict[str, Any] = {}
            for item in group:
                if isinstance(item, dict) and vf in item:
                    raw = item[vf]
                    dedup_key = json.dumps(raw, sort_keys=True) if isinstance(raw, (dict, list)) else str(raw)
                    if dedup_key not in seen and len(seen) < self.config.max_varying_samples:
                        seen[dedup_key] = self._compress_field_value(raw, vf, len(group))
            compressed_varying[vf] = list(seen.values())

        summary["_compressed_summary"] = (
            f"{len(group)} structurally similar items "
            f"({len(varying)} varying fields)"
        )

        if self.config.compress_varying_fields and varying:
            summary["_field_stats"] = field_stats
            summary["_varying_samples"] = compressed_varying
        else:
            summary["_content"] = json.dumps(group[0])

        return summary
