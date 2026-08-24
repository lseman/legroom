"""SmartCrusher — intelligent JSON compression with value-aware grouping.

Improvements over the original:
1. **Value-aware grouping**: Items are grouped by both structure AND value
   similarity (via value fingerprints), preventing structurally-similar but
   semantically-different items from being collapsed.
2. **Value entropy analysis**: Before compressing a group, check if the
   compression is worth it. High entropy → keep items as-is.
3. **Nested object compression**: Recursively compress nested objects
   within item fields.
4. **Key name shortening**: Replace repeated key names in a summary with
   a compact ``_keys`` / ``_values`` matrix representation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..query_relevance import query_relevance
from ..tokenizer import count_tokens
from .adaptive_sizer import compute_optimal_k
from .compressor_registry import CompressOutput


@dataclass
class SmartCrusherConfig:
    """Configuration for SmartCrusher."""

    max_items: int = 50
    adaptive_sizing: bool = False
    size_bias: float = 1.0
    protect_recent: int = 0
    min_group_size: int = 3
    # Field-level compression settings
    compress_varying_fields: bool = True
    max_varying_field_chars: int = 40
    max_varying_samples: int = 5
    # Value-aware grouping threshold — items with fingerprint distance
    # above this are placed in separate groups.
    value_fingerprint_max_distance: int = 3
    # Query-aware retention
    query_relevance_threshold: float = 0.3
    # Enable key name shortening in summaries (replaces repeated key names
    # with a compact _keys/_values matrix).
    key_shortening: bool = True
    # Enable nested object compression within item fields.
    nested_compression: bool = True


class SmartCrusher:
    """Compresses arrays of items with value-aware grouping and field-level dedup.

    Value-aware grouping:
    - Items are grouped by both structural fingerprint AND value fingerprint.
    - The value fingerprint captures the "shape" of values (hash of first N
      chars for strings, exact value for numbers/bools, structural fingerprint
      for nested objects).
    - This prevents grouping items that have the same keys but very different
      values (e.g. {"id": 1, "name": "foo"} and {"id": 2, "name": "bar"}).

    Key name shortening:
    - In summaries, repeated key names are replaced with a compact
      ``_keys`` / ``_values`` matrix representation, saving tokens.

    Nested compression:
    - Nested objects within item fields are recursively compressed.
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
        """Compress JSON content."""
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
        return CompressOutput(
            compressed=content,
            original_token_count=count_tokens(content, model),
            compressed_token_count=count_tokens(content, model),
            strategy="smart_crusher",
        )

    def _crush_array(
        self, items: list, metadata: dict, query_terms: set[str] | None = None
    ) -> list:
        """Compress an array of items."""
        if len(items) <= 1:
            return items

        # Protect recent items (rolling window)
        protect_recent = metadata.get("protect_recent", self.config.protect_recent)
        recent = items[-protect_recent:] if protect_recent > 0 else []
        to_compress = items[:-protect_recent] if protect_recent > 0 else items

        # Query-aware retention
        relevant: list = []
        if query_terms:
            still_to_compress = []
            for item in to_compress:
                item_text = item if isinstance(item, str) else json.dumps(item)
                if query_relevance(
                    item_text, query_terms
                ) >= self.config.query_relevance_threshold:
                    relevant.append(item)
                else:
                    still_to_compress.append(item)
            to_compress = still_to_compress

        # Value-aware grouping
        groups = self._find_value_aware_groups(to_compress)

        result = []
        for group in groups:
            if len(group) < self.config.min_group_size:
                result.extend(group)
                continue

            if self.config.adaptive_sizing:
                serialized = [json.dumps(item, sort_keys=True) for item in group]
                k = compute_optimal_k(serialized, bias=self.config.size_bias)
                if k >= len(group) * 0.75:
                    result.extend(group)
                    continue

            # Value entropy check: don't compress if the group is too diverse
            entropy = self._compute_group_entropy(group)
            if entropy > 0.7:
                result.extend(group)
                continue

            result.append(self._summarize_group(group))

        result.extend(relevant)
        result.extend(recent)
        return result

    # ------------------------------------------------------------------
    # Value-aware grouping
    # ------------------------------------------------------------------

    def _find_value_aware_groups(self, items: list) -> list[list]:
        """Group items by both structural fingerprint AND value fingerprint.

        The structural fingerprint captures key names and types.
        The value fingerprint captures the "shape" of values to ensure
        items in a group are actually similar, not just structurally similar.
        """
        groups: dict[str, list] = defaultdict(list)
        for item in items:
            if isinstance(item, dict):
                struct_key = self._structural_key(item)
                value_key = self._value_fingerprint(item)
                groups[f"{struct_key}|{value_key}"].append(item)
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

    def _value_fingerprint(self, item: dict) -> str:
        """Create a value-level fingerprint for an item.

        Captures the "shape" of values — not exact values, but enough to
        distinguish items with very different values from items with similar
        values. This prevents grouping items like
        {"id": 1, "name": "foo"} and {"id": 2, "name": "bar"}.
        """
        parts = []
        for key in sorted(item.keys()):
            val = item[key]
            parts.append(f"{key}:{self._value_fp_for(val)}")
        return "|".join(parts)

    def _value_fp_for(self, value: Any) -> str:
        """Value fingerprint for a single value."""
        if isinstance(value, str):
            # Hash the first 50 chars — captures the "shape" without
            # requiring exact match. Two strings that start differently
            # get different fingerprints.
            if len(value) <= 50:
                return f"str:{value}"
            return f"str:{value[:50]}"
        elif isinstance(value, (int, float)):
            # Numbers: use the exact value (or a bucketed version for floats)
            if isinstance(value, float):
                return f"num:{round(value, 2)}"
            return f"num:{value}"
        elif isinstance(value, bool):
            return f"bool:{value}"
        elif value is None:
            return "null"
        elif isinstance(value, dict):
            # Nested object: structural fingerprint of the nested object
            return f"obj:{self._structural_key(value)}"
        elif isinstance(value, list):
            # Array: length + structural fingerprint of first element
            first_fp = (
                self._value_fp_for(value[0]) if value else "empty"
            )
            return f"arr:{len(value)}:{first_fp}"
        else:
            return f"other:{str(value)[:30]}"

    # ------------------------------------------------------------------
    # Value entropy analysis
    # ------------------------------------------------------------------

    def _compute_group_entropy(self, group: list) -> float:
        """Compute the entropy of a group of items.

        Returns a value in [0, 1] where 0 = all items identical,
        1 = all items completely different. Groups with entropy > 0.7
        are too diverse to compress safely.
        """
        if len(group) <= 1:
            return 0.0

        # Compute pairwise value similarity using fingerprint distance
        fingerprints = [self._value_fingerprint(item) for item in group]
        total_distance = 0
        count = 0
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                dist = self._fingerprint_distance(fingerprints[i], fingerprints[j])
                total_distance += dist
                count += 1

        if count == 0:
            return 0.0

        avg_distance = total_distance / count
        # Normalize: distance of 0 = identical, distance of max = different
        # Max possible distance is roughly len(fingerprint) * 2
        max_dist = max(1, len(fingerprints[0]) * 2)
        return min(avg_distance / max_dist, 1.0)

    def _fingerprint_distance(self, fp1: str, fp2: str) -> int:
        """Compute the edit distance between two value fingerprints."""
        # Simple character-level edit distance (Levenshtein)
        # Optimized: early exit if distance exceeds threshold
        n, m = len(fp1), len(fp2)
        if abs(n - m) > self.config.value_fingerprint_max_distance:
            return self.config.value_fingerprint_max_distance + 1

        # Use a simplified distance: count differing segments
        # This is faster than full Levenshtein and good enough
        parts1 = fp1.split("|")
        parts2 = fp2.split("|")
        distance = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 != p2:
                # Count character differences within the segment
                distance += self._segment_distance(p1, p2)
        return min(distance, self.config.value_fingerprint_max_distance + 1)

    def _segment_distance(self, s1: str, s2: str) -> int:
        """Approximate distance between two fingerprint segments."""
        if s1 == s2:
            return 0
        # Quick heuristic: different prefixes → high distance
        for i in range(min(len(s1), len(s2))):
            if s1[i] != s2[i]:
                return max(1, abs(len(s1) - len(s2)) + 1)
        return max(1, abs(len(s1) - len(s2)))

    # ------------------------------------------------------------------
    # Summary with key name shortening and nested compression
    # ------------------------------------------------------------------

    def _summarize_group(self, group: list) -> dict:
        """Create a summary of a group of similar items.

        Improvements:
        - Key name shortening: replaces repeated key names with a compact
          _keys / _values matrix representation.
        - Nested object compression: recursively compresses nested objects
          within item fields.
        """
        if not group:
            return {}

        varying = self._find_varying_fields(group)
        fixed_fields = {
            k: v for k, v in group[0].items() if k not in varying
        }

        # Compress nested objects if enabled
        if self.config.nested_compression:
            fixed_fields = {
                k: self._compress_nested(v) for k, v in fixed_fields.items()
            }

        summary: dict[str, Any] = dict(fixed_fields)

        # Build varying field summaries
        field_stats = {}
        compressed_varying = {}
        for vf in varying:
            stats = self._compute_field_stats(group, vf)
            field_stats[vf] = stats

            seen: dict[str, Any] = {}
            for item in group:
                if isinstance(item, dict) and vf in item:
                    raw = item[vf]
                    dedup_key = (
                        json.dumps(raw, sort_keys=True)
                        if isinstance(raw, (dict, list))
                        else str(raw)
                    )
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

            # Key name shortening: replace repeated key names with
            # a compact matrix representation
            if self.config.key_shortening and len(group) >= 3:
                self._apply_key_shortening(summary, group, varying)
        else:
            summary["_content"] = json.dumps(group[0])

        return summary

    def _apply_key_shortening(
        self,
        summary: dict,
        group: list,
        varying_fields: list[str],
    ) -> None:
        """Replace repeated key names with a compact _keys / _values matrix.

        For a group of N items with M varying fields, the original
        representation repeats key names N times. The shortened version
        stores key names once and values as a matrix.

        Example:
            Before: {"status": "active", "id": 1, "name": "foo"}
                    {"status": "active", "id": 2, "name": "bar"}
                    {"status": "active", "id": 3, "name": "baz"}
            After:  {"status": "active", "_keys": ["id", "name"],
                    "_values": [[1, "foo"], [2, "bar"], [3, "baz"]]}
        """
        # Collect all varying field names
        keys = list(varying_fields)

        # Build value matrix: each row is one item's values for the varying fields
        values = []
        for item in group:
            row = []
            for key in keys:
                if isinstance(item, dict) and key in item:
                    row.append(self._compress_field_value(item[key], key, len(group)))
                else:
                    row.append(None)
            values.append(row)

        summary["_keys"] = keys
        summary["_values"] = values

        # Remove the verbose _varying_samples and _field_stats since
        # the matrix representation is more compact
        del summary["_varying_samples"]
        del summary["_field_stats"]

    def _compress_nested(self, value: Any) -> Any:
        """Recursively compress nested objects within field values.

        For nested dicts/lists, replaces them with a compact representation
        that captures the structure without full serialization.
        """
        if isinstance(value, dict):
            if len(value) > 5:
                # Large nested object — replace with summary
                return {
                    "_nested": True,
                    "type": "dict",
                    "keys_count": len(value),
                    "keys": list(value.keys())[:10],  # Top 10 keys
                    "summary": f"{len(value)} keys",
                }
            # Small nested object — recurse into values
            return {k: self._compress_nested(v) for k, v in value.items()}
        elif isinstance(value, list):
            if len(value) > 10:
                # Large nested array — replace with summary
                first_sample = (
                    self._compress_nested(value[0]) if value else None
                )
                return {
                    "_nested": True,
                    "type": "list",
                    "length": len(value),
                    "first_sample": first_sample,
                    "summary": f"{len(value)} elements",
                }
            # Small nested array — recurse into elements
            return [self._compress_nested(v) for v in value]
        return value

    # ------------------------------------------------------------------
    # Field-level compression (unchanged from original)
    # ------------------------------------------------------------------

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
                    values.add(
                        json.dumps(val, sort_keys=True)
                        if isinstance(val, (dict, list))
                        else str(val)
                    )
                    if len(values) > 1:
                        varying.append(key)
                        break

        return varying

    def _compute_field_stats(
        self, group: list, field_name: str
    ) -> dict[str, Any]:
        """Compute statistics for a varying field across a group."""
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
            "sample_keys": (
                list(values[0].keys()) if isinstance(values[0], dict) else None
            ),
        }

    def _compress_field_value(
        self, value: Any, field_name: str, group_size: int
    ) -> Any:
        """Compress a single varying field value."""
        if isinstance(value, str):
            max_chars = self.config.max_varying_field_chars
            if len(value) > max_chars:
                return (
                    f"{value[:max_chars // 2]}..."
                    f"{value[-max_chars // 3:]}"
                )
            return value
        elif isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, (dict, list)):
            return {
                "_truncated": True,
                "type": "dict" if isinstance(value, dict) else "list",
                "keys_count": len(value),
                "summary": f"{group_size} items with {len(value)} keys/elements",
            }
        elif value is None:
            return None
        else:
            return str(value)
