"""Output policy — defines output behavior policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputPolicy:
    """Defines output behavior policies."""

    max_output_length: int = 4000
    require_concise: bool = True
    strip_preamble: bool = True
    verbosity_level: int = 2
