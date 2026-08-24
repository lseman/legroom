"""Rolling phase calibration with conservative automatic rollback."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibrationConfig:
    min_samples: int = 20
    window_size: int = 200
    minimum_success_rate: float = 0.75
    minimum_quality: float = 0.95

    def __post_init__(self) -> None:
        if self.min_samples < 1 or self.window_size < self.min_samples:
            raise ValueError("calibration window must contain min_samples")
        if not 0 <= self.minimum_success_rate <= 1 or not 0 <= self.minimum_quality <= 1:
            raise ValueError("calibration thresholds must be between 0 and 1")


@dataclass(frozen=True)
class CalibrationSnapshot:
    phase: str
    samples: int
    success_rate: float
    success_lower_bound: float
    mean_quality: float
    disabled: bool


class CalibrationController:
    """Disable phases whose rolling evidence falls below quality gates."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._samples: dict[str, deque[tuple[bool, float]]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size)
        )

    def record(self, report: dict[str, Any], *, quality: float = 1.0) -> None:
        if not 0 <= quality <= 1:
            raise ValueError("quality must be between 0 and 1")
        name = _canonical_name(str(report.get("name", "unknown")))
        status = report.get("status")
        successful = status in {"applied", "skipped"} and quality >= self.config.minimum_quality
        self._samples[name].append((successful, quality))

    def record_reports(self, reports: list[dict[str, Any]], *, quality: float = 1.0) -> None:
        for report in reports:
            self.record(report, quality=quality)

    @property
    def disabled_phases(self) -> tuple[str, ...]:
        return tuple(
            name for name in sorted(self._samples) if self.snapshot(name).disabled
        )

    def snapshot(self, phase: str) -> CalibrationSnapshot:
        normalized = _canonical_name(phase)
        samples = self._samples.get(normalized, ())
        count = len(samples)
        successes = sum(success for success, _ in samples)
        rate = successes / count if count else 1.0
        quality = sum(value for _, value in samples) / count if count else 1.0
        lower = _wilson_lower_bound(successes, count)
        disabled = (
            count >= self.config.min_samples
            and (
                lower < self.config.minimum_success_rate
                or quality < self.config.minimum_quality
            )
        )
        return CalibrationSnapshot(normalized, count, rate, lower, quality, disabled)

    def snapshots(self) -> tuple[CalibrationSnapshot, ...]:
        return tuple(self.snapshot(name) for name in sorted(self._samples))


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def _canonical_name(value: str) -> str:
    normalized = value.strip().lower()
    return "compress" if normalized == "compression" else normalized
