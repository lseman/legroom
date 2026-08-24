"""Dependency-free operational metrics for the Legroom proxy."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProxyMetrics:
    started_at: float = field(default_factory=time.time)
    requests: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    duration_seconds: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    inflight: int = 0
    cache_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cache_cost_usd: float = 0.0
    shadow_requests: int = 0
    shadow_tokens_potentially_saved: int = 0
    calibration_disabled_phases: set[str] = field(default_factory=set)
    phase_status: Counter[tuple[str, str]] = field(default_factory=Counter)
    phase_latency_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    phase_token_delta: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def begin(self) -> float:
        self.inflight += 1
        return time.perf_counter()

    def finish(self, method: str, route: str, status: int, started: float) -> None:
        self.inflight -= 1
        self.requests[(method, route, status)] += 1
        values = self.duration_seconds[(method, route)]
        values.append(time.perf_counter() - started)
        if len(values) > 2048:
            del values[: len(values) - 2048]

    def record_error(self, kind: str) -> None:
        self.errors[kind] += 1

    def record_cache_usage(
        self, *, input_tokens: int, write_tokens: int, read_tokens: int, cost_usd: float
    ) -> None:
        self.cache_input_tokens += input_tokens
        self.cache_write_tokens += write_tokens
        self.cache_read_tokens += read_tokens
        self.cache_cost_usd += cost_usd

    def record_shadow(self, tokens_before: int, tokens_after: int) -> None:
        self.shadow_requests += 1
        self.shadow_tokens_potentially_saved += max(0, tokens_before - tokens_after)

    def set_calibration_disabled(self, phases: tuple[str, ...]) -> None:
        self.calibration_disabled_phases = set(phases)

    def record_phase_report(self, report: dict[str, Any]) -> None:
        phase = str(report.get("name", "unknown")).lower()
        status = str(report.get("status", "unknown"))
        self.phase_status[(phase, status)] += 1
        latency = report.get("latency_ms", 0.0)
        delta = report.get("token_delta", 0)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            self.phase_latency_ms[phase] += float(latency)
        if isinstance(delta, int) and not isinstance(delta, bool):
            self.phase_token_delta[phase] += delta

    def render_prometheus(self) -> str:
        lines = [
            "# HELP legroom_proxy_inflight_requests Current in-flight requests.",
            "# TYPE legroom_proxy_inflight_requests gauge",
            f"legroom_proxy_inflight_requests {self.inflight}",
            "# HELP legroom_proxy_requests_total Proxy requests by route and status.",
            "# TYPE legroom_proxy_requests_total counter",
        ]
        for (method, route, status), count in sorted(self.requests.items()):
            lines.append(
                f'legroom_proxy_requests_total{{method="{method}",route="{route}",status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP legroom_proxy_request_duration_seconds_sum Total request duration.",
                "# TYPE legroom_proxy_request_duration_seconds_sum counter",
            ]
        )
        for (method, route), values in sorted(self.duration_seconds.items()):
            labels = f'method="{method}",route="{route}"'
            lines.append(
                f"legroom_proxy_request_duration_seconds_sum{{{labels}}} {sum(values):.9f}"
            )
            lines.append(f"legroom_proxy_request_duration_seconds_count{{{labels}}} {len(values)}")
        for kind, count in sorted(self.errors.items()):
            lines.append(f'legroom_proxy_errors_total{{kind="{kind}"}} {count}')
        lines.extend(
            [
                f"legroom_provider_input_tokens_total {self.cache_input_tokens}",
                f"legroom_provider_cache_write_tokens_total {self.cache_write_tokens}",
                f"legroom_provider_cache_read_tokens_total {self.cache_read_tokens}",
                f"legroom_provider_input_cost_usd_total {self.cache_cost_usd:.9f}",
                f"legroom_shadow_requests_total {self.shadow_requests}",
                (
                    "legroom_shadow_tokens_potentially_saved_total "
                    f"{self.shadow_tokens_potentially_saved}"
                ),
            ]
        )
        for phase in sorted(self.calibration_disabled_phases):
            lines.append(f'legroom_phase_disabled{{phase="{phase}"}} 1')
        for (phase_name, phase_status), count in sorted(self.phase_status.items()):
            lines.append(
                f'legroom_phase_runs_total{{phase="{phase_name}",status="{phase_status}"}} {count}'
            )
        for phase, latency in sorted(self.phase_latency_ms.items()):
            lines.append(
                f'legroom_phase_latency_ms_sum{{phase="{phase}"}} {latency:.6f}'
            )
        for phase, delta in sorted(self.phase_token_delta.items()):
            lines.append(f'legroom_phase_token_delta_total{{phase="{phase}"}} {delta}')
        return "\n".join(lines) + "\n"
