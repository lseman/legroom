"""Dependency-free operational metrics for the Legroom proxy."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class ProxyMetrics:
    started_at: float = field(default_factory=time.time)
    requests: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    duration_seconds: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    inflight: int = 0

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
        return "\n".join(lines) + "\n"
