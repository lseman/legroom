"""Provenance-aware compression safety policy for the typed IR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .ir import CompressionRisk, Conversation


@dataclass(frozen=True)
class RiskAssessment:
    protected_indices: tuple[int, ...]
    labels: tuple[dict[str, Any], ...]

    @property
    def protected_spans(self) -> tuple[str, ...]:
        return tuple(f"message:{index}" for index in self.protected_indices)


class RiskPolicy:
    """Classify once and restore protected messages after every phase."""

    def assess(self, conversation: Conversation) -> RiskAssessment:
        protected: set[int] = set()
        labels: list[dict[str, Any]] = []
        for index, message in enumerate(conversation.messages):
            if message.risk in {CompressionRisk.IMMUTABLE, CompressionRisk.EXACT}:
                protected.add(index)
            labels.append(
                {
                    "index": index,
                    "provenance": message.provenance.value,
                    "risk": message.risk.value,
                }
            )
        # The active user request is immutable. A trailing assistant/tool
        # result may be historical context and remains governed by its own
        # provenance label rather than being protected merely by position.
        for index in range(len(conversation.messages) - 1, -1, -1):
            if conversation.messages[index].provenance.value == "user":
                protected.add(index)
                break
        return RiskAssessment(tuple(sorted(protected)), tuple(labels))

    def restore(
        self,
        original: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
        assessment: RiskAssessment,
    ) -> list[dict[str, Any]]:
        if len(candidate) != len(original):
            return candidate
        restored = list(candidate)
        for index in assessment.protected_indices:
            restored[index] = deepcopy(original[index])
        return restored
