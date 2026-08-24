"""Common execution contract for observable compression phases."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .tokenizer import count_tokens_messages

Messages = list[dict[str, Any]]
PhaseStatus = Literal["applied", "skipped", "rejected", "failed"]


@dataclass(frozen=True)
class PhaseContext:
    model: str
    protected_spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseAnalysis:
    tokens_before: int
    message_count: int
    protected_spans: tuple[str, ...]


@dataclass(frozen=True)
class PhaseProposal:
    messages: Messages
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseValidation:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class PhaseReport:
    name: str
    status: PhaseStatus
    tokens_before: int
    tokens_after: int
    token_delta: int
    protected_spans: tuple[str, ...]
    reversible: bool
    latency_ms: float
    confidence: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseOutcome:
    messages: Messages
    report: PhaseReport
    warnings: tuple[str, ...] = ()


class CompressionPhase(Protocol):
    """Deep interface implemented by every compression phase adapter."""

    name: str
    reversible: bool
    confidence: float

    def analyze(self, messages: Messages, context: PhaseContext) -> PhaseAnalysis: ...

    def propose(
        self, messages: Messages, context: PhaseContext, analysis: PhaseAnalysis
    ) -> PhaseProposal: ...

    def validate(
        self, original: Messages, proposal: PhaseProposal, context: PhaseContext
    ) -> PhaseValidation: ...

    def apply(self, proposal: PhaseProposal) -> Messages: ...


class CallablePhase:
    """Adapter that brings existing transformation functions behind the contract."""

    def __init__(
        self,
        name: str,
        transform: Callable[[Messages], Messages | PhaseProposal],
        *,
        reversible: bool = False,
        confidence: float = 1.0,
        preserve_message_count: bool = True,
        allow_inflation: bool = False,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.name = name
        self.reversible = reversible
        self.confidence = confidence
        self._transform = transform
        self._preserve_message_count = preserve_message_count
        self._allow_inflation = allow_inflation

    def analyze(self, messages: Messages, context: PhaseContext) -> PhaseAnalysis:
        return PhaseAnalysis(
            count_tokens_messages(messages, context.model),
            len(messages),
            context.protected_spans,
        )

    def propose(
        self, messages: Messages, context: PhaseContext, analysis: PhaseAnalysis
    ) -> PhaseProposal:
        proposed = self._transform(messages)
        return proposed if isinstance(proposed, PhaseProposal) else PhaseProposal(proposed)

    def validate(
        self, original: Messages, proposal: PhaseProposal, context: PhaseContext
    ) -> PhaseValidation:
        if self._preserve_message_count and len(proposal.messages) != len(original):
            return PhaseValidation(False, "message count changed")
        if not self._allow_inflation:
            before = count_tokens_messages(original, context.model)
            after = count_tokens_messages(proposal.messages, context.model)
            if after > before:
                return PhaseValidation(False, "token count increased")
        return PhaseValidation(True)

    def apply(self, proposal: PhaseProposal) -> Messages:
        return proposal.messages


class PhaseRunner:
    """Executes the analyze → propose → validate → apply lifecycle."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    def run(
        self, phase: CompressionPhase, messages: Messages, context: PhaseContext
    ) -> PhaseOutcome:
        started = time.perf_counter()
        tokens_before = count_tokens_messages(messages, context.model)
        try:
            analysis = phase.analyze(messages, context)
            proposal = phase.propose(messages, context, analysis)
            validation = phase.validate(messages, proposal, context)
            if validation.accepted:
                output = phase.apply(proposal)
                status: PhaseStatus = "applied" if output != messages else "skipped"
                error = None
            else:
                output = messages
                status = "rejected"
                error = validation.reason
            warnings = proposal.warnings
            metadata = proposal.metadata
        except Exception as exc:
            if self.strict:
                raise
            output = messages
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            warnings = (f"{phase.name} failed: {error}",)
            metadata = {}
        tokens_after = count_tokens_messages(output, context.model)
        report = PhaseReport(
            name=phase.name,
            status=status,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            token_delta=tokens_after - tokens_before,
            protected_spans=context.protected_spans,
            reversible=phase.reversible,
            latency_ms=(time.perf_counter() - started) * 1000,
            confidence=phase.confidence,
            error=error,
            metadata=metadata,
        )
        return PhaseOutcome(output, report, warnings)
