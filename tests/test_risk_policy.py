from __future__ import annotations

from legroom import CompressConfig, compress
from legroom.runtime.ir import CompressionRisk, Conversation, Provenance
from legroom.runtime.risk_policy import RiskPolicy


def test_ir_assigns_provenance_and_risk():
    conversation = Conversation.from_mappings(
        [
            {"role": "system", "content": "never change"},
            {"role": "tool", "tool_call_id": "call_1", "content": "large result"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [{"id": "call_2"}],
            },
        ]
    )
    assert conversation.messages[0].provenance is Provenance.TRUSTED
    assert conversation.messages[0].risk is CompressionRisk.IMMUTABLE
    assert conversation.messages[1].risk is CompressionRisk.REVERSIBLE
    assert conversation.messages[2].risk is CompressionRisk.EXACT


def test_risk_policy_restores_protected_messages():
    original = [
        {"role": "system", "content": "trusted policy"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
    ]
    policy = RiskPolicy()
    assessment = policy.assess(Conversation.from_mappings(original))
    candidate = [{**message, "content": "changed"} for message in original]
    restored = policy.restore(original, candidate, assessment)
    assert restored[0] == original[0]
    assert restored[1]["content"] == "changed"
    assert restored[2] == original[2]


def test_pipeline_reports_and_enforces_risk_policy():
    system = "<think>private</think> Keep this policy exact"
    result = compress(
        [
            {"role": "system", "content": system},
            {"role": "assistant", "content": "compressible " * 100},
            {"role": "user", "content": "current request"},
        ],
        config=CompressConfig(protect_recent=0, thinking_compact_enabled=True),
    )
    assert result.messages[0]["content"] == system
    assert result.messages[-1]["content"] == "current request"
    assert result.metadata["risk_assessment"][0] == {
        "index": 0,
        "provenance": "trusted",
        "risk": "immutable",
    }
    assert "message:0" in result.metadata["phase_reports"][0]["protected_spans"]
