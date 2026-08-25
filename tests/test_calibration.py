from __future__ import annotations

from legroom.calibration import CalibrationConfig, CalibrationController


def test_calibration_disables_reliably_failing_phase():
    controller = CalibrationController(
        CalibrationConfig(min_samples=4, window_size=10, minimum_success_rate=0.5)
    )
    for _ in range(4):
        controller.record({"name": "semantic_dedup", "status": "failed"})
    snapshot = controller.snapshot("semantic_dedup")
    assert snapshot.disabled
    assert controller.disabled_phases == ("semantic_dedup",)


def test_calibration_uses_downstream_quality_signal():
    controller = CalibrationController(
        CalibrationConfig(
            min_samples=2,
            window_size=2,
            minimum_success_rate=0.0,
            minimum_quality=0.9,
        )
    )
    controller.record({"name": "compress", "status": "applied"}, quality=0.5)
    controller.record({"name": "compress", "status": "applied"}, quality=0.7)
    assert controller.snapshot("compress").disabled


def test_calibration_does_not_invent_quality_when_evaluator_is_absent():
    controller = CalibrationController(
        CalibrationConfig(min_samples=2, window_size=2, minimum_success_rate=0.0)
    )
    controller.record({"name": "compress", "status": "applied"})
    controller.record({"name": "compress", "status": "skipped"})
    snapshot = controller.snapshot("compress")
    assert snapshot.mean_quality is None
    assert not snapshot.disabled
