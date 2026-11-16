"""BD-07 safety gates and threshold modes (no real data)."""
import json

import numpy as np
import pytest

from src import config, final_test


def _spec(mode):
    return {"threshold_mode": mode,
            "specs": [{"model": "xgb", "feature_set": "CASE", "threshold_2023": 0.3, "actual_share_2023": 0.4}]}


def test_carry_over_uses_2023_threshold():
    scores = {("xgb", "CASE"): np.array([0.9, 0.5, 0.2, 0.1, 0.05])}
    assert final_test._thresholds(_spec("carry_over"), scores)[("xgb", "CASE")] == 0.3


def test_share_rule_uses_scores_not_labels():
    # 40% of 5 cases = 2 -> threshold is the second-highest score
    scores = {("xgb", "CASE"): np.array([0.9, 0.5, 0.2, 0.1, 0.05])}
    assert final_test._thresholds(_spec("share_rule"), scores)[("xgb", "CASE")] == 0.5


def test_run_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test, "RUN_LOG", tmp_path / "log.json")
    with pytest.raises(RuntimeError, match="confirm-freeze"):
        final_test.run_test(confirm_freeze=False)


def test_run_only_once(tmp_path, monkeypatch):
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"status": "finished"}))
    monkeypatch.setattr(final_test, "RUN_LOG", log)
    with pytest.raises(RuntimeError, match="only once"):
        final_test.run_test(confirm_freeze=True)


def test_freeze_cannot_be_rewritten_after_test(tmp_path, monkeypatch):
    log = tmp_path / "log.json"
    log.write_text("{}")
    monkeypatch.setattr(final_test, "RUN_LOG", log)
    with pytest.raises(RuntimeError, match="already been run"):
        final_test.write_frozen_spec("carry_over")
