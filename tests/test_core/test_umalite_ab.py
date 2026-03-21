import json
import sys
import types
from importlib import import_module
from types import SimpleNamespace

import pytest

umalite_ab = import_module("core.umalite_ab")


class FakeAction:
    def __init__(self, func, options=None, available_actions=None):
        self.func = func
        self.options = options or {}
        self.available_actions = available_actions or []

    def __getitem__(self, key):
        return self.options[key]

    def __setitem__(self, key, value):
        self.options[key] = value

    def get(self, key, default=None):
        return self.options.get(key, default)


@pytest.fixture
def fake_result():
    bids = [
        SimpleNamespace(
            action=SimpleNamespace(action_id="train_spd"),
            score=12.5,
            breakdown={"stats": 10.0},
        ),
        SimpleNamespace(
            action=SimpleNamespace(action_id="rest"),
            score=8.0,
            breakdown={"energy": 8.0},
        ),
    ]
    return SimpleNamespace(
        best=SimpleNamespace(score=12.5),
        best_action_id="train_spd",
        bids=bids,
    )


def test_prepare_umalite_ab_stashes_selected_record(monkeypatch, fake_result):
    action = FakeAction(
        "do_training",
        options={
            "training_name": "spd",
            "training_data": {"score_tuple": (7.5, 0)},
            "available_trainings": {
                "spd": {"score_tuple": (7.5, 0)},
                "wit": {"score_tuple": (5.0, 0)},
            },
        },
        available_actions=["do_training", "do_rest"],
    )
    state = {
        "year": "Classic Year Early May",
        "turn": 12,
        "energy_level": 68,
        "current_mood": "GOOD",
        "current_stats": {"spd": 400},
        "training_results": {
            "spd": {"stat_gains": {"spd": 15}, "failure": 2},
            "wit": {"failure": 0},
        },
    }
    training_template = {"target_stat_set": {"spd": 1200}}

    monkeypatch.setattr(umalite_ab, "evaluate_with_umalite", lambda *_args: fake_result)
    monkeypatch.setattr(umalite_ab, "normalize_state", lambda *_args: object())
    monkeypatch.setattr(
        umalite_ab,
        "derive_checkpoint",
        lambda _norm: SimpleNamespace(kind=SimpleNamespace(value="goal_race"), turns_remaining=3),
    )
    monkeypatch.setattr(
        umalite_ab,
        "resolve_policy",
        lambda _template: SimpleNamespace(
            stat_weights={"speed": 1.0},
            target_stats={"speed": 1200},
            risk_tolerance=0.25,
            max_failure_rate=0.1,
            min_mood=SimpleNamespace(name="NORMAL"),
        ),
    )
    monkeypatch.setattr(umalite_ab.constants, "SCENARIO_NAME", "ura", raising=False)

    umalite_ab.prepare_umalite_ab(state, training_template, action)

    record = action.options["umalite_ab"]
    assert record["legacy_selected_action_id"] == "train_spd"
    assert record["legacy_selected_action_family"] == "train"
    assert record["umalite_best_action_id"] == "train_spd"
    assert record["selected_exact_match"] is True
    assert record["turn_projection_mode"] == "heuristic"
    assert record["legacy_training_scores"] == {
        "train_spd": 7.5,
        "train_wit": 5.0,
    }


def test_record_umalite_ab_writes_selected_and_executed_actions(tmp_path, monkeypatch):
    action = FakeAction(
        "do_rest",
        options={
            "umalite_ab": {
                "legacy_selected_action_id": "train_spd",
                "legacy_selected_action_family": "train",
                "umalite_best_action_id": "rest",
                "umalite_best_action_family": "rest",
            }
        },
    )

    monkeypatch.setattr(umalite_ab.log_module, "log_dir", str(tmp_path), raising=False)

    umalite_ab.record_umalite_ab(action)

    log_path = tmp_path / "umalite_ab_test.jsonl"
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["legacy_selected_action_id"] == "train_spd"
    assert payload["legacy_executed_action_id"] == "rest"
    assert payload["executed_exact_match"] is True
    assert payload["execution_fallback_used"] is True
