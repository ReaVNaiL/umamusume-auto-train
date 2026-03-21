import core.config as config
import pytest
import utils.constants as constants

from core.umalite_adapter import (
    _approximate_support_bonds,
    _build_race_actions,
    _build_train_actions,
    evaluate_with_umalite,
)
from umalite import RaceGrade, UmaLiteClient


def _friendship_levels(**counts):
    base = {"gray": 0, "blue": 0, "green": 0, "yellow": 0, "max": 0}
    base.update(counts)
    return base


def _training_lane(
    stat_gains,
    *,
    failure=0,
    total_hints=0,
    unity_trainings=0,
    unity_gauge_fills=0,
    unity_spirit_explosions=0,
    support_levels=None,
):
    lane = {
        "stat_gains": stat_gains,
        "failure": failure,
        "total_hints": total_hints,
    }
    if unity_trainings:
        lane["unity_trainings"] = unity_trainings
    if unity_gauge_fills:
        lane["unity_gauge_fills"] = unity_gauge_fills
    if unity_spirit_explosions:
        lane["unity_spirit_explosions"] = unity_spirit_explosions
    if support_levels:
        for stat_key, levels in support_levels.items():
            lane[stat_key] = {"friendship_levels": _friendship_levels(**levels)}
    return lane


@pytest.fixture(autouse=True)
def adapter_env(monkeypatch):
    monkeypatch.setattr(
        constants,
        "TIMELINE",
        [
            "Junior Year Pre-Debut",
            "Junior Year Early Jul",
            "Classic Year Early May",
            "Classic Year Mid May",
            "Classic Year Early Jul",
            "Senior Year Late Dec",
        ],
        raising=False,
    )
    monkeypatch.setattr(config, "MAX_FAILURE", 10, raising=False)
    monkeypatch.setattr(config, "MINIMUM_MOOD", "NORMAL", raising=False)
    monkeypatch.setattr(config, "RACE_SCHEDULE", {}, raising=False)


@pytest.fixture
def ura_template():
    return {
        "stat_weight_set": {
            "spd": 1.0,
            "sta": 0.8,
            "pwr": 0.7,
            "guts": 0.3,
            "wit": 0.6,
            "sp": 0.5,
        },
        "target_stat_set": {
            "spd": 1200,
            "sta": 800,
            "pwr": 900,
            "guts": 400,
            "wit": 600,
        },
        "risk_taking_set": {
            "rainbow_increase": 5,
            "normal_increase": 2,
        },
    }


@pytest.fixture
def ura_state():
    return {
        "current_stats": {
            "spd": 320,
            "sta": 260,
            "pwr": 280,
            "guts": 180,
            "wit": 250,
            "sp": 110,
        },
        "energy_level": 72,
        "current_mood": "GOOD",
        "year": "Classic Year Early May",
        "criteria": "Need more fans",
        "training_results": {
            "spd": _training_lane(
                {"spd": 15, "pwr": 5, "sp": 2},
                failure=2,
                support_levels={
                    "spd": {"gray": 1},
                    "sta": {"yellow": 1},
                },
            ),
            "wit": _training_lane(
                {"spd": 4, "wit": 16, "sp": 5},
                failure=0,
                total_hints=1,
                support_levels={"wit": {"blue": 1}},
            ),
        },
        "date_event_available": True,
        "status_effect_names": ["Night Owl"],
    }


@pytest.fixture
def unity_state():
    return {
        "current_stats": {
            "spd": 360,
            "sta": 220,
            "pwr": 310,
            "guts": 170,
            "wit": 290,
            "sp": 150,
        },
        "energy_level": 64,
        "current_mood": "GOOD",
        "year": "Classic Year Mid May",
        "criteria": "Achieved",
        "training_results": {
            "spd": _training_lane(
                {"spd": 16, "pwr": 6, "sp": 3},
                failure=3,
                total_hints=1,
                unity_trainings=3,
                unity_gauge_fills=1,
                unity_spirit_explosions=2,
                support_levels={
                    "spd": {"yellow": 1},
                    "pwr": {"green": 1},
                },
            ),
            "wit": _training_lane(
                {"spd": 5, "wit": 14, "sp": 4},
                failure=0,
                unity_trainings=1,
                unity_gauge_fills=1,
                support_levels={"wit": {"max": 1}},
            ),
        },
        "date_event_available": True,
    }


def test_approximate_support_bonds_keep_lane_scoped_ids(ura_state):
    bonds = _approximate_support_bonds(ura_state["training_results"])

    assert bonds is not None
    assert bonds["spd_spd_0"] == 10
    assert bonds["spd_sta_0"] == 80
    assert bonds["wit_wit_0"] == 30


def test_build_train_actions_adds_unity_terms_and_bond_gains(unity_state, monkeypatch):
    monkeypatch.setattr(constants, "SCENARIO_NAME", "unity", raising=False)
    sc = UmaLiteClient().unity

    actions = _build_train_actions(sc, unity_state["training_results"], "unity")
    speed_action = next(action for action in actions if action.action_id == "train_spd")
    wit_action = next(action for action in actions if action.action_id == "train_wit")

    assert speed_action.payload["scenario_terms"] == {
        "gauge_gain": 1.0,
        "facility_progress": 2.0,
        "burst_value": 2.0,
    }
    assert speed_action.payload["bond_gains"] == {
        "spd_spd_0": 5,
        "spd_pwr_0": 5,
    }
    assert wit_action.payload["scenario_terms"] == {"gauge_gain": 1.0}


def test_build_race_actions_prefers_fans_gained(monkeypatch):
    monkeypatch.setattr(constants, "SCENARIO_NAME", "ura", raising=False)
    monkeypatch.setattr(
        config,
        "RACE_SCHEDULE",
        {
            "Classic Year Early May": [
                {"name": "Spring Stakes", "grade": "g2", "fans_gained": 4200},
            ]
        },
        raising=False,
    )
    sc = UmaLiteClient().ura

    actions = _build_race_actions(sc, {}, "Classic Year Early May")

    assert len(actions) == 1
    assert actions[0].action_id == "race_Spring Stakes"
    assert actions[0].payload["fan_gain"] == 4200
    assert actions[0].payload["race_grade"] == RaceGrade.G2.value


def test_evaluate_with_umalite_returns_ura_result(ura_state, ura_template, monkeypatch):
    monkeypatch.setattr(constants, "SCENARIO_NAME", "ura", raising=False)
    monkeypatch.setattr(
        config,
        "RACE_SCHEDULE",
        {
            "Classic Year Early May": [
                {"name": "NHK Mile Cup", "grade": "g1", "fans_gained": 5200},
            ]
        },
        raising=False,
    )

    result = evaluate_with_umalite(ura_state, ura_template)

    assert result is not None
    assert result.best_action_id in {
        "train_spd",
        "train_wit",
        "race_NHK Mile Cup",
        "rest",
        "recreation",
    }


def test_evaluate_with_umalite_returns_unity_result(unity_state, ura_template, monkeypatch):
    monkeypatch.setattr(constants, "SCENARIO_NAME", "unity", raising=False)

    result = evaluate_with_umalite(unity_state, ura_template)

    assert result is not None
    assert result.best_action_id in {"train_spd", "train_wit", "rest", "recreation"}
