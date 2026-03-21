from __future__ import annotations

from typing import Any

import core.config as config
import utils.constants as constants
from utils.log import debug, info, warning
from umalite import (
    CheckpointContext,
    CheckpointKind,
    EvaluationResult,
    MoodLevel,
    RaceGrade,
    StatType,
    UmaLiteClient,
)
from umalite.config import EvaluationConfig, UtilityConfig
from umalite.models.enums import CareerYear, TurnPhase


STAT_BOT_TO_LITE = {
    "spd": "speed",
    "sta": "stamina",
    "pwr": "power",
    "guts": "guts",
    "wit": "wit",
}

CONDITION_BOT_TO_LITE: dict[str, str] = {
    "Migraine": "migraine",
    "Night Owl": "night_owl",
    "Practice Poor": "practice_poor",
    "Skin Outbreak": "dry_skin",
    "Slacker": "slacker",
    "Slow Metabolism": "slow_metabolism",
    "Charming": "charming",
    "Fast Learner": "fast_learner",
    "Hot Topic": "hot_topic",
    "Practice Perfect": "practice_perfect",
}

STAT_TYPE_FOR = {
    "speed": StatType.SPEED,
    "stamina": StatType.STAMINA,
    "power": StatType.POWER,
    "guts": StatType.GUTS,
    "wit": StatType.WIT,
}
MOOD_FOR = {
    "GREAT": MoodLevel.GREAT,
    "GOOD": MoodLevel.GOOD,
    "NORMAL": MoodLevel.NORMAL,
    "BAD": MoodLevel.BAD,
    "AWFUL": MoodLevel.AWFUL,
}
YEAR_FOR = {
    "Junior": CareerYear.JUNIOR,
    "Classic": CareerYear.CLASSIC,
    "Senior": CareerYear.SENIOR,
    "Finale": CareerYear.SENIOR,
}
PHASE_FOR = {
    "Early": TurnPhase.EARLY,
    "Mid": TurnPhase.MID,
    "Late": TurnPhase.LATE,
    "Pre-Debut": TurnPhase.EARLY,
}
GRADE_FOR = {
    "g1": RaceGrade.G1,
    "g2": RaceGrade.G2,
    "g3": RaceGrade.G3,
    "op": RaceGrade.OP,
    "pre_op": RaceGrade.PRE_OP,
}


def _translate_stats(bot_stats: dict[str, Any]) -> dict[str, int]:
    return {
        STAT_BOT_TO_LITE[k]: int(v)
        for k, v in bot_stats.items()
        if k in STAT_BOT_TO_LITE
    }


def _translate_stat_gains(raw: dict[str, Any]) -> tuple[dict[str, int], int]:
    stats = {}
    sp = 0
    for k, v in raw.items():
        if k == "sp":
            sp = max(0, int(v))
        elif k in STAT_BOT_TO_LITE and isinstance(v, (int, float)) and v >= 0:
            stats[STAT_BOT_TO_LITE[k]] = int(v)
    return stats, sp


def _parse_year(year_string: str) -> tuple[CareerYear | None, TurnPhase | None]:
    if year_string == "Finale Underway":
        return CareerYear.SENIOR, TurnPhase.LATE
    parts = year_string.split()
    if len(parts) < 3:
        return None, None
    return YEAR_FOR.get(parts[0]), PHASE_FOR.get(
        parts[2] if parts[1] == "Year" else parts[1]
    )


def _turn_index(year_string: str) -> int | None:
    return (
        constants.TIMELINE.index(year_string)
        if year_string in constants.TIMELINE
        else None
    )


def _translate_conditions(state_obj: dict[str, Any]) -> set[str] | None:
    names = state_obj.get("status_effect_names")
    if not names:
        return None
    mapped = {CONDITION_BOT_TO_LITE[n] for n in names if n in CONDITION_BOT_TO_LITE}
    return mapped or None


def _resolve_scenario() -> str:
    name = getattr(constants, "SCENARIO_NAME", "")
    if name == "unity":
        return "unity"
    if name in ("trackblazer", "mant"):
        return "trackblazer"
    if name == "" or name == "ura":
        return "ura"
    raise ValueError(f"Unsupported scenario: {name!r}")


def _scenario_client(client: UmaLiteClient, scenario: str):
    if scenario == "unity":
        return client.unity
    if scenario == "trackblazer":
        return client.trackblazer
    return client.ura


def _scheduled_race_for(year_string: str) -> str | None:
    schedule = getattr(config, "RACE_SCHEDULE", None)
    if not isinstance(schedule, dict):
        return None
    races = schedule.get(year_string, [])
    if races and isinstance(races[0], dict):
        return races[0].get("name")
    return None


def _build_config(
    scenario: str, stat_weights: dict[str, float], sp_weight: float
) -> EvaluationConfig:
    if scenario == "unity":
        from umalite.scenarios.unity.config import DEFAULT_UNITY_UTILITY as defaults
        from umalite.scenarios.unity.config import DEFAULT_UNITY_SCENARIO

        scenario_cfg = DEFAULT_UNITY_SCENARIO
    elif scenario == "trackblazer":
        from umalite.scenarios.trackblazer.config import (
            DEFAULT_TRACKBLAZER_UTILITY as defaults,
        )
        from umalite.scenarios.trackblazer.config import DEFAULT_TRACKBLAZER_SCENARIO

        scenario_cfg = DEFAULT_TRACKBLAZER_SCENARIO
    else:
        from umalite.scenarios.ura.config import DEFAULT_URA_UTILITY as defaults

        scenario_cfg = None

    utility = UtilityConfig(
        stat_weights=stat_weights,
        stat_soft_cap=defaults.stat_soft_cap,
        sp_weight=sp_weight,
        mood_weight=defaults.mood_weight,
        energy_weight=defaults.energy_weight,
        fan_weight=defaults.fan_weight,
        bond_weight=defaults.bond_weight,
        hint_weight=defaults.hint_weight,
    )
    return EvaluationConfig(utility=utility, scenario=scenario_cfg)


def _build_checkpoint(
    year_string: str, idx: int | None, criteria: str
) -> CheckpointContext:
    if year_string not in constants.TIMELINE:
        return CheckpointContext(kind=CheckpointKind.NONE, turns_remaining=24)

    current = idx or 0
    turns_to_end = len(constants.TIMELINE) - current

    schedule = getattr(config, "RACE_SCHEDULE", None)
    if isinstance(schedule, dict) and schedule:
        for i in range(current, len(constants.TIMELINE)):
            date = constants.TIMELINE[i]
            if date in schedule and schedule[date]:
                distance = max(i - current, 1)
                if distance <= 1:
                    return CheckpointContext(
                        kind=CheckpointKind.NONE, turns_remaining=1
                    )
                return CheckpointContext(
                    kind=CheckpointKind.GOAL_RACE, turns_remaining=distance
                )

    if "fan" in criteria.lower():
        return CheckpointContext(
            kind=CheckpointKind.FAN_GATE, turns_remaining=min(turns_to_end, 12)
        )

    if any(m in year_string for m in ("Early Jun", "Late May", "Early May")):
        prefix = year_string.split()[0]
        target = f"{prefix} Year Early Jul"
        if target in constants.TIMELINE:
            to_summer = constants.TIMELINE.index(target) - current
            if to_summer > 0:
                return CheckpointContext(
                    kind=CheckpointKind.SUMMER_WINDOW, turns_remaining=to_summer
                )

    return CheckpointContext(kind=CheckpointKind.NONE, turns_remaining=turns_to_end)


APPROXIMATE_SUPPORT_BOND = {
    "gray": 10,
    "blue": 30,
    "green": 50,
    "yellow": 80,
    "max": 90,
}
APPROXIMATE_BOND_GAIN = 5


def _support_slot_ids(
    lane: str,
    stat_key: str,
    levels: dict[str, Any],
) -> list[tuple[str, int]]:
    slots: list[tuple[str, int]] = []
    slot_index = 0
    for color in ("gray", "blue", "green", "yellow", "max"):
        count = levels.get(color, 0)
        if not isinstance(count, int) or count <= 0:
            continue
        bond = APPROXIMATE_SUPPORT_BOND.get(color, 50)
        for _ in range(count):
            slots.append((f"{lane}_{stat_key}_{slot_index}", bond))
            slot_index += 1
    return slots


def _approximate_support_bonds(
    training_results: dict[str, Any],
) -> dict[str, int] | None:
    bonds: dict[str, int] = {}
    for lane, data in training_results.items():
        if lane not in STAT_BOT_TO_LITE:
            continue
        for stat_key in STAT_BOT_TO_LITE:
            card_data = data.get(stat_key)
            if not isinstance(card_data, dict):
                continue
            levels = card_data.get("friendship_levels", {})
            if not isinstance(levels, dict):
                continue
            for card_id, bond in _support_slot_ids(lane, stat_key, levels):
                bonds[card_id] = bond
    return bonds or None


def _approximate_bond_gains(data: dict[str, Any], lane: str) -> dict[str, int] | None:
    gains: dict[str, int] = {}
    for stat_key in STAT_BOT_TO_LITE:
        card_data = data.get(stat_key)
        if not isinstance(card_data, dict):
            continue
        levels = card_data.get("friendship_levels", {})
        if not isinstance(levels, dict):
            continue
        for card_id, _bond in _support_slot_ids(lane, stat_key, levels):
            gains[card_id] = APPROXIMATE_BOND_GAIN
    return gains or None


def _unity_scenario_terms(data: dict[str, Any]) -> dict[str, float] | None:
    gauge_gain = max(0, int(data.get("unity_gauge_fills", 0)))
    raw_facility = max(0, int(data.get("unity_trainings", 0)))
    facility_progress = max(0, raw_facility - gauge_gain)
    burst_value = max(0, int(data.get("unity_spirit_explosions", 0)))

    terms: dict[str, float] = {}
    if gauge_gain:
        terms["gauge_gain"] = float(gauge_gain)
    if facility_progress:
        terms["facility_progress"] = float(facility_progress)
    if burst_value:
        terms["burst_value"] = float(burst_value)
    return terms or None


def _build_train_actions(sc, training_results: dict[str, Any], scenario: str) -> list:
    actions = []
    for lane, data in training_results.items():
        if lane not in STAT_BOT_TO_LITE:
            continue
        raw_gains = data.get("stat_gains")
        if not isinstance(raw_gains, dict) or not raw_gains:
            continue

        stat_gains, sp_gain = _translate_stat_gains(raw_gains)
        if not stat_gains:
            continue

        training_type = STAT_TYPE_FOR.get(STAT_BOT_TO_LITE[lane])
        if not training_type:
            continue

        fail_rate = float(data.get("failure", 0)) / 100.0
        hint_count = int(data.get("total_hints", 0))
        bond_gains = _approximate_bond_gains(data, lane)
        train_kwargs = {
            "training_type": training_type,
            "stat_gains": stat_gains,
            "sp_gain": sp_gain,
            "fail_rate": fail_rate,
            "hint_count": hint_count,
            "bond_gains": bond_gains,
        }
        if scenario == "unity":
            scenario_terms = _unity_scenario_terms(data)
            if scenario_terms:
                train_kwargs["scenario_terms"] = scenario_terms

        actions.append(
            sc.actions.train(
                f"train_{lane}",
                **train_kwargs,
            )
        )

    return actions


def _build_race_actions(sc, state_obj: dict[str, Any], year_string: str) -> list:
    actions = []
    schedule = getattr(config, "RACE_SCHEDULE", None)
    if isinstance(schedule, dict):
        for race in schedule.get(year_string, []):
            if not isinstance(race, dict):
                continue
            race_name = race.get("name", "scheduled_race")
            fans_gained = race.get(
                "fans_gained",
                race.get("fans", {}).get("gained", 3000),
            )
            grade = GRADE_FOR.get(race.get("grade", "g3").lower().replace("-", "_"))
            if not grade:
                continue
            actions.append(
                sc.actions.race(
                    f"race_{race_name}",
                    race_grade=grade,
                    fan_gain=fans_gained,
                )
            )

    if state_obj.get("race_mission_available"):
        actions.append(
            sc.actions.race("race_mission", race_grade=RaceGrade.G3, fan_gain=3000)
        )

    return actions


def evaluate_with_umalite(
    state_obj: dict[str, Any],
    training_template: dict[str, Any],
) -> EvaluationResult | None:
    info("[UmaLiteAdapter] Starting evaluation...")
    scenario = _resolve_scenario()

    weight_set = training_template.get("stat_weight_set", {})
    stat_weights = {
        STAT_BOT_TO_LITE[k]: max(0.0, float(v))
        for k, v in weight_set.items()
        if k in STAT_BOT_TO_LITE
    }
    sp_weight = float(weight_set.get("sp", 0.5))

    eval_config = _build_config(scenario, stat_weights, sp_weight)
    client = UmaLiteClient(config=eval_config)
    sc = _scenario_client(client, scenario)

    bot_stats = state_obj.get("current_stats", {})
    current_stats = _translate_stats(bot_stats)
    energy = max(0, min(100, int(round(state_obj.get("energy_level", 50)))))
    mood = MOOD_FOR.get(state_obj.get("current_mood", "NORMAL"), MoodLevel.NORMAL)
    year_string = state_obj.get("year", "")
    career_year, turn_phase = _parse_year(year_string)
    idx = _turn_index(year_string)
    conditions = _translate_conditions(state_obj)
    scheduled_race_id = _scheduled_race_for(year_string)
    support_bonds = _approximate_support_bonds(state_obj.get("training_results", {}))

    snapshot = sc.snapshots.enriched(
        stats=current_stats,
        energy=energy,
        mood=mood,
        fans=0,
        turn_index=idx,
        year=career_year,
        phase=turn_phase,
        skill_points=bot_stats.get("sp"),
        conditions=conditions,
        scheduled_race_id=scheduled_race_id,
        support_bonds=support_bonds,
    )

    criteria = state_obj.get("criteria", "")
    checkpoint = _build_checkpoint(year_string, idx, criteria)
    target_stats = _translate_stats(training_template.get("target_stat_set", {}))

    risk_taking = training_template.get(
        "risk_taking_set", {"rainbow_increase": 0, "normal_increase": 0}
    )
    risk_tolerance = min(
        1.0, (risk_taking["rainbow_increase"] + risk_taking["normal_increase"]) / 40.0
    )

    context = sc.contexts.basic(
        checkpoint=checkpoint, target_stats=target_stats, risk_tolerance=risk_tolerance
    )

    min_mood = MOOD_FOR.get(config.MINIMUM_MOOD, MoodLevel.NORMAL)
    max_fail = config.MAX_FAILURE / 100.0
    career_profile = sc.profiles.basic(
        target_stats=target_stats,
        minimum_training_mood=min_mood,
        max_training_fail_rate=max_fail,
    )

    training_results = state_obj.get("training_results", {})
    actions = _build_train_actions(sc, training_results, scenario)
    actions.extend(_build_race_actions(sc, state_obj, year_string))
    actions.append(sc.actions.rest(action_id="rest"))
    if state_obj.get("date_event_available", False):
        actions.append(sc.actions.recreation(action_id="recreation"))

    if not actions:
        warning("[UmaLiteAdapter] No actions to evaluate")
        return None

    debug(
        f"[UmaLiteAdapter] {len(actions)} actions, checkpoint={checkpoint.kind.value} turns={checkpoint.turns_remaining}"
    )

    if scenario == "trackblazer":
        plan = sc.plan_turn(
            snapshot, actions, context=context, career_profile=career_profile
        )
        info(f"[UmaLiteAdapter] Plan: {plan.final_action_id}")
        for step in plan.preparation_steps:
            debug(f"[UmaLiteAdapter]   prep: {step.step_type} {step.step_id}")
        result = plan.evaluation
    else:
        result = sc.evaluate_turn(
            snapshot, actions, context=context, career_profile=career_profile
        )

    if result.best:
        info(
            f"[UmaLiteAdapter] Best: {result.best_action_id} (score={result.best.score:.2f})"
        )
        for bid in result.bids:
            debug(f"[UmaLiteAdapter]   {bid.action.action_id}: {bid.score:.2f}")

    return result


def umalite_action_to_bot_action(best_action_id: str) -> tuple[str, str | None]:
    if best_action_id.startswith("train_"):
        return "do_training", best_action_id[len("train_") :]
    if best_action_id.startswith("race_"):
        return "do_race", best_action_id[len("race_") :]
    if best_action_id == "rest":
        return "do_rest", None
    if best_action_id == "recreation":
        return "do_recreation", None
    return "do_training", None


def _get_scenario_name():
    return _resolve_scenario()


def resolve_policy(training_template: dict[str, Any]) -> Any:
    weight_set = training_template.get("stat_weight_set", {})

    class _P:
        stat_weights = {
            STAT_BOT_TO_LITE[k]: max(0.0, float(v))
            for k, v in weight_set.items()
            if k in STAT_BOT_TO_LITE
        }
        target_stats = _translate_stats(training_template.get("target_stat_set", {}))
        risk_taking = training_template.get(
            "risk_taking_set", {"rainbow_increase": 0, "normal_increase": 0}
        )
        risk_tolerance = min(
            1.0,
            (risk_taking["rainbow_increase"] + risk_taking["normal_increase"]) / 40.0,
        )
        max_failure_rate = config.MAX_FAILURE / 100.0
        min_mood = MOOD_FOR.get(config.MINIMUM_MOOD, MoodLevel.NORMAL)

    return _P()


def normalize_state(state_obj: dict[str, Any], _scenario: str = "ura") -> Any:
    year_string = state_obj.get("year", "")

    class _S:
        year_str = year_string
        turn_index = _turn_index(year_string)
        criteria = state_obj.get("criteria", "")

    return _S()


def derive_checkpoint(norm_state: Any) -> CheckpointContext:
    return _build_checkpoint(
        norm_state.year_str, norm_state.turn_index, norm_state.criteria
    )
