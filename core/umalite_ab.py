from __future__ import annotations

import json
import os
import time
from typing import Any

import utils.constants as constants
from core.umalite_adapter import (
    _get_scenario_name,
    derive_checkpoint,
    evaluate_with_umalite,
    normalize_state,
    resolve_policy,
)
from utils import log as log_module
from utils.log import error, info, warning

AB_RECORD_KEY = "umalite_ab"
_career_id: str | None = None
_career_log_path: str | None = None


def _action_options(action: Any) -> dict[str, Any]:
    return getattr(action, "options", {})


def _action_id(action: Any) -> str:
    func = getattr(action, "func", None)
    options = _action_options(action)

    if func == "do_training":
        return f"train_{options.get('training_name', 'unknown')}"
    if func == "do_race":
        race_name = options.get("race_name")
        if race_name and race_name not in ("", "any"):
            return f"race_{race_name}"
        if options.get("race_mission_available"):
            return "race_mission"
        if options.get("is_race_day"):
            return "race_day"
        return "race_any"
    if func == "do_rest":
        return "rest"
    if func == "do_recreation":
        return "recreation"
    if func == "do_infirmary":
        return "infirmary"
    if func:
        return str(func)
    return "unknown"


def _action_family(action_id: str) -> str:
    if action_id.startswith("train_"):
        return "train"
    if action_id.startswith("race_") or action_id in {"race_day", "race_any", "race_mission"}:
        return "race"
    return action_id


def _projected_trainings(state: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    projected = {}
    all_projected = True

    for name, data in state.get("training_results", {}).items():
        gains = data.get("stat_gains")
        is_projected = isinstance(gains, dict) and bool(gains)
        if not is_projected:
            all_projected = False
        projected[name] = {
            "stat_gains": gains,
            "failure": data.get("failure"),
            "projection_mode": "projected" if is_projected else "heuristic",
        }

    return ("projected" if all_projected else "heuristic"), projected


def _legacy_training_scores(action: Any) -> dict[str, float]:
    scores = {}
    for name, data in _action_options(action).get("available_trainings", {}).items():
        score_tuple = data.get("score_tuple")
        if score_tuple:
            scores[f"train_{name}"] = float(score_tuple[0])
    return scores


def _umalite_bids(result: Any) -> list[dict[str, Any]]:
    bids = []
    for bid in result.bids:
        bids.append(
            {
                "action_id": bid.action.action_id,
                "action_family": _action_family(bid.action.action_id),
                "score": bid.score,
                "breakdown": bid.breakdown,
            }
        )
    return bids


def _resolved_policy(training_template: dict[str, Any]) -> dict[str, Any]:
    policy = resolve_policy(training_template)
    min_mood = policy.min_mood
    return {
        "stat_weights": policy.stat_weights,
        "target_stats": policy.target_stats,
        "risk_tolerance": policy.risk_tolerance,
        "max_failure_rate": policy.max_failure_rate,
        "min_mood": min_mood.name if hasattr(min_mood, "name") else str(min_mood),
    }


def _log_dir() -> str:
    resolved = getattr(log_module, "log_dir", None)
    if resolved:
        return resolved
    return os.path.join(os.getcwd(), "logs")


def _career_dir() -> str:
    return os.path.join(_log_dir(), "umalite_ab")


def _new_career_id() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    return f"career_{timestamp}_{millis:03d}"


def start_umalite_ab_career() -> str:
    global _career_id, _career_log_path

    _career_id = _new_career_id()
    os.makedirs(_career_dir(), exist_ok=True)
    _career_log_path = os.path.join(_career_dir(), f"{_career_id}.jsonl")
    return _career_id


def _ensure_career_log() -> tuple[str, str]:
    global _career_id, _career_log_path

    if _career_id is None or _career_log_path is None:
        start_umalite_ab_career()
    return _career_id, _career_log_path


def prepare_umalite_ab(
    state: dict[str, Any],
    training_template: dict[str, Any],
    action: Any,
) -> None:
    try:
        result = evaluate_with_umalite(state, training_template)
        if not result or not result.best:
            warning("[UmaLite A/B] UmaLite returned no valid actions.")
            return

        norm_state = normalize_state(state, _get_scenario_name())
        checkpoint = derive_checkpoint(norm_state)
        projection_mode, projected_trainings = _projected_trainings(state)
        selected_action_id = _action_id(action)
        umalite_action_id = result.best_action_id
        selected_exact_match = selected_action_id == umalite_action_id
        selected_family_match = _action_family(selected_action_id) == _action_family(
            umalite_action_id
        )

        record = {
            "timestamp": time.time(),
            "career_id": _ensure_career_log()[0],
            "year": state.get("year", ""),
            "turn_index": state.get("turn", -1),
            "energy": state.get("energy_level", 0),
            "mood": state.get("current_mood", ""),
            "scenario": constants.SCENARIO_NAME,
            "current_stats": state.get("current_stats", {}),
            "turn_projection_mode": projection_mode,
            "checkpoint": {
                "kind": checkpoint.kind.value,
                "turns_remaining": checkpoint.turns_remaining,
            },
            "resolved_policy": _resolved_policy(training_template),
            "projected_trainings": projected_trainings,
            "legacy_selected_action_id": selected_action_id,
            "legacy_selected_action_family": _action_family(selected_action_id),
            "legacy_selected_score": (
                action["training_data"]["score_tuple"][0]
                if _action_options(action).get("training_data")
                and _action_options(action)["training_data"].get("score_tuple")
                else None
            ),
            "legacy_training_scores": _legacy_training_scores(action),
            "umalite_best_action_id": umalite_action_id,
            "umalite_best_action_family": _action_family(umalite_action_id),
            "selected_exact_match": selected_exact_match,
            "selected_family_match": selected_family_match,
            "umalite_bids": _umalite_bids(result),
        }

        _action_options(action)[AB_RECORD_KEY] = record

        selected_marker = "MATCH" if selected_exact_match else "MISMATCH"
        info(f"[UmaLite A/B] {selected_marker} | Selected: {selected_action_id}")
        info(
            f"  > UmaLite Best: {umalite_action_id} ({result.best.score:.2f})"
        )
        top_bids = [f"{bid['action_id']}({bid['score']:.1f})" for bid in record["umalite_bids"][:3]]
        info(f"  > Top Bids: {' | '.join(top_bids)}")
    except Exception as exc:
        error(f"[UmaLite A/B] Exception during evaluation: {exc}")


def record_umalite_ab(action: Any) -> None:
    record = _action_options(action).pop(AB_RECORD_KEY, None)
    if not record:
        return

    executed_action_id = _action_id(action)
    executed_action_family = _action_family(executed_action_id)
    umalite_action_id = record["umalite_best_action_id"]
    umalite_action_family = record["umalite_best_action_family"]

    record["legacy_executed_action_id"] = executed_action_id
    record["legacy_executed_action_family"] = executed_action_family
    record["executed_exact_match"] = executed_action_id == umalite_action_id
    record["executed_family_match"] = executed_action_family == umalite_action_family
    record["execution_fallback_used"] = (
        record["legacy_selected_action_id"] != executed_action_id
    )

    _, log_path = _ensure_career_log()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
