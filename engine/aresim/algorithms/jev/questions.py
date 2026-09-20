"""Build the single masked Choice question Jev answers each step.

Instructions are literal survival/resource rules. Criteria are only legal
actions, each citing the matching 5x5 cell.

**Last updated:** September 20, 2026

**Contains:** :func:`build_action_question`.
"""

from __future__ import annotations

from typing import Mapping


ACTION_INSTRUCTIONS: dict[str, object] = {
    "question": "Which one legal rover action should be taken right now?",
    "rules": [
        "Choose only from the options below. Each option is already legal.",
        "Wait only recharges the colony battery when rover.wait_recharges is true (colony.power_margin > 0). If wait_recharges is false, do not choose wait; pick a move, extract, scan, unload, build, or service instead.",
        "Prefer survival if battery_pct or health_pct is low: move toward the pad or ice, do not idle off-surplus.",
        "If should_unload is true, go to the pad and unload: the bay is full or you are already on the pad with cargo. Unload dumps ice, ore, and samples together. If free_kg is still positive and you are not on the pad, keep extracting or scanning.",
        "If extract is legal, extract this tick. Do not walk off ice to chase a richer cell.",
        "If scan is legal and extract is not, scan this tick.",
        "prospects.best_ice is the richest ice cell in the 5x5 (ice_pct, dx, dy, steps). If extract is not legal, free_kg > 0, and best_ice exists, take first_move.",
        "prospects.best_ore is the same for ore when no ice is visible. Move options with toward_best_ice or toward_best_ore are the first step toward that cell.",
        "Move options name the destination cell with dx, dy, terrain, and ice/ore percents.",
        "recent_actions lists the last 10 actions, oldest first.",
        "If recent_repeating is true, or recent_actions show the same move repeating or a back-and-forth, do not continue that pattern: pick a different direction, leave this region, and explore a new area of the 5x5.",
        "Do not reverse the last move in recent_actions unless every other direction is blocked or a higher-priority action requires it.",
    ],
}


def build_action_question(legal_criteria: Mapping[str, object]) -> dict[str, object]:
    """Return one Choice question keyed as ``action``."""
    if not legal_criteria:
        raise ValueError("Choice criteria cannot be empty")
    return {
        "action": {
            "type": "choice",
            "instructions": ACTION_INSTRUCTIONS,
            "criteria": dict(legal_criteria),
        }
    }


__all__ = ["ACTION_INSTRUCTIONS", "build_action_question"]
