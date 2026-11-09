"""Load readable, versioned role templates and inject only public game requirements.

Custom templates are configuration inputs, frozen into bundles and exact call logs.
Required tokens keep game rules and the output boundary present in every template.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import EnvironmentConfig
from .config import RoleConfig
from .state import public_requirements


def render_prompt(
    role: str,
    config: RoleConfig,
    environment: EnvironmentConfig,
    *,
    focused: bool = False,
) -> str:
    """Render an Actor or Learner system prompt without hidden state or secrets."""
    if role not in {"actor", "learner", "goal"}:
        raise ValueError("unknown prompt role")
    custom = config.goal_prompt_path if role == "goal" else config.prompt_path
    filename = "actor_focused" if role == "actor" and focused else role
    path = (
        Path(custom)
        if custom
        else Path(__file__).with_name("prompts") / f"{filename}.txt"
    )
    template = path.read_text(encoding="utf-8")
    if role in {"actor", "goal"}:
        if (
            config.provider == "decision_models"
            and template.count("{{OUTPUT_REQUIREMENTS}}") != 1
        ):
            raise ValueError(
                "Decision Model Actor template requires one {{OUTPUT_REQUIREMENTS}} token"
            )
        output = (
            "Return exactly one TypeSafe Choice for the question named action. Select one legal criterion key. "
            "Do not return a JSON action object, a rationale or an action sequence. The adapter supplies decision identity fields."
            if config.provider == "decision_models"
            else "Return one JSON object matching aresim.llm.action.v1 with ONLY schema_version, episode_id, decision_id, step_index and action. "
            "Echo episode_id, decision_id and step_index exactly. No Markdown, rationale, confidence, or extra fields."
        )
        if role == "goal":
            output = (
                "Select exactly one TypeSafe goal Choice from the supplied goal IDs."
                if config.provider == "decision_models"
                else "Return only aresim.llm_dm.goal.v1 JSON: schema_version, episode_id, decision_id, step_index, goal_id. Echo identities exactly; goal_id must be a supplied option."
            )
        template = template.replace(
            "{{OUTPUT_REQUIREMENTS}}",
            output + " Do not claim success: use simulator feedback on the next turn.",
        )
    for token in ("{{GAME_REQUIREMENTS}}", "{{ROLE_GUIDANCE}}"):
        if template.count(token) != 1:
            raise ValueError(f"role template must contain exactly one {token}")
    rendered = template.replace(
        "{{GAME_REQUIREMENTS}}",
        json.dumps(
            public_requirements(environment),
            ensure_ascii=False,
            indent=None if focused else 2,
            separators=(",", ":") if focused else None,
        ),
    )
    return rendered.replace(
        "{{ROLE_GUIDANCE}}", config.guidance or "No additional experiment guidance."
    )
