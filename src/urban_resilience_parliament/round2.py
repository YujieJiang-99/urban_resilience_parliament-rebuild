"""Round-2 peer-aware deliberation runner."""

from pathlib import Path
from typing import Any

from .backend import LLMBackend, MockLLMBackend
from .io import agent_round_to_compact_dict, model_filename, rounds_to_compact_payload, write_json
from .personas import ModelSpec, default_models
from .prompts import build_r2_prompt
from .schemas import AgentRound, CityInput
from .validation import validate_agent_round


def run_round2(
    city: CityInput,
    first_rounds: list[AgentRound],
    model_packet: str,
    personas: list[ModelSpec] | None = None,
    backend: LLMBackend | None = None,
) -> list[AgentRound]:
    """Run R2 scoring through persona registry, prompt builder, and backend."""

    if not model_packet:
        raise ValueError("model-facing packet is required for R2 deliberation")

    active_personas = personas or default_models()
    active_backend = backend or MockLLMBackend()
    rounds = [
        active_backend.generate_round2(
            city=city,
            persona=persona,
            prompt=build_r2_prompt(city, persona, model_packet),
            first_rounds=first_rounds,
        )
        for persona in active_personas
    ]
    for agent_round in rounds:
        validate_agent_round(agent_round)
    return rounds


def build_round2_payload(city: CityInput, rounds: list[AgentRound]) -> dict[str, Any]:
    """Build the JSON shape saved by the R2 demo."""

    return rounds_to_compact_payload(
        city=city,
        rounds=rounds,
        round_number=2,
        stage="peer_aware_deliberation",
    )


def write_round2_outputs(
    round2_dir: str | Path,
    city: CityInput,
    rounds: list[AgentRound],
    payload: dict[str, Any],
) -> None:
    """Write all R2 audit files: one summary plus one JSON per agent."""

    output_dir = Path(round2_dir)
    stale_summary = output_dir / "all_agents.json"
    if stale_summary.exists():
        stale_summary.unlink()
    write_json(output_dir / "all_models.json", payload)
    for agent_round in rounds:
        write_json(
            output_dir / model_filename(agent_round.agent_id),
            agent_round_to_compact_dict(agent_round, city, "peer_aware_deliberation"),
        )
