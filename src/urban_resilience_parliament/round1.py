"""Round-1 independent scoring runner."""

from pathlib import Path
from typing import Any

from .backend import LLMBackend, MockLLMBackend
from .io import agent_round_to_compact_dict, load_city_input, model_filename, rounds_to_compact_payload, write_json
from .personas import ModelSpec, default_models
from .prompts import build_r1_prompt
from .round2 import build_round2_payload, run_round2, write_round2_outputs
from .round2_packet import (
    build_human_audit_packet,
    build_model_facing_packet,
    write_round2_packet,
)
from .schemas import AgentRound, CityInput
from .validation import validate_agent_round, validate_round_payload


def run_round1(
    city: CityInput,
    personas: list[ModelSpec] | None = None,
    backend: LLMBackend | None = None,
) -> list[AgentRound]:
    """Run independent R1 scoring for one city across multiple mock agents."""

    active_personas = personas or default_models()
    active_backend = backend or MockLLMBackend()
    rounds = [
        active_backend.generate_round1(
            city=city,
            persona=persona,
            prompt=build_r1_prompt(city, persona),
        )
        for persona in active_personas
    ]
    for agent_round in rounds:
        validate_agent_round(agent_round)
    return rounds


def build_round1_payload(city: CityInput, rounds: list[AgentRound]) -> dict[str, Any]:
    """Build the JSON shape saved by the R1 demo."""

    return rounds_to_compact_payload(
        city=city,
        rounds=rounds,
        round_number=1,
        stage="independent_scoring",
    )


def write_round1_outputs(
    round1_dir: str | Path,
    city: CityInput,
    rounds: list[AgentRound],
    payload: dict[str, Any],
) -> None:
    """Write all R1 audit files: one summary plus one JSON per agent."""

    output_dir = Path(round1_dir)
    stale_summary = output_dir / "all_agents.json"
    if stale_summary.exists():
        stale_summary.unlink()
    write_json(output_dir / "all_models.json", payload)
    for agent_round in rounds:
        write_json(
            output_dir / model_filename(agent_round.agent_id),
            agent_round_to_compact_dict(agent_round, city, "independent_scoring"),
        )


def run_round1_from_files(input_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Load a city input, run R1, save packets, and run deterministic R2."""

    city = load_city_input(input_path)
    rounds = run_round1(city)
    payload = build_round1_payload(city, rounds)
    validate_round_payload(payload)
    run_path = Path(run_dir)
    write_round1_outputs(run_path / "round1", city, rounds, payload)
    model_packet = build_model_facing_packet(payload)
    write_round2_packet(run_path / "round2" / "packet.txt", model_packet)
    write_round2_packet(
        run_path / "round2" / "human_audit_packet.txt",
        build_human_audit_packet(payload),
    )
    second_rounds = run_round2(city, rounds, model_packet)
    round2_payload = build_round2_payload(city, second_rounds)
    validate_round_payload(round2_payload)
    write_round2_outputs(run_path / "round2", city, second_rounds, round2_payload)
    from .consul import run_consul_from_files, write_consul_outputs

    consul_report = run_consul_from_files(
        run_path / "round1" / "all_models.json",
        run_path / "round2" / "all_models.json",
    )
    write_consul_outputs(run_path, consul_report)
    for stale_path in (
        run_path / "final_aggregate.json",
        run_path / "round2" / "model_facing_packet.txt",
    ):
        if stale_path.exists():
            stale_path.unlink()
    return payload
