"""JSON helpers for the minimal demo."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .schemas import AgentRound, CityInput, IndicatorScore


def model_filename(model_name: str) -> str:
    """Return a stable JSON filename for one model."""

    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_name)
    return f"model_{safe}.json"


def load_city_input(path: str | Path) -> CityInput:
    """Load a city input JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CityInput(
        city_id=data["city_id"],
        city_name=data["city_name"],
        country_or_region=data["country_or_region"],
        summary=data["summary"],
        evidence=data.get("evidence", {}),
    )


def agent_round_to_dict(agent_round: AgentRound) -> dict[str, Any]:
    """Serialize an agent round dataclass to plain JSON-compatible data."""

    return asdict(agent_round)


def agent_round_to_compact_dict(
    agent_round: AgentRound,
    city: CityInput,
    stage: str,
    validation_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize one agent round as a compact research-log JSON object."""

    payload = {
        "city_id": city.city_id,
        "city_name": city.city_name,
        "round": agent_round.round_number,
        "stage": stage,
        "model": agent_round.agent_id,
        "indicators": {
            score.indicator: {
                "score": score.score,
                "reasoning": score.rationale,
            }
            for score in agent_round.scores
        },
    }
    if validation_warnings is not None:
        payload["validation_warnings"] = validation_warnings
    return payload


def rounds_to_compact_payload(
    city: CityInput,
    rounds: list[AgentRound],
    round_number: int,
    stage: str,
) -> dict[str, Any]:
    """Serialize several agent rounds as a compact machine-readable payload."""

    return {
        "city_id": city.city_id,
        "city_name": city.city_name,
        "round": round_number,
        "stage": stage,
        "models": {
            agent_round.agent_id: agent_round_to_compact_dict(agent_round, city, stage)
            for agent_round in rounds
        },
    }


def agent_round_from_dict(data: dict[str, Any]) -> AgentRound:
    """Parse an agent round from JSON-compatible data."""

    if "indicators" in data:
        return AgentRound(
            agent_id=data.get("model", data.get("agent_id")),
            round_number=data["round"],
            scores=[
                IndicatorScore(
                    indicator=indicator,
                    score=item["score"],
                    rationale=item["reasoning"],
                )
                for indicator, item in data["indicators"].items()
            ],
        )

    return AgentRound(
        agent_id=data["agent_id"],
        round_number=data["round_number"],
        scores=[
            IndicatorScore(
                indicator=item["indicator"],
                score=item["score"],
                rationale=item["rationale"],
                confidence=item.get("confidence"),
            )
            for item in data["scores"]
        ],
        notes=data.get("notes", ""),
    )


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write pretty JSON, creating parent directories as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
