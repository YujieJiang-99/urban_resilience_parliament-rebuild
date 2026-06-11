"""Validation helpers for compact research-log outputs."""

from typing import Any

from .indicators import EXPECTED_INDICATOR_COUNT, RESILIENCE_INDICATORS
from .schemas import AgentRound


def validate_agent_round(agent_round: AgentRound) -> None:
    """Validate one in-memory agent round."""

    indicators = {score.indicator for score in agent_round.scores}
    if indicators != set(RESILIENCE_INDICATORS):
        raise ValueError("agent round does not cover the metadata indicator set")
    if len(agent_round.scores) != EXPECTED_INDICATOR_COUNT:
        raise ValueError("agent round must include 18 indicators")
    for score in agent_round.scores:
        if not 0 <= score.score <= 100:
            raise ValueError(f"score out of range for {score.indicator}")
        if not score.rationale:
            raise ValueError(f"missing reasoning for {score.indicator}")


def validate_round_payload(payload: dict[str, Any]) -> None:
    """Validate compact round JSON payload shape."""

    expected = set(RESILIENCE_INDICATORS)
    model_items = payload.get("models", payload.get("agents", {}))
    for model_name, agent_data in model_items.items():
        if agent_data.get("model", agent_data.get("agent_id")) != model_name:
            raise ValueError(f"model key mismatch for {model_name}")
        indicators = agent_data.get("indicators", {})
        if set(indicators) != expected:
            raise ValueError(f"{model_name} does not cover the metadata indicator set")
        for indicator, cell in indicators.items():
            if set(cell) != {"score", "reasoning"}:
                raise ValueError(f"invalid indicator cell for {model_name} / {indicator}")
            if not 0 <= cell["score"] <= 100:
                raise ValueError(f"score out of range for {model_name} / {indicator}")
            if not cell["reasoning"]:
                raise ValueError(f"missing reasoning for {model_name} / {indicator}")
