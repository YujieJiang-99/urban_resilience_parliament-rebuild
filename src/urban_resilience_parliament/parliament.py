"""Deterministic placeholder aggregation for the parliament workflow."""

from .indicators import RESILIENCE_INDICATORS
from .schemas import AgentRound


def aggregate_parliamentary_score(second_rounds: list[AgentRound]) -> tuple[dict[str, float], float]:
    """Average second-round agent scores into indicator and total scores.

    This is intentionally simple. A future version can add referee adjustments,
    robust aggregation, persona weights, or uncertainty intervals.
    """

    if not second_rounds:
        raise ValueError("at least one second-round agent assessment is required")

    indicator_scores: dict[str, float] = {}
    for indicator in RESILIENCE_INDICATORS:
        values = [
            score.score
            for agent_round in second_rounds
            for score in agent_round.scores
            if score.indicator == indicator
        ]
        if not values:
            raise ValueError(f"missing score for indicator: {indicator}")
        indicator_scores[indicator] = round(sum(values) / len(values), 2)

    parliamentary_score = round(
        sum(indicator_scores.values()) / len(indicator_scores),
        2,
    )
    return indicator_scores, parliamentary_score
