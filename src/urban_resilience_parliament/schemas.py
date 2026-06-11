"""Small dataclass schemas for the parliamentary assessment workflow."""

from dataclasses import dataclass, field
from typing import Any

from .indicators import EXPECTED_INDICATOR_COUNT, RESILIENCE_INDICATORS


@dataclass(frozen=True)
class CityInput:
    """Input shape for one city case."""

    city_id: str
    city_name: str
    country_or_region: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndicatorScore:
    """A score and short explanation for one resilience indicator."""

    indicator: str
    score: float
    rationale: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.indicator not in RESILIENCE_INDICATORS:
            raise ValueError(f"Unknown indicator: {self.indicator}")
        if not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class AgentRound:
    """One agent's scoring round."""

    agent_id: str
    round_number: int
    scores: list[IndicatorScore]
    notes: str = ""

    def __post_init__(self) -> None:
        indicators = {score.indicator for score in self.scores}
        if len(indicators) != EXPECTED_INDICATOR_COUNT:
            raise ValueError("each agent round must include 18 unique indicators")


@dataclass(frozen=True)
class RefereeReview:
    """Referee checks after agent deliberation."""

    flagged_items: list[str] = field(default_factory=list)
    adjustment_notes: str = ""


@dataclass(frozen=True)
class ParliamentaryAssessment:
    """Final output shape for the B-1 module."""

    city_id: str
    first_rounds: list[AgentRound]
    second_rounds: list[AgentRound]
    referee_review: RefereeReview
    indicator_scores: dict[str, float]
    parliamentary_score: float
