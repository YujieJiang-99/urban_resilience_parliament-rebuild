from urban_resilience_parliament import (
    RESILIENCE_INDICATORS,
    AgentRound,
    IndicatorScore,
    aggregate_parliamentary_score,
)


def make_scores(value: float) -> list[IndicatorScore]:
    return [
        IndicatorScore(
            indicator=indicator,
            score=value,
            rationale="placeholder rationale",
            confidence=0.5,
        )
        for indicator in RESILIENCE_INDICATORS
    ]


def test_agent_round_requires_all_18_indicators() -> None:
    agent_round = AgentRound(
        agent_id="agent_a",
        round_number=1,
        scores=make_scores(0.7),
    )

    assert len(agent_round.scores) == 18


def test_placeholder_aggregation_averages_second_round_scores() -> None:
    rounds = [
        AgentRound("agent_a", 2, make_scores(0.6)),
        AgentRound("agent_b", 2, make_scores(0.8)),
    ]

    indicator_scores, parliamentary_score = aggregate_parliamentary_score(rounds)

    assert set(indicator_scores) == set(RESILIENCE_INDICATORS)
    assert parliamentary_score == 0.7
