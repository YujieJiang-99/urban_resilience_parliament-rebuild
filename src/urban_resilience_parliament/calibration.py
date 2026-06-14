"""Anchor calibration checks for model-produced R1 scores."""

from __future__ import annotations

import re
from typing import Any

from .indicators import get_indicator_meta


CALIBRATION_RE = re.compile(
    r"Calibration:\s*lower_anchor=(?P<lower_city>[^=,]+)="
    r"(?P<lower_value>[0-9]*\.?[0-9]+)\s*,\s*upper_anchor="
    r"(?P<upper_city>[^=,]+)=(?P<upper_value>[0-9]*\.?[0-9]+)"
)


def calibration_bracket_hint(indicator: str, target_city: str) -> str:
    """Return a suggested lower/upper anchor bracket for prompt context."""

    anchors = get_indicator_meta(indicator).city_anchors_resilience
    if not anchors:
        return "not available"

    ordered = sorted(anchors.items(), key=lambda item: item[1])
    if target_city in anchors:
        target_value = anchors[target_city]
        lower = ordered[0]
        upper = ordered[-1]
        for city, value in ordered:
            if value <= target_value and city != target_city:
                lower = (city, value)
            if value >= target_value and city != target_city:
                upper = (city, value)
                break
        return _format_bracket(lower, upper)

    lower, upper = _representative_middle_bracket(ordered)
    return _format_bracket(lower, upper)


def validate_calibration_reasoning(indicator: str, reasoning: str, target_city: str) -> None:
    """Fail fast when a real-model reasoning omits required calibration text."""

    if "anchored against" not in reasoning:
        raise ValueError(f"Indicator {indicator} reasoning must include anchor calibration")
    if parse_calibration_bracket(reasoning) is None:
        raise ValueError(
            f"Indicator {indicator} reasoning must include "
            "Calibration: lower_anchor=<city>=<value>, upper_anchor=<city>=<value>"
        )

    anchors = get_indicator_meta(indicator).city_anchors_resilience
    if target_city in anchors:
        required = f"target city appears in anchors: {target_city}="
        if required not in reasoning:
            raise ValueError(
                f"Indicator {indicator} reasoning must disclose target city anchor"
            )


def calibration_warnings_for_indicators(
    indicators: dict[str, dict[str, Any]],
    target_city: str,
) -> list[dict[str, Any]]:
    """Return non-fatal warnings for scores that conflict with anchor calibration."""

    warnings: list[dict[str, Any]] = []
    for indicator, cell in indicators.items():
        score = float(cell["score"])
        reasoning = str(cell["reasoning"])
        anchors = get_indicator_meta(indicator).city_anchors_resilience
        bracket = parse_calibration_bracket(reasoning)

        if target_city in anchors:
            anchor_value = anchors[target_city]
            delta = abs(score - anchor_value)
            if delta > 0.08:
                warnings.append(
                    {
                        "indicator": indicator,
                        "warning": "target_anchor_score_mismatch",
                        "score": round(score, 4),
                        "target_anchor": round(anchor_value, 4),
                        "delta": round(delta, 4),
                        "message": (
                            f"score differs from {target_city} anchor by more than 0.08"
                        ),
                    }
                )

        if bracket is None:
            continue
        low = min(bracket["lower_value"], bracket["upper_value"])
        high = max(bracket["lower_value"], bracket["upper_value"])
        if not low <= score <= high and "deviation_reason=" not in reasoning:
            warnings.append(
                {
                    "indicator": indicator,
                    "warning": "score_outside_calibration_bracket",
                    "score": round(score, 4),
                    "lower_anchor": {
                        "city": bracket["lower_city"],
                        "value": round(bracket["lower_value"], 4),
                    },
                    "upper_anchor": {
                        "city": bracket["upper_city"],
                        "value": round(bracket["upper_value"], 4),
                    },
                    "message": (
                        "score is outside the stated calibration bracket and no "
                        "deviation_reason=... was provided"
                    ),
                }
            )

    return warnings


def parse_calibration_bracket(reasoning: str) -> dict[str, float | str] | None:
    """Extract the first required calibration bracket from reasoning text."""

    match = CALIBRATION_RE.search(reasoning)
    if match is None:
        return None
    return {
        "lower_city": match.group("lower_city").strip(),
        "lower_value": float(match.group("lower_value")),
        "upper_city": match.group("upper_city").strip(),
        "upper_value": float(match.group("upper_value")),
    }


def _representative_middle_bracket(
    ordered: list[tuple[str, float]],
) -> tuple[tuple[str, float], tuple[str, float]]:
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    midpoint = len(ordered) // 2
    return ordered[max(0, midpoint - 1)], ordered[min(len(ordered) - 1, midpoint)]


def _format_bracket(
    lower: tuple[str, float],
    upper: tuple[str, float],
) -> str:
    return (
        f"Calibration: lower_anchor={lower[0]}={lower[1]:.4f}, "
        f"upper_anchor={upper[0]}={upper[1]:.4f}"
    )
