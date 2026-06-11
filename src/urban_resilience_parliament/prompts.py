"""Prompt builders for the parliament workflow."""

from .indicators import RESILIENCE_INDICATORS, get_indicator_meta
from .personas import ModelSpec
from .schemas import CityInput


RESILIENCE_VIEW_INSTRUCTION = (
    "All indicators are scored in resilience view: higher = more resilient."
)


def build_r1_prompt(city: CityInput, model: ModelSpec) -> str:
    """Build a round-1 independent scoring prompt."""

    lines = [
        f"City: {city.city_name}",
        f"Region: {city.country_or_region}",
        f"Summary: {city.summary}",
        f"Model: {model.model_name}",
        RESILIENCE_VIEW_INSTRUCTION,
        "For each indicator, output a score from 0 to 100 and a useful reasoning field.",
        "Reasoning must be 1-2 informative sentences that reference city evidence,",
        "the indicator definition, or city_anchors_resilience calibration.",
        "Do not use placeholders such as 'short reason' or 'N/A'.",
        "Return only valid JSON with this shape:",
        '{"indicators":{"<indicator_id>":{"score":<number>,"reasoning":"<informative 1-2 sentence justification>"}}}',
        "",
        "Indicators:",
    ]
    for indicator in RESILIENCE_INDICATORS:
        meta = get_indicator_meta(indicator)
        lines.extend(
            [
                f"- {meta.alias_name}",
                f"  id: {indicator}",
                f"  dimension: {meta.dimension}",
                f"  definition: {meta.city_level_definition}",
                f"  city_anchors_resilience: {_format_anchors(meta.city_anchors_resilience)}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_r2_prompt(city: CityInput, model: ModelSpec, packet: str) -> str:
    """Build a round-2 peer-aware deliberation prompt."""

    return "\n".join(
        [
            f"City: {city.city_name}",
            f"Model: {model.model_name}",
            RESILIENCE_VIEW_INSTRUCTION,
            "Use the deliberation packet below to revise scores only when warranted.",
            "",
            packet.rstrip(),
            "",
        ]
    )


def build_r1_smoke_prompt(
    city: CityInput,
    model: ModelSpec,
    indicator_ids: list[str],
) -> str:
    """Build a tiny R1 prompt for API smoke testing."""

    lines = [
        f"City: {city.city_name}",
        f"Region: {city.country_or_region}",
        f"Summary: {city.summary}",
        f"Model: {model.model_name}",
        RESILIENCE_VIEW_INSTRUCTION,
        "Return only valid JSON. Do not wrap it in markdown.",
        "Use this schema, replacing the reasoning text with actual evidence-based reasoning:",
        '{"indicators":{"<indicator_id>":{"score":<number>,"reasoning":"<informative justification>"}}}',
        "Reasoning must not be a placeholder.",
        "",
        "Score only these indicators:",
    ]
    for indicator in indicator_ids:
        meta = get_indicator_meta(indicator)
        lines.extend(
            [
                f"- {meta.alias_name}",
                f"  id: {indicator}",
                f"  dimension: {meta.dimension}",
                f"  definition: {meta.city_level_definition}",
                f"  city_anchors_resilience: {_format_anchors(meta.city_anchors_resilience)}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _format_anchors(anchors: dict[str, float], limit: int = 6) -> str:
    if not anchors:
        return "none"
    preferred = ["Hong Kong", "Singapore", "Tokyo", "London", "Niamey", "Kabul"]
    selected: list[tuple[str, float]] = [
        (city, anchors[city]) for city in preferred if city in anchors
    ]
    selected_names = {name for name, _ in selected}
    for city, value in anchors.items():
        if len(selected) >= limit:
            break
        if city not in selected_names:
            selected.append((city, value))
            selected_names.add(city)
    return ", ".join(f"{city}={value:.4f}" for city, value in selected[:limit])
