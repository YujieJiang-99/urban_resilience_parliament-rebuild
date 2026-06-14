"""Prompt builders for the parliament workflow."""

from .calibration import calibration_bracket_hint
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
        "For each indicator, output a score from 0.0 to 1.0 and a useful reasoning field.",
        "Reasoning must be 1-2 informative sentences and must include:",
        "(1) one city factual judgment;",
        "(2) at least one city_anchors_resilience comparison using the exact phrase",
        "'anchored against', for example 'anchored against Singapore=1.0000 / Tokyo=0.9474';",
        "(3) an explicit calibration bracket using this exact format:",
        "'Calibration: lower_anchor=<city>=<value>, upper_anchor=<city>=<value>';",
        f"(4) if {city.city_name} appears in the anchors, explicitly write ",
        f"'target city appears in anchors: {city.city_name}=<value>'.",
        "The score must be consistent with the calibration bracket:",
        f"- if {city.city_name} appears in city_anchors_resilience, score should be within 0.05 of that anchor value;",
        "- otherwise, score should lie between the two stated bracket anchors;",
        "- if you intentionally score outside the bracket, include 'deviation_reason=...' in reasoning.",
        "Do not use placeholders such as 'short reason' or 'N/A'.",
        "Return only valid JSON with this shape:",
        '{"indicators":{"<indicator_id>":{"score":<number>,"reasoning":"<informative 1-2 sentence justification>"}}}',
        "",
        "Indicators:",
    ]
    for indicator in RESILIENCE_INDICATORS:
        meta = get_indicator_meta(indicator)
        target_anchor = meta.city_anchors_resilience.get(city.city_name)
        target_anchor_line = (
            f"target_city_anchor: {city.city_name}={target_anchor:.4f}"
            if target_anchor is not None
            else f"target_city_anchor: {city.city_name} not present in anchors"
        )
        target_clause = (
            f"must include exact text: target city appears in anchors: {city.city_name}={target_anchor:.4f}"
            if target_anchor is not None
            else "do not claim the target city appears in anchors"
        )
        lines.extend(
            [
                f"- {meta.alias_name}",
                f"  id: {indicator}",
                f"  dimension: {meta.dimension}",
                f"  definition: {meta.city_level_definition}",
                f"  city_anchors_resilience: {_format_anchors(meta.city_anchors_resilience)}",
                f"  {target_anchor_line}",
                f"  calibration_bracket_hint: {calibration_bracket_hint(indicator, city.city_name)}",
                "  reasoning_requirements:",
                "    - include one factual judgment about the city",
                "    - include exact phrase 'anchored against' with at least one anchor comparison",
                "    - include exact bracket format: Calibration: lower_anchor=<city>=<value>, upper_anchor=<city>=<value>",
                "    - keep score inside the bracket unless reasoning includes deviation_reason=...",
                f"    - {target_clause}",
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
        "Scores must be from 0.0 to 1.0.",
        "Return only valid JSON. Do not wrap it in markdown.",
        "Use this schema, replacing the reasoning text with actual evidence-based reasoning:",
        '{"indicators":{"<indicator_id>":{"score":<number>,"reasoning":"<informative justification>"}}}',
        "Reasoning must not be a placeholder and must include anchor calibration with the phrase 'anchored against'.",
        "Reasoning must include: Calibration: lower_anchor=<city>=<value>, upper_anchor=<city>=<value>.",
        "",
        "Score only these indicators:",
    ]
    for indicator in indicator_ids:
        meta = get_indicator_meta(indicator)
        target_anchor = meta.city_anchors_resilience.get(city.city_name)
        target_anchor_line = (
            f"target_city_anchor: {city.city_name}={target_anchor:.4f}"
            if target_anchor is not None
            else f"target_city_anchor: {city.city_name} not present in anchors"
        )
        lines.extend(
            [
                f"- {meta.alias_name}",
                f"  id: {indicator}",
                f"  dimension: {meta.dimension}",
                f"  definition: {meta.city_level_definition}",
                f"  city_anchors_resilience: {_format_anchors(meta.city_anchors_resilience)}",
                f"  {target_anchor_line}",
                f"  calibration_bracket_hint: {calibration_bracket_hint(indicator, city.city_name)}",
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
