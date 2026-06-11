"""Build deliberation packets from compact round JSON payloads."""

from pathlib import Path
from statistics import pstdev
from typing import Any

from .indicators import RESILIENCE_INDICATORS, get_indicator_meta


def anonymous_agent_labels(agent_ids: list[str]) -> dict[str, str]:
    """Map internal agent IDs to stable anonymous labels for the packet."""

    return {
        agent_id: f"Agent_{chr(ord('A') + index)}"
        for index, agent_id in enumerate(agent_ids)
    }


def build_model_facing_packet(
    round_payload: dict[str, Any],
    flags: list[dict] | None = None,
) -> str:
    """Create an anonymous packet intended for R2 model deliberation."""

    agent_ids = list(round_payload["models"])
    return _build_packet(
        round_payload=round_payload,
        labels=anonymous_agent_labels(agent_ids),
        heading="Round 2 deliberation packet",
        task_line=(
            "All scores are resilience view: higher = more resilient. "
            "Use anonymous peer scores and reasoning to revise each indicator if warranted."
        ),
        flags=flags or [],
    )


def build_human_audit_packet(
    round_payload: dict[str, Any],
    flags: list[dict] | None = None,
) -> str:
    """Create an audit packet that preserves internal agent IDs."""

    agent_ids = list(round_payload["models"])
    return _build_packet(
        round_payload=round_payload,
        labels={agent_id: agent_id for agent_id in agent_ids},
        heading="Round 2 human audit packet",
        task_line=(
            "All scores are resilience view: higher = more resilient. "
            "Researcher view with real agent IDs preserved."
        ),
        flags=flags or [],
    )


def build_round2_packet(round_payload: dict[str, Any]) -> str:
    """Backward-compatible alias for the model-facing packet."""

    return build_model_facing_packet(round_payload)


def _build_packet(
    round_payload: dict[str, Any],
    labels: dict[str, str],
    heading: str,
    task_line: str,
    flags: list[dict],
) -> str:
    """Create a text packet summarizing R1 opinions by indicator."""

    lines = [
        heading,
        f"City: {round_payload['city_name']}",
        task_line,
        "",
    ]

    for indicator in RESILIENCE_INDICATORS:
        meta = get_indicator_meta(indicator)
        cells = [
            _indicator_cell(agent_data, indicator)
            for agent_data in round_payload["models"].values()
        ]
        scores = [cell["score"] for cell in cells]
        indicator_flags = [
            flag for flag in flags if flag.get("indicator") == indicator
        ]
        lines.append(f"## {meta.alias_name}")
        lines.append(f"id: {indicator}")
        lines.append(f"dimension: {meta.dimension}")
        lines.append(f"stats: mean={_mean(scores)}, std={_std(scores)}, n={len(scores)}")
        anchor_text = _format_anchors(meta.city_anchors_resilience)
        if anchor_text:
            lines.append(f"city_anchors_resilience: {anchor_text}")
        for flag in indicator_flags:
            flagged_agent = flag.get("model", flag.get("agent_id", "unknown_model"))
            flag_label = labels.get(flagged_agent, flagged_agent)
            lines.append(f"consul_flag: {flag_label} outlier; {flag.get('reason', '')}")
        for agent_id, agent_data in round_payload["models"].items():
            cell = _indicator_cell(agent_data, indicator)
            reasoning = _apply_labels_to_text(cell["reasoning"], labels)
            lines.append(
                f"- {labels[agent_id]}: "
                f"score={cell['score']}; reasoning={reasoning}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_round2_packet(path: str | Path, packet: str) -> None:
    """Write the R2 deliberation packet to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet, encoding="utf-8")


def _indicator_cell(agent_data: dict[str, Any], indicator: str) -> dict[str, Any]:
    try:
        cell = agent_data["indicators"][indicator]
        reasoning = cell["reasoning"]
    except KeyError as exc:
        agent_id = agent_data.get("agent_id", "unknown_agent")
        raise ValueError(f"missing reasoning for {agent_id} / {indicator}") from exc
    if not reasoning:
        agent_id = agent_data.get("agent_id", "unknown_agent")
        raise ValueError(f"empty reasoning for {agent_id} / {indicator}")
    return cell


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 2)


def _format_anchors(anchors: dict[str, float], limit: int = 6) -> str:
    if not anchors:
        return ""
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


def _apply_labels_to_text(text: str, labels: dict[str, str]) -> str:
    anonymized = text
    for raw_label, replacement in labels.items():
        anonymized = anonymized.replace(raw_label, replacement)
    return anonymized
