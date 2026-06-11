"""Deterministic mock Consul/referee audit."""

from pathlib import Path
from statistics import median
from typing import Any

from .indicators import RESILIENCE_INDICATORS
from .io import read_json, write_json

OUTLIER_THRESHOLD = 8.0


def run_consul_from_payloads(
    round1_payload: dict[str, Any],
    round2_payload: dict[str, Any],
) -> dict[str, Any]:
    """Audit R2 outliers without rewriting scores or aggregating outputs."""

    flags = _find_outlier_flags(round2_payload)
    report = {
        "city_id": round2_payload["city_id"],
        "stage": "consul_audit",
        "policy": {
            "outlier_threshold": OUTLIER_THRESHOLD,
            "basis": "R2 score deviation from peer median by indicator",
        },
        "rounds_read": {
            "round1": round1_payload["round"],
            "round2": round2_payload["round"],
        },
        "status": "flagged" if flags else "no_flag",
        "decision": "exclude_flagged_from_aggregate" if flags else "keep_all",
        "flags": flags,
    }
    return report


def run_consul_from_files(
    round1_all_agents_path: str | Path,
    round2_all_agents_path: str | Path,
) -> dict[str, Any]:
    """Read compact round outputs from disk, then run the Consul audit."""

    return run_consul_from_payloads(
        read_json(round1_all_agents_path),
        read_json(round2_all_agents_path),
    )


def write_consul_outputs(
    run_dir: str | Path,
    consul_report: dict[str, Any],
) -> None:
    """Write the Consul report file."""

    output_dir = Path(run_dir)
    write_json(output_dir / "consul_report.json", consul_report)


def _find_outlier_flags(round2_payload: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for indicator in RESILIENCE_INDICATORS:
        values = [
            agent_data["indicators"][indicator]["score"]
            for agent_data in round2_payload["models"].values()
        ]
        center = median(values)
        for model_name, agent_data in round2_payload["models"].items():
            score = agent_data["indicators"][indicator]["score"]
            deviation = abs(score - center)
            if deviation >= OUTLIER_THRESHOLD:
                flags.append(
                    {
                        "indicator": indicator,
                        "model": model_name,
                        "score": score,
                        "peer_median": center,
                        "reason": (
                            f"R2 score deviates from peer median by {round(deviation, 2)}, "
                            f"meeting threshold {OUTLIER_THRESHOLD}."
                        ),
                    }
                )
    return flags
