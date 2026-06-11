"""Indicator metadata loaded from bundled research files."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "bundled_data"
    / "indicator_meta_minimal.json"
)


@dataclass(frozen=True)
class IndicatorMeta:
    """Metadata for one resilience indicator."""

    indicator_id: str
    dimension: str
    display_name: str
    alias_name: str
    polarity: str
    city_level_definition: str
    city_anchors_resilience: dict[str, float]


def load_indicator_metadata(path: str | Path = METADATA_PATH) -> dict[str, IndicatorMeta]:
    """Load indicator metadata keyed by indicator id."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    indicators = data["_indicators"]
    return {
        indicator_id: IndicatorMeta(
            indicator_id=indicator_id,
            dimension=item["dimension"],
            display_name=item["display_name"],
            alias_name=item["alias_name"],
            polarity=item["polarity"],
            city_level_definition=item["city_level_definition"],
            city_anchors_resilience=item.get("city_anchors_resilience", {}),
        )
        for indicator_id, item in indicators.items()
    }


INDICATOR_METADATA = load_indicator_metadata()
RESILIENCE_INDICATORS: tuple[str, ...] = tuple(INDICATOR_METADATA)
EXPECTED_INDICATOR_COUNT = len(RESILIENCE_INDICATORS)


def get_indicator_meta(indicator_id: str) -> IndicatorMeta:
    """Return metadata for one indicator id."""

    return INDICATOR_METADATA[indicator_id]
