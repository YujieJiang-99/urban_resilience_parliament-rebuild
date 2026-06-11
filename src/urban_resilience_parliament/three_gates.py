"""Lightweight Three Gates conversions between storage and resilience views."""

from .indicators import get_indicator_meta


STORAGE_DEFICIT_DIMENSIONS = {"cap_abs", "cap_resp"}
STORAGE_CAPABILITY_DIMENSIONS = {"cap_rec"}


def storage_kind_for_indicator(indicator_id: str) -> str:
    """Return whether storage view is deficit or capability for one indicator."""

    dimension = get_indicator_meta(indicator_id).dimension
    if dimension in STORAGE_DEFICIT_DIMENSIONS:
        return "deficit"
    if dimension in STORAGE_CAPABILITY_DIMENSIONS:
        return "capability"
    raise ValueError(f"unknown indicator dimension for {indicator_id}: {dimension}")


def storage_to_resilience(indicator_id: str, value: float) -> float:
    """Convert a 0-1 storage value to resilience view where higher is better."""

    _validate_unit_value(value)
    if storage_kind_for_indicator(indicator_id) == "deficit":
        return round(1 - value, 6)
    return value


def resilience_to_storage(indicator_id: str, value: float) -> float:
    """Convert a 0-1 resilience value to storage view."""

    _validate_unit_value(value)
    if storage_kind_for_indicator(indicator_id) == "deficit":
        return round(1 - value, 6)
    return value


def _validate_unit_value(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("Three Gates conversion values must be between 0 and 1")
