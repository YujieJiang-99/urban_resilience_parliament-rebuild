"""Model registry for the parliamentary demo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one model in the parliament."""

    model_name: str
    backend_model: str | None
    base_score: float
    confidence: float
    dimension_emphasis: dict[str, float]


def default_models() -> list[ModelSpec]:
    """Return the default mock model registry."""

    return [
        ModelSpec(
            model_name="glm-4-flash",
            backend_model="glm-4-flash",
            base_score=68,
            confidence=0.62,
            dimension_emphasis={
                "cap_abs": 3,
                "cap_resp": 1,
                "cap_rec": 0,
            },
        ),
        ModelSpec(
            model_name="deepseek-v4-flash",
            backend_model="deepseek-v4-flash",
            base_score=64,
            confidence=0.58,
            dimension_emphasis={
                "cap_abs": 1,
                "cap_resp": 0,
                "cap_rec": 3,
            },
        ),
        ModelSpec(
            model_name="qwen3.5-flash",
            backend_model="qwen3.5-flash",
            base_score=66,
            confidence=0.6,
            dimension_emphasis={
                "cap_abs": 2,
                "cap_resp": 3,
                "cap_rec": 1,
            },
        ),
    ]


def model_spec_from_name(model_name: str) -> ModelSpec:
    """Create a single-model spec for a real backend run."""

    return ModelSpec(
        model_name=model_name,
        backend_model=model_name,
        base_score=66,
        confidence=0.6,
        dimension_emphasis={
            "cap_abs": 0,
            "cap_resp": 0,
            "cap_rec": 0,
        },
    )


# Backward-compatible aliases while the project transitions from personas to models.
AgentPersona = ModelSpec
default_personas = default_models
