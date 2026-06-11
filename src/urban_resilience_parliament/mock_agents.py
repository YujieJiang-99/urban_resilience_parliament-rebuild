"""Backward-compatible aliases for the persona registry."""

from .personas import AgentPersona as MockAgent
from .personas import default_models


def default_mock_agents() -> list[MockAgent]:
    """Return the default personas under the old mock-agent name."""

    return default_models()
