"""Minimal structures for a city resilience parliament demo."""

from .backend import LLMBackend, MockLLMBackend, OpenAICompatibleLLMBackend
from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMConfig, load_llm_config
from .indicators import INDICATOR_METADATA, RESILIENCE_INDICATORS, get_indicator_meta
from .consul import run_consul_from_files, run_consul_from_payloads
from .parliament import aggregate_parliamentary_score
from .mock_agents import MockAgent, default_mock_agents
from .personas import AgentPersona, ModelSpec, default_models, default_personas, model_spec_from_name
from .prompts import RESILIENCE_VIEW_INSTRUCTION, build_r1_prompt, build_r1_smoke_prompt, build_r2_prompt
from .round1 import run_round1
from .round2 import run_round2
from .round2_packet import build_human_audit_packet, build_model_facing_packet, build_round2_packet
from .schemas import (
    AgentRound,
    CityInput,
    IndicatorScore,
    ParliamentaryAssessment,
    RefereeReview,
)
from .three_gates import resilience_to_storage, storage_kind_for_indicator, storage_to_resilience

__all__ = [
    "AgentRound",
    "AgentPersona",
    "ModelSpec",
    "CityInput",
    "IndicatorScore",
    "INDICATOR_METADATA",
    "LLMBackend",
    "LLMConfig",
    "MockAgent",
    "MockLLMBackend",
    "OpenAICompatibleLLMBackend",
    "ParliamentaryAssessment",
    "RESILIENCE_VIEW_INSTRUCTION",
    "RefereeReview",
    "RESILIENCE_INDICATORS",
    "aggregate_parliamentary_score",
    "build_r1_prompt",
    "build_r1_smoke_prompt",
    "build_r2_prompt",
    "build_round2_packet",
    "build_human_audit_packet",
    "build_model_facing_packet",
    "default_mock_agents",
    "default_models",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "default_personas",
    "get_indicator_meta",
    "load_llm_config",
    "model_spec_from_name",
    "resilience_to_storage",
    "run_round1",
    "run_consul_from_payloads",
    "run_consul_from_files",
    "run_round2",
    "storage_kind_for_indicator",
    "storage_to_resilience",
]
