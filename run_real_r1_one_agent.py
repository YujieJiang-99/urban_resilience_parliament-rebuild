"""Run one real LLM-backed R1 assessment for Hong Kong.

This is a minimal real-backend run:
- one city
- one model
- all 18 metadata indicators
- no R2
- no Consul
- no Tavily
- no concurrency
"""

from dataclasses import replace
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament import OpenAICompatibleLLMBackend  # noqa: E402
from urban_resilience_parliament.config import load_llm_config  # noqa: E402
from urban_resilience_parliament.io import (  # noqa: E402
    agent_round_to_compact_dict,
    load_city_input,
    model_filename,
    rounds_to_compact_payload,
    write_json,
)
from urban_resilience_parliament.personas import model_spec_from_name  # noqa: E402
from urban_resilience_parliament.prompts import build_r1_prompt  # noqa: E402
from urban_resilience_parliament.validation import validate_agent_round, validate_round_payload  # noqa: E402


def main() -> None:
    run_dir = PROJECT_ROOT / "data" / "runs" / "hong_kong_real_r1"
    round1_dir = run_dir / "round1"
    prompt_dir = round1_dir / "prompts"
    raw_log_dir = round1_dir / "raw_logs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        round1_dir / "all_agents.json",
        round1_dir / "infrastructure_planner.json",
    ):
        if stale_path.exists():
            stale_path.unlink()
    for old_log in raw_log_dir.glob("*.json"):
        old_log.unlink()

    city = load_city_input(PROJECT_ROOT / "data" / "examples" / "city_input_minimal.json")
    base_config = load_llm_config(PROJECT_ROOT / ".env")
    config = replace(base_config, timeout=max(base_config.timeout, 180), log_dir=raw_log_dir)
    model = model_spec_from_name(config.model)

    print("About to call one real OpenAI-compatible chat/completions model.")
    print(f"Model: {config.model}")
    print(f"Base URL: {config.base_url}")
    print(f"City: {city.city_name}")
    print("Scope: R1 only, one model, 18 indicators, no R2, no Consul, no Tavily, no concurrency.")

    prompt = build_r1_prompt(city, model)
    prompt_path = prompt_dir / f"model_{config.model}_r1_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    backend = OpenAICompatibleLLMBackend(config=config)
    agent_round = backend.generate_round1(city=city, persona=model, prompt=prompt)
    validate_agent_round(agent_round)

    agent_payload = agent_round_to_compact_dict(agent_round, city, "independent_scoring")
    model_path = round1_dir / model_filename(config.model)
    write_json(model_path, agent_payload)

    all_models_payload = rounds_to_compact_payload(
        city=city,
        rounds=[agent_round],
        round_number=1,
        stage="independent_scoring",
    )
    validate_round_payload(all_models_payload)
    write_json(round1_dir / "all_models.json", all_models_payload)

    print(f"Saved prompt to {prompt_path}")
    print(f"Saved R1 model JSON to {model_path}")
    print(f"Saved all_models JSON to {round1_dir / 'all_models.json'}")
    print(f"Saved raw logs under {raw_log_dir}")


if __name__ == "__main__":
    main()
