"""Run a tiny OpenAI-compatible LLM smoke test.

This calls one model for one city and two indicators. It does not run the full
parliament, R2, Consul, web search, or any physical-risk modules.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament import (  # noqa: E402
    OpenAICompatibleLLMBackend,
    RESILIENCE_INDICATORS,
    build_r1_smoke_prompt,
    load_llm_config,
)
from urban_resilience_parliament.io import load_city_input, write_json  # noqa: E402
from urban_resilience_parliament.personas import model_spec_from_name  # noqa: E402


def main() -> None:
    city = load_city_input(PROJECT_ROOT / "data" / "examples" / "city_input_minimal.json")
    indicator_ids = list(RESILIENCE_INDICATORS[:2])
    config = load_llm_config(PROJECT_ROOT / ".env")
    model = model_spec_from_name(config.model)
    backend = OpenAICompatibleLLMBackend(config=config)
    prompt = build_r1_smoke_prompt(city, model, indicator_ids)
    result = backend.generate_indicator_subset_payload(
        city=city,
        persona=model,
        prompt=prompt,
        indicator_ids=indicator_ids,
        round_number=1,
        stage="llm_smoke_test",
    )
    output_path = PROJECT_ROOT / "data" / "runs" / "llm_smoke" / "smoke_result.json"
    write_json(output_path, result)
    print(f"Saved LLM smoke result to {output_path}")


if __name__ == "__main__":
    main()
