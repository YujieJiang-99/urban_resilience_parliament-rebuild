"""Run real LLM-backed R1 assessments for one city and selected models.

This runner is serial and resumable:
- one city
- selected models
- all 18 metadata indicators
- no R2
- no Consul
- no Tavily
- no concurrency
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import sys
from pathlib import Path
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament import OpenAICompatibleLLMBackend  # noqa: E402
from urban_resilience_parliament.config import load_llm_config  # noqa: E402
from urban_resilience_parliament.io import (  # noqa: E402
    agent_round_from_dict,
    agent_round_to_compact_dict,
    load_city_input,
    model_filename,
    read_json,
    rounds_to_compact_payload,
    write_json,
)
from urban_resilience_parliament.personas import default_models, model_spec_from_name  # noqa: E402
from urban_resilience_parliament.prompts import build_r1_prompt  # noqa: E402
from urban_resilience_parliament.validation import validate_agent_round, validate_round_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run serial real R1 assessments for selected models.")
    parser.add_argument(
        "--input",
        default="data/examples/city_input_minimal.json",
        help="City input JSON path.",
    )
    parser.add_argument(
        "--run-dir",
        default="data/runs/hong_kong_real_r1",
        help="Output run directory.",
    )
    parser.add_argument(
        "--models",
        default=",".join(model.model_name for model in default_models()),
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if model_<name>.json already exists.",
    )
    args = parser.parse_args()

    city = load_city_input(PROJECT_ROOT / args.input)
    run_dir = PROJECT_ROOT / args.run_dir
    round1_dir = run_dir / "round1"
    prompt_dir = round1_dir / "prompts"
    raw_log_dir = round1_dir / "raw_logs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_files(round1_dir)

    base_config = load_llm_config(PROJECT_ROOT / ".env")
    model_names = _parse_models(args.models)
    errors: list[dict] = []

    print("About to run serial real R1 model calls.")
    print(f"Base URL: {base_config.base_url}")
    print(f"City: {city.city_name}")
    print(f"Models: {', '.join(model_names)}")
    print("Scope: R1 only, 18 indicators, no R2, no Consul, no Tavily, no concurrency.")

    for model_name in model_names:
        model_path = round1_dir / model_filename(model_name)
        if args.force and model_path.exists():
            model_path.unlink()
        if model_path.exists() and not args.force:
            print(f"Skipping existing model output: {model_name}")
            continue

        print(f"Calling model: {model_name}")
        model = model_spec_from_name(model_name)
        model_log_dir = raw_log_dir / _safe_model_name(model_name)
        config = replace(
            base_config,
            model=model_name,
            timeout=max(base_config.timeout, 180),
            log_dir=model_log_dir,
        )
        prompt = build_r1_prompt(city, model)
        prompt_path = prompt_dir / f"model_{_safe_model_name(model_name)}_r1_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        try:
            backend = OpenAICompatibleLLMBackend(config=config)
            agent_round = backend.generate_round1(city=city, persona=model, prompt=prompt)
            validate_agent_round(agent_round)
            write_json(
                model_path,
                agent_round_to_compact_dict(agent_round, city, "independent_scoring"),
            )
            print(f"Saved: {model_path}")
        except Exception as exc:  # noqa: BLE001 - runner must continue after one model fails
            error = {
                "model": model_name,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
                "log_dir": str(model_log_dir),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            errors.append(error)
            print(f"Model failed: {model_name}. Error recorded in errors.json")

    all_models_payload, load_errors = _build_all_models_from_files(city, round1_dir, model_names)
    errors.extend(load_errors)
    if all_models_payload["models"]:
        validate_round_payload(all_models_payload)
    write_json(round1_dir / "all_models.json", all_models_payload)
    write_json(round1_dir / "errors.json", {"errors": errors})
    print(f"Saved all_models JSON to {round1_dir / 'all_models.json'}")
    print(f"Saved errors JSON to {round1_dir / 'errors.json'}")


def _build_all_models_from_files(city, round1_dir: Path, model_names: list[str]) -> tuple[dict, list[dict]]:
    rounds = []
    errors = []
    for model_name in model_names:
        model_path = round1_dir / model_filename(model_name)
        if not model_path.exists():
            continue
        try:
            rounds.append(agent_round_from_dict(read_json(model_path)))
        except Exception as exc:  # noqa: BLE001 - keep runner resumable
            errors.append(
                {
                    "model": model_name,
                    "error": f"could not load existing model output: {exc}",
                    "path": str(model_path),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
    return (
        rounds_to_compact_payload(
            city=city,
            rounds=rounds,
            round_number=1,
            stage="independent_scoring",
        ),
        errors,
    )


def _parse_models(raw_models: str) -> list[str]:
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not models:
        raise ValueError("at least one model must be specified")
    return models


def _safe_model_name(model_name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in model_name)


def _cleanup_legacy_files(round1_dir: Path) -> None:
    for stale_path in (
        round1_dir / "all_agents.json",
        round1_dir / "infrastructure_planner.json",
        round1_dir / "social_resilience_reviewer.json",
        round1_dir / "climate_adaptation_analyst.json",
    ):
        if stale_path.exists():
            stale_path.unlink()


if __name__ == "__main__":
    main()
