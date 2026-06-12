"""Compatibility wrapper for running one real model R1 assessment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament.config import load_llm_config  # noqa: E402

from run_round1_models import main as run_models_main  # noqa: E402


if __name__ == "__main__":
    model = load_llm_config(PROJECT_ROOT / ".env").model
    sys.argv = [sys.argv[0], "--models", model]
    run_models_main()
