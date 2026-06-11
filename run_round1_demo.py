"""Run the deterministic R1 mock-agent demo from the project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament.run_round1 import main  # noqa: E402


if __name__ == "__main__":
    # Keep default relative paths stable when the script is launched from an IDE.
    import os

    os.chdir(PROJECT_ROOT)
    main()
