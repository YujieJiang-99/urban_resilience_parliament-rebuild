"""CLI entry point for deterministic model-based scoring."""

import argparse
from pathlib import Path

from .round1 import run_round1_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic model-based mock scoring.")
    parser.add_argument(
        "--input",
        default="data/examples/city_input_minimal.json",
        help="Path to a city input JSON file.",
    )
    parser.add_argument(
        "--run-dir",
        default="data/runs/hong_kong_demo",
        help="Directory where round outputs and packets should be saved.",
    )
    args = parser.parse_args()

    payload = run_round1_from_files(args.input, args.run_dir)
    run_dir = Path(args.run_dir)
    print(
        f"Saved R1 audit outputs for {payload['city_name']} "
        f"with {len(payload['models'])} models under {run_dir}"
    )
    print(f"Saved R2 packets and mock deliberation outputs under {run_dir / 'round2'}")
    print(f"Saved Consul report under {run_dir}")


if __name__ == "__main__":
    main()
