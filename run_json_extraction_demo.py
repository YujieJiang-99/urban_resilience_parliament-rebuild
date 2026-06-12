"""Offline demo for robust JSON extraction from markdown-wrapped model text."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_resilience_parliament import RESILIENCE_INDICATORS  # noqa: E402
from urban_resilience_parliament.backend import extract_first_json_object  # noqa: E402


def main() -> None:
    first, second = RESILIENCE_INDICATORS[:2]
    fake_response = f"""
The assessment is below.

```json
{{
  "indicators": {{
    "{first}": {{"score": 0.67, "reasoning": "first short reason"}},
    "{second}": {{"score": 0.72, "reasoning": "second short reason"}}
  }}
}}
```
"""
    extracted = extract_first_json_object(fake_response)
    parsed = json.loads(extracted)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
