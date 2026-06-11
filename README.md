# Urban Resilience Parliament Rebuild
<img width="2475" height="1573" alt="image" src="https://github.com/user-attachments/assets/8367d5c2-a9ff-4b82-9bea-520333aad681" />

This is a minimal research demo for the B-1 module: multi-agent parliamentary
assessment of city resilience.

The demo focuses on one idea: a city can be evaluated against 18 resilience
indicators by several LLM-style agents. Each agent first gives an independent
score and rationale, then reviews anonymous peer opinions in a second round.
A referee or consul then checks for obvious directional mistakes and extreme
outliers in a compact audit report.

The 18 indicators are loaded from
`src/bundled_data/indicator_meta_minimal.json`; JSON outputs use indicator IDs,
while packets show the human-facing `alias_name`, dimension, and city anchors.

This repository intentionally does not implement:

- physical risk modeling
- AAL, exposure, or vulnerability calculations
- Bayesian fusion
- real web search
- real LLM API calls

The goal is to keep the method legible before adding heavier machinery.

## Project Layout

```text
urban_resilience_parliament-rebuild/
  README.md
  requirements.txt
  data/
    examples/
      city_input_minimal.json
      parliamentary_output_minimal.json
  src/
    urban_resilience_parliament/
      __init__.py
      backend.py
      indicators.py
      personas.py
      prompts.py
      parliament.py
      schemas.py
      three_gates.py
  tests/
    test_schemas.py
```

## Core Flow

1. Load a city profile and its available evidence.
2. Ask multiple agents to score 18 resilience indicators independently.
3. Share anonymized first-round opinions with all agents.
4. Ask agents to revise or defend their scores in a second round.
5. Let a referee inspect directionality errors and extreme values.

For now, the code only defines the data shapes and a deterministic placeholder
aggregation path. Future versions can replace the placeholder agents with real
model adapters.

The execution path is organized around a small backend interface. The default
`MockLLMBackend` is deterministic, but R1 and R2 already flow through the same
persona registry, prompt builders, backend methods, and JSON validation hooks
that a real LLM adapter can use later.

Three Gates utilities convert between storage view and resilience view when
needed. The active prompt, packet, and mock-agent scoring path use resilience
view throughout: higher means more resilient.

## Quick Start

```bash
python -m pytest
```

Run the deterministic R1 mock-agent demo:

```bash
python run_round1_demo.py
```

By default this reads `data/examples/city_input_minimal.json` and writes
the following audit artifacts:

- `data/runs/hong_kong_demo/round1/all_agents.json`
- `data/runs/hong_kong_demo/round1/<agent_id>.json`
- `data/runs/hong_kong_demo/round2/packet.txt`
- `data/runs/hong_kong_demo/round2/human_audit_packet.txt`
- `data/runs/hong_kong_demo/round2/all_agents.json`
- `data/runs/hong_kong_demo/round2/<agent_id>.json`
- `data/runs/hong_kong_demo/consul_report.json`

The project currently uses only the Python standard library at runtime.

## LLM Smoke Test

The default backend is still `MockLLMBackend`. To test one real
OpenAI-compatible chat-completions call, create `.env` from `.env.example` and
fill in:

```text
LLM_BASE_URL=https://api.zhizengzeng.com/v1
LLM_API_KEY=...
LLM_MODEL=...
```

Then run:

```bash
python run_llm_smoke.py
```

This calls one model for Hong Kong and two indicators only, then writes
`data/runs/llm_smoke/smoke_result.json`. It does not run R2, Consul, web search,
or the full 18-indicator parliament.
