import json

import pytest

from urban_resilience_parliament import (
    INDICATOR_METADATA,
    LLMConfig,
    OpenAICompatibleLLMBackend,
    RESILIENCE_INDICATORS,
    RESILIENCE_VIEW_INSTRUCTION,
    CityInput,
    build_r1_prompt,
    build_r1_smoke_prompt,
    build_r2_prompt,
    default_models,
    get_indicator_meta,
    run_round1,
    storage_kind_for_indicator,
    storage_to_resilience,
)
from urban_resilience_parliament.backend import extract_first_json_object
from urban_resilience_parliament.io import agent_round_from_dict
from urban_resilience_parliament.round1 import run_round1_from_files
from urban_resilience_parliament.round2_packet import build_model_facing_packet


def make_city() -> CityInput:
    return CityInput(
        city_id="hk-demo",
        city_name="Hong Kong",
        country_or_region="Hong Kong SAR, China",
        summary="Minimal test city.",
        evidence={"hazard_context": "placeholder"},
    )


def write_city_input(path) -> None:
    path.write_text(
        json.dumps(
            {
                "city_id": "hk-demo",
                "city_name": "Hong Kong",
                "country_or_region": "Hong Kong SAR, China",
                "summary": "Minimal test city.",
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )


def assert_indicator_cells(indicators: dict) -> None:
    assert set(indicators) == set(RESILIENCE_INDICATORS)
    for cell in indicators.values():
        assert set(cell) == {"score", "reasoning"}
        assert isinstance(cell["score"], float | int)
        assert 0 <= cell["score"] <= 1
        assert cell["reasoning"]


def test_round1_mock_agents_output_all_18_indicators() -> None:
    rounds = run_round1(make_city())

    assert len(rounds) == 3
    for agent_round in rounds:
        assert agent_round.round_number == 1
        assert len(agent_round.scores) == 18
        assert {score.indicator for score in agent_round.scores} == set(RESILIENCE_INDICATORS)


def test_indicator_metadata_has_18_indicators_across_three_capacities() -> None:
    assert len(INDICATOR_METADATA) == 18
    dimensions = [meta.dimension for meta in INDICATOR_METADATA.values()]
    assert dimensions.count("cap_abs") == 6
    assert dimensions.count("cap_resp") == 6
    assert dimensions.count("cap_rec") == 6


def test_prompts_declare_resilience_view() -> None:
    city = make_city()
    persona = default_models()[0]
    r1_prompt = build_r1_prompt(city, persona)
    smoke_prompt = build_r1_smoke_prompt(city, persona, list(RESILIENCE_INDICATORS[:2]))
    r2_prompt = build_r2_prompt(city, persona, "## packet body")

    assert RESILIENCE_VIEW_INSTRUCTION in r1_prompt
    assert RESILIENCE_VIEW_INSTRUCTION in smoke_prompt
    assert RESILIENCE_VIEW_INSTRUCTION in r2_prompt
    assert "higher = more resilient" in r1_prompt
    assert "higher = more resilient" in smoke_prompt
    assert "higher = more resilient" in r2_prompt
    assert "city_anchors_resilience" in r1_prompt
    assert "Return only valid JSON" in smoke_prompt
    assert "Agent persona" not in r1_prompt


def test_three_gates_flip_storage_deficits_but_not_capabilities() -> None:
    cap_abs = "ind__cap_abs__Building_quality_control_index"
    cap_resp = "ind__cap_resp__Control_of_corruption"
    cap_rec = "ind__cap_rec__Government_Effectiveness_Index"

    assert storage_kind_for_indicator(cap_abs) == "deficit"
    assert storage_kind_for_indicator(cap_resp) == "deficit"
    assert storage_kind_for_indicator(cap_rec) == "capability"
    assert storage_to_resilience(cap_abs, 0.2) == 0.8
    assert storage_to_resilience(cap_resp, 0.2) == 0.8
    assert storage_to_resilience(cap_rec, 0.2) == 0.2


def test_round1_mock_agent_scores_are_in_legal_ranges() -> None:
    rounds = run_round1(make_city())

    for agent_round in rounds:
        for score in agent_round.scores:
            assert 0 <= score.score <= 1
            assert score.confidence is not None
            assert 0 <= score.confidence <= 1
            assert score.rationale


def test_round1_file_runner_saves_json(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    payload = run_round1_from_files(input_path, run_dir)
    output_path = run_dir / "round1" / "all_models.json"
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == saved
    assert saved["stage"] == "independent_scoring"
    assert len(saved["models"]) == 3
    for agent_data in saved["models"].values():
        assert set(agent_data) == {"city_id", "city_name", "round", "stage", "model", "indicators"}
        assert_indicator_cells(agent_data["indicators"])


def test_round1_file_runner_saves_one_file_per_agent(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    payload = run_round1_from_files(input_path, run_dir)

    assert (run_dir / "round1" / "all_models.json").exists()
    for model_name in payload["models"]:
        safe_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_name)
        agent_path = run_dir / "round1" / f"model_{safe_name}.json"
        assert agent_path.exists()
        saved_agent = json.loads(agent_path.read_text(encoding="utf-8"))
        assert saved_agent["model"] == model_name
        assert_indicator_cells(saved_agent["indicators"])
        assert "scores" not in saved_agent
        assert "agent_id" not in saved_agent
        assert "rationale" not in json.dumps(saved_agent)
        agent_round = agent_round_from_dict(saved_agent)
        assert len(agent_round.scores) == 18


def test_round2_packets_are_generated_with_model_and_human_views(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    payload = run_round1_from_files(input_path, run_dir)
    model_packet_path = run_dir / "round2" / "packet.txt"
    audit_packet_path = run_dir / "round2" / "human_audit_packet.txt"
    model_packet = model_packet_path.read_text(encoding="utf-8")
    audit_packet = audit_packet_path.read_text(encoding="utf-8")

    assert model_packet_path.exists()
    assert audit_packet_path.exists()
    assert "City: Hong Kong" in model_packet
    for indicator in RESILIENCE_INDICATORS:
        meta = get_indicator_meta(indicator)
        assert f"## {meta.alias_name}" in model_packet
        assert f"id: {indicator}" in model_packet
        assert f"dimension: {meta.dimension}" in model_packet
        assert f"## {meta.alias_name}" in audit_packet
        assert f"id: {indicator}" in audit_packet
        assert f"dimension: {meta.dimension}" in audit_packet
    assert "city_anchors_resilience:" in model_packet
    assert model_packet.count("stats: mean=") == 18
    assert audit_packet.count("stats: mean=") == 18
    assert "std=" in model_packet
    assert "n=3" in model_packet
    assert "confidence" not in model_packet.lower()
    assert "confidence" not in audit_packet.lower()
    assert "Agent_A" in model_packet
    for model_name in payload["models"]:
        assert model_name not in model_packet
        assert model_name in audit_packet


def test_round2_outputs_all_agents_and_changes_some_scores(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    run_round1_from_files(input_path, run_dir)
    round1 = json.loads((run_dir / "round1" / "all_models.json").read_text(encoding="utf-8"))
    round2 = json.loads((run_dir / "round2" / "all_models.json").read_text(encoding="utf-8"))

    assert round2["stage"] == "peer_aware_deliberation"
    assert len(round2["models"]) == 3
    changed_scores = 0
    for model_name, r2_agent in round2["models"].items():
        safe_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_name)
        assert (run_dir / "round2" / f"model_{safe_name}.json").exists()
        assert_indicator_cells(r2_agent["indicators"])
        for indicator, r2_cell in r2_agent["indicators"].items():
            r1_cell = round1["models"][model_name]["indicators"][indicator]
            if r1_cell["score"] != r2_cell["score"]:
                changed_scores += 1

    assert changed_scores > 0


def test_consul_report_is_generated_without_final_aggregate(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    run_round1_from_files(input_path, run_dir)
    report = json.loads((run_dir / "consul_report.json").read_text(encoding="utf-8"))

    assert report["stage"] == "consul_audit"
    assert report["decision"] in {"keep_all", "exclude_flagged_from_aggregate"}
    assert not (run_dir / "final_aggregate.json").exists()


def test_round2_packets_cover_all_indicators_with_compact_stats(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    run_round1_from_files(input_path, run_dir)
    for packet_name in ("packet.txt", "human_audit_packet.txt"):
        packet = (run_dir / "round2" / packet_name).read_text(encoding="utf-8")
        for indicator in RESILIENCE_INDICATORS:
            meta = get_indicator_meta(indicator)
            block_start = packet.index(f"## {meta.alias_name}")
            next_marker = packet.find("\n## ", block_start + 1)
            block = packet[block_start:] if next_marker == -1 else packet[block_start:next_marker]
            assert f"id: {indicator}" in block
            assert f"dimension: {meta.dimension}" in block
            assert "stats: mean=" in block
            assert "std=" in block
            assert "n=3" in block
            assert "reasoning=" in block


def test_round_outputs_and_consul_cover_metadata_indicators(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)

    run_round1_from_files(input_path, run_dir)
    expected = set(INDICATOR_METADATA)
    round1 = json.loads((run_dir / "round1" / "all_models.json").read_text(encoding="utf-8"))
    round2 = json.loads((run_dir / "round2" / "all_models.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "consul_report.json").read_text(encoding="utf-8"))

    for payload in (round1, round2):
        for agent_data in payload["models"].values():
            assert set(agent_data["indicators"]) == expected
            assert_indicator_cells(agent_data["indicators"])
    for flag in report["flags"]:
        assert flag["indicator"] in expected


def test_packet_fails_when_reasoning_is_missing(tmp_path) -> None:
    input_path = tmp_path / "city.json"
    run_dir = tmp_path / "runs" / "hong_kong_demo"
    write_city_input(input_path)
    payload = run_round1_from_files(input_path, run_dir)
    first_agent = next(iter(payload["models"].values()))
    first_indicator = next(iter(first_agent["indicators"]))
    del first_agent["indicators"][first_indicator]["reasoning"]

    with pytest.raises(ValueError, match="missing reasoning"):
        build_model_facing_packet(payload)


def test_openai_compatible_backend_parses_indicator_json(tmp_path) -> None:
    indicator_ids = list(RESILIENCE_INDICATORS[:2])
    backend = OpenAICompatibleLLMBackend(
        config=LLMConfig(
            base_url="https://example.test/v1",
            api_key="sk-test",
            model="test-model",
            timeout=1,
            debug=False,
            log_dir=tmp_path,
        )
    )
    content = json.dumps(
        {
            "indicators": {
                indicator_ids[0]: {"score": 0.67, "reasoning": "first reason"},
                indicator_ids[1]: {"score": 0.725, "reasoning": "second reason"},
            }
        }
    )

    parsed = backend.parse_indicator_content(
        content=content,
        raw_response='{"choices":[]}',
        indicator_ids=indicator_ids,
    )

    assert set(parsed) == set(indicator_ids)
    assert parsed[indicator_ids[0]]["score"] == 0.67
    assert parsed[indicator_ids[0]]["reasoning"] == "first reason"


def test_json_extraction_handles_markdown_and_explanatory_text(tmp_path) -> None:
    indicator_ids = list(RESILIENCE_INDICATORS[:2])
    wrapped = f"""
Here is the result:

```json
{{
  "indicators": {{
    "{indicator_ids[0]}": {{"score": 0.67, "reasoning": "first reason"}},
    "{indicator_ids[1]}": {{"score": 0.72, "reasoning": "second reason"}}
  }}
}}
```

Thanks.
"""
    extracted = extract_first_json_object(wrapped)
    assert json.loads(extracted)["indicators"][indicator_ids[0]]["score"] == 0.67

    backend = OpenAICompatibleLLMBackend(
        config=LLMConfig(
            base_url="https://example.test/v1",
            api_key="sk-test",
            model="test-model",
            timeout=1,
            debug=False,
            log_dir=tmp_path,
        )
    )
    parsed = backend.parse_indicator_content(
        content=wrapped,
        raw_response="RAW",
        indicator_ids=indicator_ids,
    )
    assert set(parsed) == set(indicator_ids)


def test_openai_compatible_backend_saves_raw_response_for_non_json(tmp_path) -> None:
    backend = OpenAICompatibleLLMBackend(
        config=LLMConfig(
            base_url="https://example.test/v1",
            api_key="sk-test",
            model="test-model",
            timeout=1,
            debug=False,
            log_dir=tmp_path,
        )
    )

    with pytest.raises(ValueError, match="Raw response saved"):
        backend.parse_indicator_content(
            content="not json",
            raw_response="RAW RESPONSE BODY",
            indicator_ids=[RESILIENCE_INDICATORS[0]],
        )

    assert (tmp_path / "last_llm_raw_response.txt").read_text(encoding="utf-8") == "RAW RESPONSE BODY"


def test_openai_compatible_backend_validates_indicator_cells(tmp_path) -> None:
    backend = OpenAICompatibleLLMBackend(
        config=LLMConfig(
            base_url="https://example.test/v1",
            api_key="sk-test",
            model="test-model",
            timeout=1,
            debug=False,
            log_dir=tmp_path,
        )
    )
    indicator = RESILIENCE_INDICATORS[0]
    bad_content = json.dumps({"indicators": {indicator: {"score": 1.01, "reasoning": ""}}})

    with pytest.raises(ValueError, match="score must be between 0 and 1"):
        backend.parse_indicator_content(
            content=bad_content,
            raw_response="RAW",
            indicator_ids=[indicator],
        )


def test_openai_backend_requires_anchor_calibration_for_real_city(tmp_path) -> None:
    backend = OpenAICompatibleLLMBackend(
        config=LLMConfig(
            base_url="https://example.test/v1",
            api_key="sk-test",
            model="test-model",
            timeout=1,
            debug=False,
            log_dir=tmp_path,
        )
    )
    indicator = "ind__cap_abs__Building_quality_control_index"
    content = json.dumps(
        {
            "indicators": {
                indicator: {
                    "score": 0.85,
                    "reasoning": (
                        "Hong Kong has strong building governance, anchored against "
                        "Tokyo=1.0000 / London=0.8947. target city appears in anchors: "
                        "Hong Kong=0.9474."
                    ),
                }
            }
        }
    )

    parsed = backend.parse_indicator_content(
        content=content,
        raw_response="RAW",
        indicator_ids=[indicator],
        target_city="Hong Kong",
    )

    assert parsed[indicator]["score"] == 0.85
