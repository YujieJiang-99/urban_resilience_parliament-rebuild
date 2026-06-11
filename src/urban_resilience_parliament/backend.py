"""LLM backend interface and deterministic mock implementation."""

import json
from pathlib import Path
from time import time
from typing import Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from .config import LLMConfig, load_llm_config
from .indicators import RESILIENCE_INDICATORS, get_indicator_meta
from .personas import ModelSpec
from .schemas import AgentRound, CityInput, IndicatorScore


class LLMBackend(Protocol):
    """Backend interface used by the R1/R2 runners."""

    def generate_round1(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
    ) -> AgentRound:
        """Generate one R1 agent assessment."""

    def generate_round2(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
        first_rounds: list[AgentRound],
    ) -> AgentRound:
        """Generate one R2 agent assessment."""


class MockLLMBackend:
    """Deterministic backend used until real LLM APIs are connected."""

    def generate_round1(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
    ) -> AgentRound:
        if "higher = more resilient" not in prompt:
            raise ValueError("prompt must define resilience view scoring")

        scores = [
            IndicatorScore(
                indicator=indicator,
                score=self._score_indicator(city, persona, indicator, index),
                rationale=self._r1_reasoning(city, persona, indicator),
                confidence=persona.confidence,
            )
            for index, indicator in enumerate(RESILIENCE_INDICATORS)
        ]
        return AgentRound(
            agent_id=persona.model_name,
            round_number=1,
            scores=scores,
            notes=f"Mock backend R1 assessment for {persona.model_name}.",
        )

    def generate_round2(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
        first_rounds: list[AgentRound],
    ) -> AgentRound:
        if "higher = more resilient" not in prompt:
            raise ValueError("prompt must define resilience view scoring")
        own_round = _find_round(first_rounds, persona.model_name)
        revised_scores = [
            self._revise_score(city, persona, own_round, first_rounds, indicator)
            for indicator in RESILIENCE_INDICATORS
        ]
        return AgentRound(
            agent_id=persona.model_name,
            round_number=2,
            scores=revised_scores,
            notes="Mock backend R2 revision after reviewing anonymous peer evidence.",
        )

    def _score_indicator(
        self,
        city: CityInput,
        persona: ModelSpec,
        indicator: str,
        index: int,
    ) -> float:
        meta = get_indicator_meta(indicator)
        pattern_adjustment = ((index % 5) - 2) * 1.5
        dimension_adjustment = persona.dimension_emphasis.get(meta.dimension, 0)
        anchor_adjustment = _city_anchor_adjustment(meta.city_anchors_resilience, city.city_name)
        score = persona.base_score + pattern_adjustment + dimension_adjustment + anchor_adjustment
        return round(min(100, max(0, score)), 1)

    def _r1_reasoning(self, city: CityInput, persona: ModelSpec, indicator: str) -> str:
        meta = get_indicator_meta(indicator)
        label = meta.alias_name.replace("_", " ")
        anchor = meta.city_anchors_resilience.get(city.city_name)
        anchor_note = (
            f" city anchor={anchor:.4f}."
            if anchor is not None
            else " no direct city anchor."
        )
        return (
            f"{city.city_name} is scored by {persona.model_name}; "
            f"the mock backend treats {label} as a {meta.dimension} indicator with"
            f"{anchor_note}"
        )

    def _revise_score(
        self,
        city: CityInput,
        persona: ModelSpec,
        own_round: AgentRound,
        all_rounds: list[AgentRound],
        indicator: str,
    ) -> IndicatorScore:
        own_score = _score_for(own_round, indicator)
        peer_scores = [
            _score_for(agent_round, indicator)
            for agent_round in all_rounds
            if agent_round.agent_id != persona.model_name
        ]
        if not peer_scores:
            return own_score

        peer_average = sum(score.score for score in peer_scores) / len(peer_scores)
        adjustment = max(-2.0, min(2.0, (peer_average - own_score.score) * 0.25))
        revised_value = round(max(0, min(100, own_score.score + adjustment)), 1)
        revised_confidence = (
            None
            if own_score.confidence is None
            else round(min(1, own_score.confidence + 0.03), 2)
        )
        label = get_indicator_meta(indicator).alias_name.replace("_", " ")
        direction = "upward" if adjustment > 0 else "downward" if adjustment < 0 else "unchanged"
        return IndicatorScore(
            indicator=indicator,
            score=revised_value,
            rationale=(
                f"{city.city_name} R2 mock revision for {label}: adjusted {direction} "
                f"after comparing anonymous peer scores and rationales."
            ),
            confidence=revised_confidence,
        )


class OpenAICompatibleLLMBackend:
    """OpenAI-compatible /chat/completions backend.

    This is intentionally not the default backend. Use it for small smoke tests
    first, then wire it into full parliament runs once parsing is stable.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()
        if not self.config.api_key or self.config.api_key == "sk-your-key-here":
            raise ValueError("LLM_API_KEY is missing. Put it in .env or the process environment.")
        self._call_index = 0

    def generate_round1(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
    ) -> AgentRound:
        return self.generate_indicator_subset(
            city=city,
            persona=persona,
            prompt=prompt,
            indicator_ids=list(RESILIENCE_INDICATORS),
            round_number=1,
        )

    def generate_round2(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
        first_rounds: list[AgentRound],
    ) -> AgentRound:
        return self.generate_indicator_subset(
            city=city,
            persona=persona,
            prompt=prompt,
            indicator_ids=list(RESILIENCE_INDICATORS),
            round_number=2,
        )

    def generate_indicator_subset(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
        indicator_ids: list[str],
        round_number: int = 1,
    ) -> AgentRound:
        """Call one model and parse a subset of indicator scores."""

        raw_response, log_path = self._post_chat_completion(
            prompt=prompt,
            purpose="indicator_subset",
        )
        content = self._extract_message_content(raw_response)
        indicators = self.parse_indicator_content(
            content=content,
            raw_response=raw_response,
            indicator_ids=indicator_ids,
            original_prompt=prompt,
            log_path=log_path,
        )
        scores = [
            IndicatorScore(
                indicator=indicator,
                score=indicators[indicator]["score"],
                rationale=indicators[indicator]["reasoning"],
            )
            for indicator in indicator_ids
        ]
        return AgentRound(
            agent_id=persona.model_name,
            round_number=round_number,
            scores=scores,
            notes=f"OpenAI-compatible backend response from {self.config.model}",
        )

    def generate_indicator_subset_payload(
        self,
        city: CityInput,
        persona: ModelSpec,
        prompt: str,
        indicator_ids: list[str],
        round_number: int = 1,
        stage: str = "smoke_test",
    ) -> dict:
        """Call one model and parse a subset into compact research-log JSON."""

        raw_response, log_path = self._post_chat_completion(
            prompt=prompt,
            purpose=stage,
        )
        content = self._extract_message_content(raw_response)
        indicators = self.parse_indicator_content(
            content=content,
            raw_response=raw_response,
            indicator_ids=indicator_ids,
            original_prompt=prompt,
            log_path=log_path,
        )
        return {
            "city_id": city.city_id,
            "city_name": city.city_name,
            "round": round_number,
            "stage": stage,
            "indicators": indicators,
            "model": self.config.model,
        }

    def parse_indicator_content(
        self,
        content: str,
        raw_response: str,
        indicator_ids: list[str],
        original_prompt: str | None = None,
        log_path: Path | None = None,
    ) -> dict[str, dict[str, float | str]]:
        """Parse model content into indicators -> {score, reasoning}."""

        try:
            indicators = self._parse_indicator_object(content, indicator_ids)
            self._update_log(log_path, extracted_content=content, parsed_json={"indicators": indicators})
            return indicators
        except ValueError as first_error:
            self._update_log(log_path, extracted_content=content, parse_error=str(first_error))
            if original_prompt is None:
                raw_path = self._save_raw_response(raw_response)
                raise ValueError(
                    f"{first_error}. Raw response saved to {raw_path}"
                ) from first_error

            repair_prompt = _build_repair_prompt(content, indicator_ids)
            repair_response, repair_log_path = self._post_chat_completion(
                prompt=repair_prompt,
                purpose="json_repair",
            )
            repair_content = self._extract_message_content(repair_response)
            try:
                repaired = self._parse_indicator_object(repair_content, indicator_ids)
                self._update_log(
                    repair_log_path,
                    extracted_content=repair_content,
                    parsed_json={"indicators": repaired},
                )
                self._update_log(log_path, repair_log=str(repair_log_path))
                return repaired
            except ValueError as repair_error:
                self._update_log(
                    repair_log_path,
                    extracted_content=repair_content,
                    parse_error=str(repair_error),
                )
                raw_path = self._save_raw_response(raw_response)
                raise ValueError(
                    "Model returned invalid JSON and repair failed. "
                    f"Original raw response saved to {raw_path}; repair log: {repair_log_path}"
                ) from repair_error

    def _parse_indicator_object(
        self,
        content: str,
        indicator_ids: list[str],
    ) -> dict[str, dict[str, float | str]]:
        try:
            data = json.loads(extract_first_json_object(content))
        except json.JSONDecodeError as exc:
            raise ValueError("model content did not contain a valid JSON object") from exc

        indicators = data.get("indicators")
        if not isinstance(indicators, dict):
            raise ValueError("model JSON is missing an indicators object")

        parsed: dict[str, dict[str, float | str]] = {}
        for indicator in indicator_ids:
            cell = indicators.get(indicator)
            if not isinstance(cell, dict):
                raise ValueError(f"Model JSON is missing indicator: {indicator}")
            if "score" not in cell or "reasoning" not in cell:
                raise ValueError(f"Indicator {indicator} must include score and reasoning")
            score = float(cell["score"])
            reasoning = str(cell["reasoning"]).strip()
            if not 0 <= score <= 100:
                raise ValueError(f"Indicator {indicator} score must be between 0 and 100")
            if not reasoning:
                raise ValueError(f"Indicator {indicator} reasoning must be non-empty")
            if _is_placeholder_reasoning(reasoning):
                raise ValueError(f"Indicator {indicator} reasoning appears to be a placeholder")
            parsed[indicator] = {
                "score": score,
                "reasoning": reasoning,
            }
        return parsed

    def _post_chat_completion(
        self,
        prompt: str,
        purpose: str,
        max_retries: int = 2,
    ) -> tuple[str, Path]:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        log_path = self._new_call_log_path(purpose)
        self._write_call_log(
            log_path,
            {
                "request": {
                    "purpose": purpose,
                    "base_url": self.config.base_url,
                    "endpoint": "/chat/completions",
                    "model": self.config.model,
                    "timeout": self.config.timeout,
                    "prompt_chars": len(prompt),
                    "has_api_key": bool(self.config.api_key),
                }
            },
        )
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            req = request.Request(
                url=f"{self.config.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.config.timeout) as response:
                    raw = response.read().decode("utf-8")
                    self._update_log(log_path, attempts=attempt, raw_response=raw)
                    return raw, log_path
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                self._update_log(
                    log_path,
                    attempts=attempt,
                    raw_response=error_body,
                    request_error=f"HTTP {exc.code}",
                )
                last_error = exc
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except URLError as exc:
                self._update_log(
                    log_path,
                    attempts=attempt,
                    request_error=f"URL error: {exc.reason}",
                )
                last_error = exc
            except TimeoutError as exc:
                self._update_log(
                    log_path,
                    attempts=attempt,
                    request_error="timeout while waiting for model response",
                )
                last_error = exc
        raise RuntimeError(
            f"LLM request failed after retries. Log saved to {log_path}"
        ) from last_error

    def _extract_message_content(self, raw_response: str) -> str:
        try:
            data = json.loads(raw_response)
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raw_path = self._save_raw_response(raw_response)
            raise ValueError(
                f"Chat completion response shape was not recognized. Raw response saved to {raw_path}"
            ) from exc

    def _save_raw_response(self, raw_response: str) -> Path:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.log_dir / "last_llm_raw_response.txt"
        output_path.write_text(raw_response, encoding="utf-8")
        return output_path

    def _new_call_log_path(self, purpose: str) -> Path:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._call_index += 1
        safe_purpose = "".join(char if char.isalnum() or char in "-_" else "_" for char in purpose)
        return self.config.log_dir / f"llm_call_{int(time() * 1000)}_{self._call_index}_{safe_purpose}.json"

    def _write_call_log(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _update_log(self, path: Path | None, **updates: object) -> None:
        if path is None:
            return
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        data.update({key: value for key, value in updates.items() if value is not None})
        self._write_call_log(path, data)


def _city_anchor_adjustment(anchors: dict[str, float], city_name: str) -> float:
    anchor = anchors.get(city_name)
    if anchor is None:
        return 0.0
    return round((anchor - 0.5) * 6, 2)


def _find_round(rounds: list[AgentRound], agent_id: str) -> AgentRound:
    for agent_round in rounds:
        if agent_round.agent_id == agent_id:
            return agent_round
    raise ValueError(f"missing first-round result for model: {agent_id}")


def _score_for(agent_round: AgentRound, indicator: str) -> IndicatorScore:
    for score in agent_round.scores:
        if score.indicator == indicator:
            return score
    raise ValueError(f"missing score for indicator: {indicator}")


def extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from arbitrary model text."""

    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise json.JSONDecodeError("unterminated JSON object", text, start)


def _build_repair_prompt(content: str, indicator_ids: list[str]) -> str:
    ids = "\n".join(f"- {indicator}" for indicator in indicator_ids)
    return (
        "Repair the following model output into valid JSON only. Do not regenerate "
        "or reinterpret scores or reasoning; only recover fields already present "
        "in the original output. If a reasoning field is missing in the original "
        "content, return {\"repair_error\":\"missing_reasoning\"} rather than "
        "inventing placeholder text. Do not use placeholder reasoning such as "
        "'short reason', 'N/A', or 'not provided'. The required JSON shape is "
        "{\"indicators\":{\"<indicator_id>\":{\"score\":<number>,"
        "\"reasoning\":\"<reasoning recovered from original output>\"}}}. "
        "Required indicator ids:\n"
        f"{ids}\n\nOriginal output:\n{content}"
    )


def _is_placeholder_reasoning(reasoning: str) -> bool:
    normalized = reasoning.strip().lower().rstrip(".")
    return normalized in {
        "short reason",
        "reason",
        "n/a",
        "na",
        "not provided",
        "placeholder",
    }
