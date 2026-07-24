"""The AI-feedback source: an LLM grader over an OpenAI-compatible endpoint.

Works against a local `vllm serve` process *or* hosted OpenAI (leave
`GRADER_BASE_URL` unset to hit api.openai.com). Because the grader *is* the reward
function, it's hardened for a paid/rate-limited backend:

* **Retries with backoff** on 429/5xx/timeout (via the OpenAI SDK's `max_retries`).
* **Fail loud, not silent.** A request that still fails after retries does NOT get
  scored 0.0 by default (`on_error="raise"`) — a dropped grader call is a
  false-negative reward that quietly corrupts training. Set `on_error="zero"` to
  fall back to 0.0 instead of aborting.
* **Reasoning-model path.** o-series / reasoning models reject `max_tokens` +
  `temperature`; set `reasoning=true` to send `max_completion_tokens` and drop
  `temperature`.
* **Failure-rate reporting.** Each batch prints request-failure and
  unparseable-reply rates so you can see if the teacher is degrading.

Config via env (see .env.example) or a `grader:` block in a config YAML.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from data import Conversation, Rubric

_GRADER_SYSTEM = (
    "You are a strict, consistent grader. Follow the rubric exactly and always "
    "emit a score in the requested format."
)


@dataclass
class GraderConfig:
    base_url: str | None = None
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.0
    max_concurrency: int = 64
    # Robustness knobs
    max_retries: int = 6  # SDK-level retries w/ exponential backoff on 429/5xx
    timeout: float | None = None  # per-request seconds; None = SDK default
    on_error: str = "raise"  # "raise" (fail loud) | "zero" (score 0.0 and continue)
    reasoning: bool = False  # o-series/reasoning models: use max_completion_tokens

    @staticmethod
    def from_env() -> "GraderConfig":
        timeout = os.environ.get("GRADER_TIMEOUT")
        return GraderConfig(
            base_url=os.environ.get("GRADER_BASE_URL"),
            api_key=os.environ.get("GRADER_API_KEY", "EMPTY"),
            model=os.environ.get("GRADER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            max_tokens=int(os.environ.get("GRADER_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("GRADER_TEMPERATURE", "0.0")),
            max_concurrency=int(os.environ.get("GRADER_MAX_CONCURRENCY", "64")),
            max_retries=int(os.environ.get("GRADER_MAX_RETRIES", "6")),
            timeout=float(timeout) if timeout else None,
            on_error=os.environ.get("GRADER_ON_ERROR", "raise"),
            reasoning=os.environ.get("GRADER_REASONING", "").lower()
            in ("1", "true", "yes"),
        )


@dataclass
class GraderStats:
    """Per-batch counters, so a degrading teacher is visible, not silent."""

    calls: int = 0
    request_failures: int = 0  # exceptions after all retries
    unparsed: int = 0  # HTTP ok but no parseable <score> in the reply


class RubricGrader:
    """Scores (conversation, rubric) pairs by asking an LLM for a numeric score."""

    def __init__(self, config: GraderConfig | None = None):
        from openai import AsyncOpenAI

        self.cfg = config or GraderConfig.from_env()
        if self.cfg.on_error not in ("raise", "zero"):
            raise ValueError(f"on_error must be 'raise' or 'zero', got {self.cfg.on_error!r}")
        client_kwargs: dict = {
            "base_url": self.cfg.base_url,
            "api_key": self.cfg.api_key,
            "max_retries": self.cfg.max_retries,
        }
        if self.cfg.timeout is not None:
            client_kwargs["timeout"] = self.cfg.timeout
        self._client = AsyncOpenAI(**client_kwargs)
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._stats = GraderStats()
        self.last_stats = GraderStats()

    def _request_params(self, messages: list[dict]) -> dict:
        """Build create() kwargs, accounting for reasoning-model quirks."""
        params: dict = {"model": self.cfg.model, "messages": messages}
        if self.cfg.reasoning:
            # o-series/reasoning models: no temperature, token cap is named differently.
            params["max_completion_tokens"] = self.cfg.max_tokens
        else:
            params["max_tokens"] = self.cfg.max_tokens
            params["temperature"] = self.cfg.temperature
        return params

    async def _score_one(self, convo: Conversation, rubric: Rubric) -> float:
        messages = [
            {"role": "system", "content": _GRADER_SYSTEM},
            {"role": "user", "content": rubric.grader_prompt(convo)},
        ]
        async with self._sem:
            self._stats.calls += 1
            try:
                resp = await self._client.chat.completions.create(
                    **self._request_params(messages)
                )
            except Exception as e:  # noqa: BLE001 — retries already exhausted by the SDK
                self._stats.request_failures += 1
                if self.cfg.on_error == "raise":
                    raise RuntimeError(
                        f"grader request failed after {self.cfg.max_retries} "
                        f"retries (model={self.cfg.model}): {e}"
                    ) from e
                print(f"[grader] request failed after retries, scoring 0.0: {e}")
                return 0.0

        score = rubric.extract_score(resp.choices[0].message.content or "")
        if score is None:
            # HTTP succeeded but the grader didn't emit a parseable score. That's a
            # legitimate 0 reward (non-compliant answer), but we count it: a high
            # unparsed rate means the grader/format instruction needs fixing.
            self._stats.unparsed += 1
            return 0.0
        return score

    async def _score_datapoint(
        self, convo: Conversation, rubrics: list[Rubric]
    ) -> float:
        """Mean score across all rubric items for one completion."""
        scores = await asyncio.gather(*(self._score_one(convo, r) for r in rubrics))
        return sum(scores) / len(scores) if scores else 0.0

    async def _grade_batch_async(
        self, convos: list[Conversation], rubrics: list[list[Rubric]]
    ) -> list[float]:
        return list(
            await asyncio.gather(
                *(self._score_datapoint(c, r) for c, r in zip(convos, rubrics))
            )
        )

    def grade_batch(
        self, convos: list[Conversation], rubrics: list[list[Rubric]]
    ) -> list[float]:
        """Synchronous entry point for the (sync) TRL reward function.

        Returns one mean-rubric score in [0, 1] per completion. With
        ``on_error="raise"`` (default), a persistent grader outage raises here and
        aborts the step rather than feeding the trainer corrupted rewards.
        """
        self._stats = GraderStats()
        scores = asyncio.run(self._grade_batch_async(convos, rubrics))
        self.last_stats = self._stats
        self._report(scores)
        return scores

    def _report(self, scores: list[float]) -> None:
        s = self._stats
        if not s.calls:
            return
        mean = sum(scores) / len(scores) if scores else 0.0
        fail_rate = s.request_failures / s.calls
        unparsed_rate = s.unparsed / s.calls
        extra = ""
        if s.request_failures:
            extra += f"  request_failures={s.request_failures} ({fail_rate:.1%})"
        if s.unparsed:
            extra += f"  unparsed={s.unparsed} ({unparsed_rate:.1%})"
        print(f"[grader] {s.calls} calls  reward_mean={mean:.3f}{extra}")
        if unparsed_rate > 0.2:
            print(
                f"[grader] WARNING: {unparsed_rate:.0%} of replies had no parseable "
                f"<score> — check the grader model or the format instruction."
            )
