"""The AI-feedback source: an LLM grader over an OpenAI-compatible endpoint.

This is deliberately decoupled from the training GPU. The policy trains on your
A100; the grader is *any* OpenAI-compatible server — a hosted API, or a second
`vllm serve` process on another GPU/host. That separation is what lets a single
A100 run GRPO without also paying for the grader's memory.

Config via env (see .env.example):
    GRADER_BASE_URL   e.g. http://localhost:8001/v1   (a `vllm serve` endpoint)
    GRADER_API_KEY    any non-empty string for keyless local servers
    GRADER_MODEL      model name the endpoint serves

Grading a whole GRPO batch is I/O-bound (many short completions), so we fan the
requests out concurrently with a bounded semaphore and grade synchronously from
the reward function via `grade_batch`.
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

    @staticmethod
    def from_env() -> "GraderConfig":
        return GraderConfig(
            base_url=os.environ.get("GRADER_BASE_URL"),
            api_key=os.environ.get("GRADER_API_KEY", "EMPTY"),
            model=os.environ.get("GRADER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            max_tokens=int(os.environ.get("GRADER_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("GRADER_TEMPERATURE", "0.0")),
            max_concurrency=int(os.environ.get("GRADER_MAX_CONCURRENCY", "64")),
        )


class RubricGrader:
    """Scores (conversation, rubric) pairs by asking an LLM for a numeric score."""

    def __init__(self, config: GraderConfig | None = None):
        from openai import AsyncOpenAI

        self.cfg = config or GraderConfig.from_env()
        self._client = AsyncOpenAI(
            base_url=self.cfg.base_url, api_key=self.cfg.api_key
        )
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)

    async def _score_one(self, convo: Conversation, rubric: Rubric) -> float:
        async with self._sem:
            try:
                resp = await self._client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": _GRADER_SYSTEM},
                        {"role": "user", "content": rubric.grader_prompt(convo)},
                    ],
                )
            except Exception as e:  # noqa: BLE001 — a flaky grader call = 0 reward, keep training
                print(f"[grader] request failed, scoring 0.0: {e}")
                return 0.0
        return rubric.extract_score(resp.choices[0].message.content or "")

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

        Returns one mean-rubric score in [0, 1] per completion.
        """
        return asyncio.run(self._grade_batch_async(convos, rubrics))
