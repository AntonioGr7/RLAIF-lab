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
import atexit
import json
import os
import threading
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
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_tokens: int = 1024  # room for a thinking model's reasoning + verdict
    temperature: float = 0.0
    max_concurrency: int = 64
    # Robustness knobs
    max_retries: int = 6  # SDK-level retries w/ exponential backoff on 429/5xx
    timeout: float | None = None  # per-request seconds; None = SDK default
    on_error: str = "raise"  # "raise" (fail loud) | "zero" (score 0.0 and continue)
    reasoning: bool = False  # o-series/reasoning models: use max_completion_tokens
    # For open thinking models (Qwen3, ...) served via vLLM: None = server default,
    # False = disable the <think> block for a short, fast, unambiguous verdict.
    # Sent as chat_template_kwargs; only vLLM/compatible servers honor it.
    enable_thinking: bool | None = None

    @staticmethod
    def from_env() -> "GraderConfig":
        timeout = os.environ.get("GRADER_TIMEOUT")
        thinking = os.environ.get("GRADER_ENABLE_THINKING")
        return GraderConfig(
            base_url=os.environ.get("GRADER_BASE_URL"),
            api_key=os.environ.get("GRADER_API_KEY", "EMPTY"),
            model=os.environ.get("GRADER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            max_tokens=int(os.environ.get("GRADER_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("GRADER_TEMPERATURE", "0.0")),
            max_concurrency=int(os.environ.get("GRADER_MAX_CONCURRENCY", "64")),
            max_retries=int(os.environ.get("GRADER_MAX_RETRIES", "6")),
            timeout=float(timeout) if timeout else None,
            on_error=os.environ.get("GRADER_ON_ERROR", "raise"),
            reasoning=os.environ.get("GRADER_REASONING", "").lower()
            in ("1", "true", "yes"),
            enable_thinking=None
            if thinking is None
            else thinking.lower() in ("1", "true", "yes"),
        )


@dataclass
class GraderStats:
    """Per-batch counters, so a degrading teacher is visible, not silent."""

    calls: int = 0  # actual grader API calls made (one per distinct datapoint grading)
    graded: int = 0  # criteria scored across those calls (denominator for parse rates)
    request_failures: int = 0  # exceptions after all retries
    unparsed: int = 0  # HTTP ok but no parseable <score> for a criterion
    out_of_range: int = 0  # score parsed but off-scale (clamped) — grader drifted
    deduped: int = 0  # datapoint gradings served from an identical (convo, rubrics) — calls saved


class RubricGrader:
    """Scores (conversation, rubric) pairs by asking an LLM for a numeric score."""

    def __init__(self, config: GraderConfig | None = None):
        self.cfg = config or GraderConfig.from_env()
        if self.cfg.on_error not in ("raise", "zero"):
            raise ValueError(f"on_error must be 'raise' or 'zero', got {self.cfg.on_error!r}")
        self._stats = GraderStats()
        self.last_stats = GraderStats()
        # A persistent event loop in a dedicated daemon thread. grade_batch() is
        # called once per training step; the AsyncOpenAI client and its HTTP
        # connection pool are built ONCE on this loop and reused across all steps,
        # so TCP/TLS handshakes aren't repaid for every batch. (asyncio.run() per
        # step would open+close a fresh loop, and a client can't cross loops —
        # "bound to a different loop" — so the pool couldn't survive.)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client = None
        self._sem: asyncio.Semaphore | None = None
        self._start_lock = threading.Lock()

    def _ensure_loop(self) -> None:
        """Lazily start the loop thread and build the client + semaphore on it."""
        if self._loop is not None:
            return
        with self._start_lock:
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="grader-loop", daemon=True
            )
            thread.start()
            # Build client + semaphore ON the loop thread so both bind to it.
            asyncio.run_coroutine_threadsafe(self._build_client(), loop).result()
            self._loop, self._thread = loop, thread
            atexit.register(self.close)

    async def _build_client(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(**self._client_kwargs())
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)

    def close(self) -> None:
        """Close the client and stop the loop thread (best-effort, idempotent)."""
        loop = self._loop
        if loop is None:
            return
        self._loop = None
        try:
            if self._client is not None:
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), loop
                ).result(timeout=10)
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass
        finally:
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)

    def _client_kwargs(self) -> dict:
        kw: dict = {
            "base_url": self.cfg.base_url,
            "api_key": self.cfg.api_key,
            "max_retries": self.cfg.max_retries,
        }
        if self.cfg.timeout is not None:
            kw["timeout"] = self.cfg.timeout
        return kw

    def _request_params(self, messages: list[dict]) -> dict:
        """Build create() kwargs, accounting for reasoning-model quirks."""
        params: dict = {"model": self.cfg.model, "messages": messages}
        if self.cfg.reasoning:
            # o-series/reasoning models: no temperature, token cap is named differently.
            params["max_completion_tokens"] = self.cfg.max_tokens
        else:
            params["max_tokens"] = self.cfg.max_tokens
            params["temperature"] = self.cfg.temperature
        if self.cfg.enable_thinking is not None:
            # vLLM passes chat_template_kwargs to the model's chat template;
            # Qwen3 etc. read enable_thinking to toggle the <think> block.
            params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}
            }
        return params

    async def _grade_call(self, prompt: str) -> str | None:
        """One grader HTTP request. Returns the reply text, or None if the request
        failed after retries and on_error='zero'. Raises on on_error='raise'."""
        messages = [
            {"role": "system", "content": _GRADER_SYSTEM},
            {"role": "user", "content": prompt},
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
                return None
        return resp.choices[0].message.content or ""

    def _tally(self, result) -> float:
        """Fold one criterion's ScoreResult into the stats and a numeric reward."""
        self._stats.graded += 1
        if result.score is None:
            # HTTP ok but no parseable score — a legitimate 0 reward (non-compliant
            # answer), counted so a high unparsed rate flags a grader/format problem.
            self._stats.unparsed += 1
            return 0.0
        if result.out_of_range:
            # Off the rubric's scale, clamped to a boundary — a high rate means the
            # grader slipped onto a different scale and the reward is saturating.
            self._stats.out_of_range += 1
        return result.score

    async def _score_datapoint(self, convo: Conversation, rubrics: list[Rubric]) -> float:
        """Mean score over a datapoint's rubric items.

        The context is sent once in a single MERGED call when it's safe to — more
        than one rubric AND all use the default <score>/0-1 contract the merged
        prompt assumes (RAG groundedness qualifies). This halves input tokens on
        such tasks. Otherwise each rubric is graded on its own single-criterion
        prompt, which honors a custom extraction_regex or scale — so a rubric with
        a bespoke contract stays correct, just without the token saving. A single
        rubric always takes this path, byte-identical to the pre-merge recipe.

        A failed request (on_error='zero' -> None reply) scores that criterion 0.0
        without being counted as unparsed; the failure is already tallied in
        _grade_call.
        """
        if not rubrics:
            return 0.0
        if len(rubrics) > 1 and all(r.is_merge_safe() for r in rubrics):
            reply = await self._grade_call(Rubric.grader_prompt_multi(convo, rubrics))
            if reply is None:
                return 0.0
            scores = [self._tally(r) for r in Rubric.extract_scores_multi(reply, rubrics)]
        else:
            replies = await asyncio.gather(
                *(self._grade_call(r.grader_prompt(convo)) for r in rubrics)
            )
            scores = [
                0.0 if reply is None else self._tally(r.extract_score(reply))
                for r, reply in zip(rubrics, replies)
            ]
        return sum(scores) / len(scores)

    @staticmethod
    def _dedup_key(convo: Conversation, rubrics: list[Rubric]) -> tuple[str, str]:
        """Identity of one datapoint grading: the exact (conversation, rubric set)
        the grader conditions on.

        Keying on the *full conversation* — not just the completion text — is
        essential: in RAG groundedness the rubric set is shared across every
        datapoint, and the discriminating signal is the passage in the prior turns,
        so a completion-only key would collapse distinct gradings.
        """
        return (
            json.dumps(convo, sort_keys=True, ensure_ascii=False),
            json.dumps([r.to_dict() for r in rubrics], sort_keys=True, ensure_ascii=False),
        )

    async def _grade_batch_async(
        self, convos: list[Conversation], rubrics: list[list[Rubric]]
    ) -> list[float]:
        # Runs on the persistent loop thread; client + semaphore were built there
        # once (see _ensure_loop) and are reused across steps.
        # One grading call per datapoint (all its criteria in a single request),
        # deduplicated by (conversation, rubric set). A GRPO group samples G
        # completions per prompt; on short outputs (arithmetic answers are a few
        # tokens) many are byte-identical, so the same grading recurs across the
        # group. We grade each distinct one once and fan the score back out —
        # cutting grader requests (the dominant per-step cost) with no change to
        # the rewards, since the grader is deterministic at temperature 0 (default).
        keys: list[tuple[str, str]] = []
        key_input: dict[tuple[str, str], tuple[Conversation, list[Rubric]]] = {}
        for convo, rubric_list in zip(convos, rubrics):
            key = self._dedup_key(convo, rubric_list)
            keys.append(key)
            key_input.setdefault(key, (convo, rubric_list))

        unique = list(key_input)
        self._stats.deduped = len(keys) - len(unique)

        scored = await asyncio.gather(
            *(self._score_datapoint(*key_input[k]) for k in unique)
        )
        key_score = dict(zip(unique, scored))
        return [key_score[k] for k in keys]

    def grade_batch(
        self, convos: list[Conversation], rubrics: list[list[Rubric]]
    ) -> list[float]:
        """Synchronous entry point for the (sync) TRL reward function.

        Returns one mean-rubric score in [0, 1] per completion. With
        ``on_error="raise"`` (default), a persistent grader outage raises here and
        aborts the step rather than feeding the trainer corrupted rewards.
        """
        self._stats = GraderStats()
        self._ensure_loop()
        # Submit onto the persistent loop and block this (the trainer's) thread
        # until the batch is graded — same synchronous contract as before, but the
        # loop and connection pool live across steps instead of per call.
        scores = asyncio.run_coroutine_threadsafe(
            self._grade_batch_async(convos, rubrics), self._loop
        ).result()
        self.last_stats = self._stats
        self._report(scores)
        return scores

    def _report(self, scores: list[float]) -> None:
        s = self._stats
        if not s.calls:
            return
        mean = sum(scores) / len(scores) if scores else 0.0
        fail_rate = s.request_failures / s.calls
        # Parse-quality rates are per-criterion (a merged call scores several), so
        # they divide by criteria graded, not by API calls.
        graded = s.graded or 1
        unparsed_rate = s.unparsed / graded
        out_of_range_rate = s.out_of_range / graded
        extra = ""
        if s.request_failures:
            extra += f"  request_failures={s.request_failures} ({fail_rate:.1%})"
        if s.unparsed:
            extra += f"  unparsed={s.unparsed} ({unparsed_rate:.1%})"
        if s.out_of_range:
            extra += f"  out_of_range={s.out_of_range} ({out_of_range_rate:.1%})"
        dedup_note = ""
        if s.deduped:
            total = s.calls + s.deduped
            dedup_note = f"  (deduped {s.deduped}/{total}, {s.deduped / total:.0%} fewer calls)"
        print(f"[grader] {s.calls} calls{dedup_note}  reward_mean={mean:.3f}{extra}")
        if unparsed_rate > 0.2:
            print(
                f"[grader] WARNING: {unparsed_rate:.0%} of replies had no parseable "
                f"<score> — check the grader model or the format instruction."
            )
        if out_of_range_rate > 0.2:
            print(
                f"[grader] WARNING: {out_of_range_rate:.0%} of scores were off the "
                f"rubric's [min_score, max_score] scale and got clamped — the grader "
                f"is likely using a different scale; the reward is saturating."
            )
