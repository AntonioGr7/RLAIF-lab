"""Bridge the rubric grader into TRL's GRPO reward-function interface.

TRL calls a reward function as ``fn(prompts, completions, **extra_columns)`` where
each argument is a list aligned to the batch of sampled completions, and returns
a list of floats. We reconstruct the full conversation for each sample (prompt
turns + the sampled assistant turn), hand it to the LLM grader, and return the
mean rubric score — exactly the signal the reference recipe's env produces.
"""

from __future__ import annotations

import json

from data import Rubric
from grader import GraderConfig, RubricGrader


def _completion_text(completion) -> str:
    """Normalize a TRL completion to plain text (handles chat- or text-format)."""
    if isinstance(completion, str):
        return completion
    # conversational format: list of {"role","content"} message dicts
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def _prompt_convo(prompt) -> list[dict]:
    """Normalize a TRL prompt to a list of message dicts."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def make_rubric_reward(config: GraderConfig | None = None):
    """Return a TRL-compatible reward function closed over a single grader.

    The returned function's ``__name__`` is what TRL logs the reward under
    (``rewards/rubric_reward``), so keep it stable.
    """
    grader = RubricGrader(config)

    def rubric_reward(prompts, completions, **kwargs):
        rubric_json = kwargs["rubric_json"]  # extra dataset column, one per sample
        convos, rubrics = [], []
        for prompt, completion, rj in zip(prompts, completions, rubric_json):
            convo = _prompt_convo(prompt) + [
                {"role": "assistant", "content": _completion_text(completion)}
            ]
            convos.append(convo)
            rubrics.append([Rubric.from_dict(d) for d in json.loads(rj)])
        return grader.grade_batch(convos, rubrics)

    return rubric_reward
