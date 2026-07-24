"""Data model for rubric-based RLAIF.

Ported from the Thinking Machines `tinker_cookbook` rubric recipe, trimmed to be
self-contained (no tinker dependency).

A datapoint is a *conversation prefix* the policy must continue, plus a list of
*rubric items*. Each rubric item carries: the criterion text shown to the grader,
the output-format instruction, and the regex used to pull a numeric score out of
the grader's reply. Keeping the extraction contract next to the criterion is what
makes the grader swappable — any LLM that follows the format instruction works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# A message is the usual chat dict; a conversation is a list of them.
Message = dict[str, str]
Conversation = list[Message]


@dataclass
class Rubric:
    """One gradable criterion + how to read the grader's score back out."""

    rubric_str: str
    extraction_regex: str = r"<score>(.*?)</score>"
    grader_output_format_instruction: str = (
        "First reason briefly, then output your score as a number between 0 and 1 "
        "wrapped in <score> ... </score>."
    )
    # Score range, used to normalize the extracted value into [0, 1].
    min_score: float = 0.0
    max_score: float = 1.0

    @staticmethod
    def _role_label(role: str) -> str:
        return "Human" if role in ("user", "system") else "Chatbot"

    def _flatten(self, convo: Conversation) -> str:
        return "\n\n".join(
            f"{self._role_label(m['role'])}: {m['content']}" for m in convo
        )

    def grader_prompt(self, convo: Conversation) -> str:
        """Render the single user-turn prompt shown to the grader LLM.

        The prior turns are given as context; only the final assistant message is
        the completion under grading — this keeps the grader focused on the most
        recent response rather than the whole transcript.
        """
        context, completion = convo[:-1], convo[-1]
        return "\n".join(
            [
                "You are grading a chatbot's most recent completion against a rubric.",
                "",
                "<context>",
                self._flatten(context) if context else "(no prior context)",
                "</context>",
                "",
                "<completion_to_grade>",
                f"Chatbot: {completion['content']}",
                "</completion_to_grade>",
                "",
                "<rubric>",
                self.rubric_str,
                "</rubric>",
                "",
                f"Grade the completion against the rubric. "
                f"{self.grader_output_format_instruction}",
            ]
        )

    def extract_score(self, grader_reply: str) -> float | None:
        """Pull the score out of the grader reply and normalize to [0, 1].

        Returns ``None`` when no numeric score could be found — the caller decides
        what that means. This is deliberately distinct from a real ``0.0`` score:
        "the grader said 0" and "the grader ignored the format" are different
        events, and only the caller can see how often the latter happens.

        We take the LAST match, not the first: graders often reason out loud and
        echo the format instruction (which itself contains ``<score>...</score>``)
        before stating the final verdict. The verdict is the last score tag.
        """
        matches = re.findall(self.extraction_regex, grader_reply, re.DOTALL)
        if not matches:
            return None
        try:
            raw = float(matches[-1].strip())
        except ValueError:
            return None
        span = self.max_score - self.min_score
        if span <= 0:
            return None
        return max(0.0, min(1.0, (raw - self.min_score) / span))

    def to_dict(self) -> dict:
        return {
            "rubric_str": self.rubric_str,
            "extraction_regex": self.extraction_regex,
            "grader_output_format_instruction": self.grader_output_format_instruction,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

    @staticmethod
    def from_dict(d: dict) -> "Rubric":
        return Rubric(
            rubric_str=d["rubric_str"],
            extraction_regex=d.get("extraction_regex", r"<score>(.*?)</score>"),
            grader_output_format_instruction=d.get(
                "grader_output_format_instruction",
                Rubric.grader_output_format_instruction,
            ),
            min_score=d.get("min_score", 0.0),
            max_score=d.get("max_score", 1.0),
        )


@dataclass
class RubricDatapoint:
    """A conversation prefix the policy continues + the rubric(s) grading it.

    ``meta`` holds task bookkeeping (e.g. gold answer, answerability) that is NOT
    shown to the policy or the grader — it exists purely so an independent eval
    can score the model without trusting the training grader. Training ignores it.
    """

    convo: Conversation
    rubric_items: list[Rubric] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "convo": self.convo,
                "rubric_items": [r.to_dict() for r in self.rubric_items],
                "meta": self.meta,
            }
        )

    @staticmethod
    def from_json(line: str) -> "RubricDatapoint":
        d = json.loads(line)
        return RubricDatapoint(
            convo=d["convo"],
            rubric_items=[Rubric.from_dict(r) for r in d["rubric_items"]],
            meta=d.get("meta", {}),
        )


def load_jsonl(path: str | Path) -> list[RubricDatapoint]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first:\n"
            f"  python tasks/addition.py"
        )
    with open(path) as f:
        return [RubricDatapoint.from_json(line) for line in f if line.strip()]


def save_jsonl(datapoints: list[RubricDatapoint], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for dp in datapoints:
            f.write(dp.to_json() + "\n")
