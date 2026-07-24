"""Task: RAG groundedness / faithful abstention.

Given a retrieved passage + a question, the policy must answer using ONLY the
passage — and, crucially, say "I can't find that in the context" when the passage
doesn't contain the answer, instead of hallucinating.

Source: SQuAD v2 (`rajpurkar/squad_v2`). We use v2 specifically because it labels
**unanswerable** questions (passage is on-topic but the answer is absent) — that's
what lets us reward the model's ability to abstain, not just to extract. We build
a balanced mix of answerable and unanswerable datapoints; the balance matters,
because a set that is all-unanswerable would train the model to *always* say "I
don't know" (reward-hacking the abstention criterion).

Grading is **reference-free**: the grader never sees the gold answer. It must
judge, from the passage + the model's answer alone, (1) whether every claim is
grounded in the passage and (2) whether the model answered-or-abstained
appropriately. The gold answer / answerability is stored in `meta` (not shown to
policy or grader) only so an independent eval can measure abstention accuracy.

    python tasks/rag_groundedness.py                      # 4000 train / 400 test, 50/50
    python tasks/rag_groundedness.py --answerable-ratio 0.5 --num-train 8000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import Rubric, RubricDatapoint, save_jsonl  # noqa: E402

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the information needed to answer, reply that "
    "you cannot find the answer in the context. Do not use outside knowledge and "
    "do not guess."
)

# Reference-free rubric: no gold answer is embedded. The grader reasons from the
# passage (which it sees via the flattened conversation) and the model's answer.
GROUNDED = Rubric(
    rubric_str=(
        "Consider only the context shown above. Award 1 if every factual claim in "
        "the chatbot's answer is directly supported by that context. Award 0 if the "
        "answer includes any fact that is not supported by the context."
    ),
    grader_output_format_instruction=(
        "Reason briefly, then output <score>1</score> or <score>0</score>."
    ),
)
APPROPRIATE = Rubric(
    rubric_str=(
        "First decide, from the context alone, whether it actually contains the "
        "information needed to answer the question. Then: if it DOES, award 1 only "
        "if the chatbot provides that answer; if it does NOT, award 1 only if the "
        "chatbot states that the answer cannot be found in the context (rather than "
        "guessing or fabricating). Award 0 otherwise."
    ),
    grader_output_format_instruction=(
        "Reason briefly, then output <score>1</score> or <score>0</score>."
    ),
)


def make_datapoint(context: str, question: str, gold: str | None) -> RubricDatapoint:
    return RubricDatapoint(
        convo=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        rubric_items=[GROUNDED, APPROPRIATE],
        # eval-only; never shown to policy or grader
        meta={"answerable": gold is not None, "gold": gold},
    )


def _split_by_answerability(rows) -> tuple[list, list]:
    answerable, unanswerable = [], []
    for r in rows:
        texts = r["answers"]["text"]
        (answerable if texts else unanswerable).append(r)
    return answerable, unanswerable


def build_split(rows, n: int, answerable_ratio: float, rng: random.Random) -> list:
    answerable, unanswerable = _split_by_answerability(rows)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    n_ans = round(n * answerable_ratio)
    n_unans = n - n_ans
    if n_ans > len(answerable) or n_unans > len(unanswerable):
        raise SystemExit(
            f"not enough rows: need {n_ans} answerable / {n_unans} unanswerable, "
            f"have {len(answerable)} / {len(unanswerable)}"
        )
    picks = answerable[:n_ans] + unanswerable[:n_unans]
    rng.shuffle(picks)
    return [
        make_datapoint(
            context=r["context"],
            question=r["question"],
            gold=(r["answers"]["text"][0] if r["answers"]["text"] else None),
        )
        for r in picks
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=4000)
    ap.add_argument("--num-test", type=int, default=400)
    ap.add_argument("--answerable-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="example_data")
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    ds = load_dataset("rajpurkar/squad_v2")
    # SQuAD's test labels aren't public, so carve train/test from the two splits.
    train = build_split(ds["train"], args.num_train, args.answerable_ratio, rng)
    test = build_split(ds["validation"], args.num_test, args.answerable_ratio, rng)

    train_path = f"{args.out_dir}/rag_groundedness_train.jsonl"
    test_path = f"{args.out_dir}/rag_groundedness_test.jsonl"
    save_jsonl(train, train_path)
    save_jsonl(test, test_path)
    n_ans = sum(dp.meta["answerable"] for dp in train)
    print(f"wrote {len(train)} train ({n_ans} answerable / {len(train) - n_ans} unanswerable) -> {train_path}")
    print(f"wrote {len(test)} test -> {test_path}")


if __name__ == "__main__":
    main()
