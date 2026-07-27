"""Task: addition. Generates the `a + b` demo dataset.

One file per task lives in this folder; each builds a jsonl of RubricDatapoints
that train.py / eval.py consume. To add a task, copy this file, change the
`convo` and `rubric_items`, and write to a new jsonl.

The task, straight from the reference recipe: the policy is asked `What is a + b?`
and must answer with the number. We *could* check the answer with `==`, but the
whole point of rubric-based RLAIF is to route the reward through an LLM grader —
here the grader is simply asked "Does the chatbot get the answer N?".

We use larger operands than the reference (default up to 5 digits) so a small
local policy is wrong often enough at step 0 to leave a clearly visible learning
curve. Tune `--max-operand` down for an easier task, up for a harder one.

    python tasks/addition.py                       # 4000 train / 400 test
    python tasks/addition.py --max-operand 999     # easy mode (like the ref)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Make the recipe root importable when run as `python tasks/addition.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import Rubric, RubricDatapoint, save_jsonl  # noqa: E402

SYSTEM = "You are a calculator. Reply with only the final integer, nothing else."


def make_one(rng: random.Random, max_operand: int) -> RubricDatapoint:
    a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
    return RubricDatapoint(
        convo=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"What is {a} + {b}?"},
        ],
        rubric_items=[
            Rubric(
                rubric_str=(
                    f"Does the chatbot's completion give the correct answer, "
                    f"which is {a + b}? Award 1 only if the number {a + b} is the "
                    f"stated result; otherwise award 0."
                ),
                grader_output_format_instruction=(
                    "Output <score>1</score> if correct, else <score>0</score>."
                ),
            )
        ],
        # gold lets eval.py run an independent exact-match check (grader vs truth),
        # not shown to the policy/grader. Kept consistent with multiplication.py.
        meta={"a": a, "b": b, "gold": a + b},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=4000)
    ap.add_argument("--num-test", type=int, default=400)
    ap.add_argument("--max-operand", type=int, default=99_999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="example_data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.num_train + args.num_test
    data = [make_one(rng, args.max_operand) for _ in range(n)]

    train_path = f"{args.out_dir}/addition_train.jsonl"
    test_path = f"{args.out_dir}/addition_test.jsonl"
    save_jsonl(data[: args.num_train], train_path)
    save_jsonl(data[args.num_train :], test_path)
    print(f"wrote {args.num_train} train -> {train_path}")
    print(f"wrote {args.num_test} test  -> {test_path}")


if __name__ == "__main__":
    main()
