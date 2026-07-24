"""Task: multiplication. A harder arithmetic task that leaves a learning curve.

Addition is trivial for modern small instruct models — they score ~1.0 at step 0,
so every GRPO group is all-correct, variance is zero, and there is no gradient
(see the addition smoke test). Multiplication is the opposite: small models get it
*partially* right, so within a group of samples some are correct and some aren't
— which is exactly the within-group reward variance GRPO needs to learn.

Goal when picking digit sizes: base accuracy somewhere in the ~20-70% band. Too
easy (all right) and too hard (all wrong) both give zero variance / zero signal.
Calibrate with the baseline eval before a long run:

    python tasks/multiplication.py --max-a 99 --max-b 999
    python eval.py --test-jsonl example_data/multiplication_test.jsonl   # check base acc
    # then adjust --max-a/--max-b so the mean rubric score is mid-range.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import Rubric, RubricDatapoint, save_jsonl  # noqa: E402

SYSTEM = "You are a calculator. Reply with only the final integer, nothing else."


def make_one(rng: random.Random, max_a: int, max_b: int) -> RubricDatapoint:
    a, b = rng.randint(2, max_a), rng.randint(2, max_b)
    return RubricDatapoint(
        convo=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"What is {a} * {b}?"},
        ],
        rubric_items=[
            Rubric(
                rubric_str=(
                    f"Does the chatbot's completion give the correct answer, "
                    f"which is {a * b}? Award 1 only if the number {a * b} is the "
                    f"stated result; otherwise award 0."
                ),
                grader_output_format_instruction=(
                    "Output <score>1</score> if correct, else <score>0</score>."
                ),
            )
        ],
        meta={"a": a, "b": b, "gold": a * b},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=4000)
    ap.add_argument("--num-test", type=int, default=400)
    ap.add_argument("--max-a", type=int, default=99, help="first operand upper bound")
    ap.add_argument("--max-b", type=int, default=999, help="second operand upper bound")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="example_data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.num_train + args.num_test
    data = [make_one(rng, args.max_a, args.max_b) for _ in range(n)]

    train_path = f"{args.out_dir}/multiplication_train.jsonl"
    test_path = f"{args.out_dir}/multiplication_test.jsonl"
    save_jsonl(data[: args.num_train], train_path)
    save_jsonl(data[args.num_train :], test_path)
    print(f"wrote {args.num_train} train -> {train_path}")
    print(f"wrote {args.num_test} test  -> {test_path}")
    print(f"difficulty: [2..{args.max_a}] * [2..{args.max_b}] "
          f"— calibrate so baseline accuracy lands ~20-70%")


if __name__ == "__main__":
    main()
