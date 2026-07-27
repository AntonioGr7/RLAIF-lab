"""Evaluate a policy on the test set with the same rubric grader used in training.

Run it before and after training to see behavior reach what we want:

    python eval.py --config configs/multiplication.yaml            # base model
    python eval.py --config configs/multiplication.yaml \
        --adapter outputs/multiplication-grpo                      # trained LoRA

Pass --config to reuse a training config's `grader:` block (and its policy
`model`), so the grader endpoint is defined in exactly one place. CLI flags still
override the config; without --config the grader falls back to GRADER_* env vars.

Reports the mean rubric score over the test set and prints a few sample
completions so you can eyeball *how* the answers changed, not just the number.
"""

from __future__ import annotations

import argparse
import re

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_jsonl
from grader import GraderConfig, RubricGrader


# Abstention detection for the RAG groundedness eval — the anti-Goodhart metric.
# It must NOT fire on grounded *content* answers that merely contain a phrase like
# "did not contain" ("The treaty did not contain a clause about ..."), which plain
# substring matching over cue words does. So we match the abstention *speech act*:
# an inability/absence marker bound to the ANSWER or to the CONTEXT itself — the
# form the task's system prompt asks for ("reply that you cannot find the answer in
# the context").
_ABSTAIN_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # (I) cannot / can't / could not / am unable to  ...  find|answer|locate|...
        r"\b(cannot|can'?t|couldn'?t|could not|unable to|not able to)\b[^.]{0,40}\b(find|answer|determine|locate|provide|tell)\b",
        # the answer/information ... cannot ... be found|determined|answered
        r"\b(answer|information|it)\b[^.]{0,25}\b(cannot|can'?t|couldn'?t|could not)\b[^.]{0,15}\bbe (found|determined|answered)\b",
        # the context/passage/text ... does not|no ... contain|mention|provide|state|...
        r"\b(context|passage|text|document|information)\b[^.]{0,25}\b(does not|doesn'?t|do not|don'?t|no)\b[^.]{0,15}\b(contain|mention|provide|include|specify|state|indicat|give)",
        # not ... in the context/passage/text/document
        r"\bnot\b[^.]{0,30}\bin the (context|passage|text|document)\b",
        # explicit no-answer / unanswerable
        r"\bno (answer|information|mention|indication|details?)\b",
        r"\b(un-?answerable|not answerable|cannot be answered)\b",
        # (I) don't know
        r"\b(do not|does not|don'?t|doesn'?t)\s+know\b",
    )
)


def _abstained(text: str) -> bool:
    """True if the reply declines to answer from the context (an abstention).

    Regex over the abstention *speech act* rather than substring cues: it needs an
    inability/absence marker tied to the answer or the context, so grounded content
    answers ("The treaty did not contain a clause ...") aren't misread as
    abstentions. Heuristic, not perfect — but reliable in exactly the ambiguous
    cases this metric exists to catch.
    """
    return any(p.search(text) for p in _ABSTAIN_PATTERNS)


def _exact_match(gold: int, text: str) -> bool:
    """True if the reply's *final* stated integer equals the gold answer.

    We take the last integer (thousands separators stripped) rather than "gold
    appears anywhere": a reply that echoes the operands ("42 * 17 = 13") would
    otherwise match whenever gold happens to equal an operand. The last integer
    is the model's stated result, matching the system prompt ("only the final
    integer") and the grader's own "find the number stated as its result".
    """
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return bool(ints) and int(ints[-1]) == gold


def generate_batch(
    model, tok, convos: list, max_new_tokens: int, batch_size: int
) -> list[str]:
    """Greedily complete each conversation, in batches.

    Batched with **left** padding: decoder-only generation requires it so every
    prompt in a batch ends at the same position and one uniform slice removes them.
    Greedy decoding with the attention mask makes each completion identical to the
    one-at-a-time version — this is purely a throughput win (5-10x on this step),
    not a behavior change. The caller must set ``tok.padding_side = "left"`` and a
    pad token (see main).
    """
    completions: list[str] = []
    for start in range(0, len(convos), batch_size):
        chunk = convos[start : start + batch_size]
        # return_dict=True -> {input_ids, attention_mask}; padding=True left-pads
        # the batch to the longest prompt.
        inputs = tok.apply_chat_template(
            chunk,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]  # uniform: batch is left-padded
        completions.extend(
            tok.decode(row, skip_special_tokens=True).strip()
            for row in out[:, prompt_len:]
        )
    return completions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="training config to reuse (grader block + policy model)")
    ap.add_argument("--model", default=None, help="policy model (overrides config)")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument(
        "--test-jsonl",
        default=None,
        help="test set (overrides config's test_jsonl; default: addition_test.jsonl)",
    )
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32, help="completions generated per forward batch")
    ap.add_argument("--num-samples-to-print", type=int, default=6)
    args = ap.parse_args()

    # Grader: env first, then a config's grader block on top (single source of truth).
    grader_cfg = GraderConfig.from_env()
    cfg_model = None
    cfg_test = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg_model = cfg.get("model")
        cfg_test = cfg.get("test_jsonl")
        for k, v in (cfg.get("grader") or {}).items():
            if hasattr(grader_cfg, k):
                setattr(grader_cfg, k, v)
        print(f"[eval] grader: {grader_cfg.model} @ {grader_cfg.base_url or 'default(OpenAI)'}")

    # Policy model precedence: --model > config model > fallback default.
    model_name = args.model or cfg_model or "Qwen/Qwen2.5-1.5B-Instruct"
    # Test-set precedence, same rule: --test-jsonl > config test_jsonl > fallback.
    # Without this, --config only steered the model/grader while the test set
    # stayed pinned to the addition default — silently evaluating one task's
    # config on another task's data.
    test_jsonl = args.test_jsonl or cfg_test or "example_data/addition_test.jsonl"
    print(f"[eval] test set: {test_jsonl}")

    tok = AutoTokenizer.from_pretrained(model_name)
    # Left padding is required for batched decoder-only generation (see
    # generate_batch); fall back to EOS as the pad token if the model lacks one.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    dps = load_jsonl(test_jsonl)[: args.limit]
    completions = generate_batch(
        model, tok, [dp.convo for dp in dps], args.max_new_tokens, args.eval_batch_size
    )
    convos = [
        dp.convo + [{"role": "assistant", "content": a}]
        for dp, a in zip(dps, completions)
    ]
    rubrics = [dp.rubric_items for dp in dps]

    scores = RubricGrader(grader_cfg).grade_batch(convos, rubrics)
    mean = sum(scores) / len(scores) if scores else 0.0

    tag = f"adapter={args.adapter}" if args.adapter else "base model"
    print("=" * 70)
    for dp, ans, sc in list(zip(dps, completions, scores))[: args.num_samples_to_print]:
        q = dp.convo[-1]["content"]
        print(f"  score={sc:.2f}  Q: {q!r}  ->  A: {ans!r}")
    print("=" * 70)
    print(f"{tag}: mean rubric score over {len(scores)} examples = {mean:.3f}")

    # Independent check for arithmetic tasks (meta carries an integer `gold`).
    # The mean rubric score above is the grader's own opinion of the completions —
    # on its own that's "the grader grading itself". Here we score the SAME
    # completions against ground truth mechanically and report how often the grader
    # agrees with truth, so an unreliable teacher is caught before it's trusted as
    # the reward signal.
    gold0 = dps[0].meta.get("gold") if dps else None
    if isinstance(gold0, int):
        n = len(dps)
        exact = [_exact_match(dp.meta["gold"], a) for dp, a in zip(dps, completions)]
        exact_acc = sum(exact) / n if n else 0.0
        agree = sum((s >= 0.5) == e for s, e in zip(scores, exact)) / n if n else 0.0
        false_pos = sum((s >= 0.5) and not e for s, e in zip(scores, exact))
        false_neg = sum((s < 0.5) and e for s, e in zip(scores, exact))
        print("-" * 70)
        print("independent check (exact match vs gold, no grader):")
        print(f"  exact-match accuracy     = {exact_acc:.3f}  (n={n})")
        print(f"  grader agrees with truth = {agree:.3f}")
        print(f"  grader false-positives   = {false_pos}  (rewarded a wrong number)")
        print(f"  grader false-negatives   = {false_neg}  (missed a correct number)")
        print("  (low agreement => the teacher itself is unreliable; fix the rubric/grader before trusting the reward)")

    # Independent metric (no LLM grader): for tasks that tag datapoints with
    # `answerable` in meta (RAG groundedness), report answer/abstention accuracy
    # mechanically. This is the Goodhart check — a model that hacks the reward by
    # ALWAYS abstaining shows high unanswerable-abstention but low answerable-acc.
    if dps and dps[0].meta.get("answerable") is not None:
        ans = [(dp, a) for dp, a in zip(dps, completions) if dp.meta.get("answerable")]
        unans = [(dp, a) for dp, a in zip(dps, completions) if not dp.meta.get("answerable")]

        def frac(pairs, pred):
            return sum(pred(dp, a) for dp, a in pairs) / len(pairs) if pairs else 0.0

        def answered_correct(dp, a):
            gold = (dp.meta.get("gold") or "").strip().lower()
            return gold != "" and gold in a.lower() and not _abstained(a)

        print("-" * 70)
        print("independent check (no grader):")
        print(f"  answerable   (n={len(ans)}):   correct answer   = {frac(ans, answered_correct):.3f}")
        print(f"  answerable   (n={len(ans)}):   wrongly abstained = {frac(ans, lambda d, a: _abstained(a)):.3f}")
        print(f"  unanswerable (n={len(unans)}): correctly abstained = {frac(unans, lambda d, a: _abstained(a)):.3f}")
        print("  (reward-hacking tell: high 'unanswerable abstained' + high 'answerable wrongly abstained')")


if __name__ == "__main__":
    main()
