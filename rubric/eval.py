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

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_jsonl
from grader import GraderConfig, RubricGrader


def generate(model, tok, convo, max_new_tokens: int) -> str:
    # return_dict=True -> {input_ids, attention_mask}; newer transformers no longer
    # returns a bare tensor here, so pass the dict through with **inputs.
    inputs = tok.apply_chat_template(
        convo, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="training config to reuse (grader block + policy model)")
    ap.add_argument("--model", default=None, help="policy model (overrides config)")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument("--test-jsonl", default="example_data/addition_test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--num-samples-to-print", type=int, default=6)
    args = ap.parse_args()

    # Grader: env first, then a config's grader block on top (single source of truth).
    grader_cfg = GraderConfig.from_env()
    cfg_model = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg_model = cfg.get("model")
        for k, v in (cfg.get("grader") or {}).items():
            if hasattr(grader_cfg, k):
                setattr(grader_cfg, k, v)
        print(f"[eval] grader: {grader_cfg.model} @ {grader_cfg.base_url or 'default(OpenAI)'}")

    # Policy model precedence: --model > config model > fallback default.
    model_name = args.model or cfg_model or "Qwen/Qwen2.5-1.5B-Instruct"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    dps = load_jsonl(args.test_jsonl)[: args.limit]
    convos, rubrics, completions = [], [], []
    for dp in dps:
        answer = generate(model, tok, dp.convo, args.max_new_tokens)
        completions.append(answer)
        convos.append(dp.convo + [{"role": "assistant", "content": answer}])
        rubrics.append(dp.rubric_items)

    scores = RubricGrader(grader_cfg).grade_batch(convos, rubrics)
    mean = sum(scores) / len(scores) if scores else 0.0

    tag = f"adapter={args.adapter}" if args.adapter else "base model"
    print("=" * 70)
    for dp, ans, sc in list(zip(dps, completions, scores))[: args.num_samples_to_print]:
        q = dp.convo[-1]["content"]
        print(f"  score={sc:.2f}  Q: {q!r}  ->  A: {ans!r}")
    print("=" * 70)
    print(f"{tag}: mean rubric score over {len(scores)} examples = {mean:.3f}")


if __name__ == "__main__":
    main()
