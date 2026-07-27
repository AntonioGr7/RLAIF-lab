"""GRPO training with a rubric-based LLM grader — single-A100 recipe.

Design mirrors the tinker_cookbook rubric recipe (policy samples a group of
completions per prompt, an LLM grader scores each against the rubric, reward =
mean rubric score, RL update), but runs locally on open weights via TRL + vLLM.

Two separate models: the POLICY (trained here; generated locally, accelerated by
colocated vLLM) and the GRADER (a remote/OpenAI-compatible server that scores).

Prereqs:
  1. uv pip install --system -e '.[train,vllm]'   # policy generation uses local vLLM
  2. python tasks/addition.py        # writes example_data/addition_{train,test}.jsonl
  3. Stand up a GRADER endpoint (any OpenAI-compatible server; a different model,
     often on another machine):  vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8001
     then:      export GRADER_BASE_URL=http://localhost:8001/v1
                export GRADER_MODEL=Qwen/Qwen3-4B-Instruct-2507

Launch:
  python train.py --config configs/addition.yaml
  python train.py --config configs/addition.yaml --no-use-vllm   # fallback: no local vLLM (slow)

Watch `reward` climb in the compact per-step log (or in wandb with --wandb-project).
"""

from __future__ import annotations

import argparse
import json
import os

import yaml
from datasets import Dataset

from data import load_jsonl
from grader import GraderConfig
from reward import make_rubric_reward


def build_dataset(jsonl_path: str) -> Dataset:
    """One row per datapoint: a conversational `prompt` + serialized rubric(s)."""
    dps = load_jsonl(jsonl_path)
    rows = {
        "prompt": [dp.convo for dp in dps],
        "rubric_json": [json.dumps([r.to_dict() for r in dp.rubric_items]) for dp in dps],
    }
    return Dataset.from_dict(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help="path to a YAML experiment config (see configs/). CLI flags override it.",
    )
    # Model / data
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-jsonl", default="example_data/addition_train.jsonl")
    ap.add_argument("--output-dir", default="outputs/rubric-grpo")
    # GRPO core
    ap.add_argument("--num-generations", type=int, default=8, help="group size G")
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coef (0 = off, as in ref)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed: weight init, data shuffle, sampling")
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-completion-length", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute activations in the backward pass to save VRAM. Default on "
        "(safe for big models / long completions); turn OFF for roughly a third "
        "more throughput when VRAM is plentiful (e.g. a small LoRA on a free A100).",
    )
    # LoRA
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    # Generation backend. vLLM (colocated on the same GPU) is fast but a heavy
    # install; --no-use-vllm falls back to plain transformers generation (slower,
    # no vllm dependency). The grader server is unrelated to this flag.
    ap.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use colocated vLLM for generation (needs the `vllm` extra)",
    )
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.3)
    # Logging
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=40)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument(
        "--log-completions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print TRL's per-step completion tables (noisy; off by default)",
    )
    return ap


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top-level YAML must be a mapping of keys.")
    return cfg


def main() -> None:
    # Pre-parse just --config so its values can seed argparse defaults. Precedence
    # ends up: CLI flag > config file > code default.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict = {}
    grader_overrides: dict = {}
    if pre_args.config:
        cfg = _load_config(pre_args.config)
        grader_overrides = cfg.pop("grader", {}) or {}  # handled separately below

    ap = build_parser()
    # All argument dests, via the public API (parse an empty argv) rather than the
    # private ap._actions. Every arg has a default, so this never errors.
    known = set(vars(ap.parse_known_args([])[0]))
    # `test_jsonl` is an eval-only key in the shared config (eval.py reads it);
    # train doesn't use a test set, but must tolerate the key rather than reject it.
    known.add("test_jsonl")
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(
            f"unknown keys in {pre_args.config}: {sorted(unknown)}\n"
            f"valid keys: {sorted(known - {'help', 'config'})}"
        )
    ap.set_defaults(**cfg)  # config overrides code defaults
    args = ap.parse_args()  # CLI overrides config
    if pre_args.config:
        print(f"[config] loaded {pre_args.config}")

    # Fail fast on the divisibility rule TRL enforces internally, with a clear msg.
    effective = args.per_device_batch * args.grad_accum
    if effective % args.num_generations != 0:
        raise SystemExit(
            f"per_device_batch*grad_accum ({effective}) must be divisible by "
            f"num_generations ({args.num_generations}); "
            f"{effective // args.num_generations if args.num_generations else 0} "
            f"prompts/step would be uneven."
        )

    # Imported here so data generators / the grader don't need the heavy train stack.
    import dataclasses

    import trl
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    class CompactLogCallback(TrainerCallback):
        """One tight line per step instead of TRL's 25-key metrics dict."""

        def __init__(self, total):
            self.total = total

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "reward" not in logs:
                return  # skip non-training logs (final summary, eval, ...)
            f = lambda k, d=3: f"{logs[k]:.{d}f}" if k in logs else "-"
            line = (
                f"step {state.global_step:>4}/{self.total}  "
                f"reward {f('reward')}  std {f('reward_std')}  "
                f"zero_std {f('frac_reward_zero_std', 2)}  "
                f"grad {f('grad_norm', 2)}  loss {f('loss', 4)}"
            )
            if "kl" in logs:
                line += f"  kl {f('kl', 4)}"
            print(line, flush=True)

    # Grader config: start from env, then apply any `grader:` block from the YAML.
    grader_cfg = GraderConfig.from_env()
    for k, v in grader_overrides.items():
        if not hasattr(grader_cfg, k):
            raise SystemExit(
                f"unknown grader config key {k!r}; "
                f"valid: {sorted(vars(grader_cfg))}"
            )
        setattr(grader_cfg, k, v)

    # Reproducibility: dump the *effective* run config next to the adapter, so a
    # directory in outputs/ is self-describing — which grader, which hyperparameters,
    # which library versions produced this adapter. Without it, an adapter is an
    # orphan you can't re-derive. api_key is redacted (it may be a real hosted key).
    import transformers

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "args": vars(args),
        "grader": {
            k: ("<redacted>" if k == "api_key" else v)
            for k, v in vars(grader_cfg).items()
        },
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__},
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[train] wrote run manifest -> {args.output_dir}/run_config.json")

    train_ds = build_dataset(args.train_jsonl)
    reward_fn = make_rubric_reward(grader_cfg)

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        # sampling / groups
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        # optimization
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.use_vllm,
        # logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_completions=args.log_completions,
        num_completions_to_print=4,
        report_to="wandb" if args.wandb_project else "none",
        run_name="rubric-grpo",
    )
    if args.use_vllm:
        # colocated on the same GPU as training; only touched when vLLM is on.
        grpo_kwargs["vllm_mode"] = "colocate"
        grpo_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_mem
    else:
        print("[train] vLLM off — using transformers generation (slower).")

    # Prompt-length control moved/was removed across TRL versions: only pass
    # max_prompt_length when this GRPOConfig actually supports it (TRL <1.9),
    # instead of adding-then-dropping it (which spammed a warning every run).
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    if "max_prompt_length" in valid:
        grpo_kwargs["max_prompt_length"] = args.max_prompt_length

    # Safety net for any *other* field drift: drop unknown kwargs loudly rather
    # than crash. With max_prompt_length handled above, this should normally be
    # empty on a supported TRL.
    dropped = sorted(set(grpo_kwargs) - valid)
    for k in dropped:
        print(f"[train] note: trl {trl.__version__} GRPOConfig has no '{k}'; dropping it.")
    grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in valid}
    grpo_config = GRPOConfig(**grpo_kwargs)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=lora_config,
    )

    # Replace TRL/transformers' noisy default logging (the 25-key metrics dict
    # printer) with one compact reward-focused line per step. Keep the tqdm bar
    # unless completion tables are on (they clash with the bar).
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CompactLogCallback(total=args.max_steps))

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
