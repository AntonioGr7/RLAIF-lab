"""GRPO training with a rubric-based LLM grader — single-A100 recipe.

Design mirrors the tinker_cookbook rubric recipe (policy samples a group of
completions per prompt, an LLM grader scores each against the rubric, reward =
mean rubric score, RL update), but runs locally on open weights via TRL + vLLM.

Prereqs:
  1. uv sync --extra train           # installs trl, vllm, peft, ...
  2. python tasks/addition.py        # writes example_data/addition_{train,test}.jsonl
  3. Stand up a grader endpoint (any OpenAI-compatible server), e.g. on another
     GPU/host:  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
     then:      export GRADER_BASE_URL=http://localhost:8001/v1
                export GRADER_MODEL=Qwen/Qwen2.5-7B-Instruct

Launch:
  uv run python train.py                       # defaults below
  uv run python train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 150

Watch `reward` climb and `loss` move in the logs (or in wandb with --wandb-project).
"""

from __future__ import annotations

import argparse
import json

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
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-completion-length", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=120)
    # LoRA
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    # vLLM generation (colocated on the same A100)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.3)
    # Logging
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=40)
    ap.add_argument("--wandb-project", default=None)
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
    known = {a.dest for a in ap._actions}
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
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    # Grader config: start from env, then apply any `grader:` block from the YAML.
    grader_cfg = GraderConfig.from_env()
    for k, v in grader_overrides.items():
        if not hasattr(grader_cfg, k):
            raise SystemExit(
                f"unknown grader config key {k!r}; "
                f"valid: {sorted(vars(grader_cfg))}"
            )
        setattr(grader_cfg, k, v)

    train_ds = build_dataset(args.train_jsonl)
    reward_fn = make_rubric_reward(grader_cfg)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        # sampling / groups
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # optimization
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        # fast generation, colocated with training on one GPU
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        # logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_completions=True,
        num_completions_to_print=4,
        report_to="wandb" if args.wandb_project else "none",
        run_name="rubric-grpo",
    )
    if args.wandb_project:
        import os

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
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
