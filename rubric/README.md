# Rubric-Based RLAIF — single-A100 GRPO recipe

A small, well-factored reference implementation of **rubric-based RLAIF**: train a
policy with RL where the reward comes from an **LLM grader** scoring each sampled
response against an explicit **rubric**.

Design is ported from the Thinking Machines
[`tinker_cookbook` rubric recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/rubric),
which runs on the hosted Tinker API. This version keeps the same shape but runs
**locally on a single A100** using open weights + [TRL](https://github.com/huggingface/trl)
GRPO + vLLM, so you can launch it on your own box.

## The loop

```
                                       ┌──────── group of G samples ────────┐
  prompt ──▶ policy (Qwen, LoRA) ──────┤ a₁  a₂  …  a_G                      │
   ▲                                   └──────────────────┬─────────────────┘
   │                                                      │  each sample
   │                          ┌───────────────────────────▼──────────────────┐
   │      rubric ────────────▶│   LLM GRADER (OpenAI-compatible endpoint)      │
   │  (per datapoint)         │   "Does the answer meet the rubric?" -> score  │
   │                          └───────────────────────────┬──────────────────┘
   │                                                       │ reward = mean rubric score
   └──────────────── GRPO update (advantage = reward − group mean) ◀──────────┘
```

GRPO samples `G` completions per prompt, grades each, and pushes the policy toward
the above-average ones. The grader is a **separate** OpenAI-compatible server, so
the A100 is spent on the policy, not the grader.

## Files

| File | Role |
|------|------|
| [data.py](data.py) | `Rubric` + `RubricDatapoint`: criterion text, grader output-format instruction, score-extraction regex, jsonl I/O |
| [tasks/](tasks/) | One dataset generator per task; [tasks/addition.py](tasks/addition.py) builds the **addition** demo (`a + b`) |
| [grader.py](grader.py) | Async, batched rubric grader over any OpenAI-compatible endpoint |
| [reward.py](reward.py) | Wraps the grader as a TRL GRPO reward function |
| [train.py](train.py) | GRPO + LoRA + colocated vLLM training loop |
| [eval.py](eval.py) | Before/after mean rubric score + sample completions |
| [configs/](configs/) | Per-experiment YAML (`--config`); keys map 1:1 to `train.py` flags, plus a `grader:` block |

## The concrete example: addition

Straight from the reference. The policy is asked `What is a + b?` and must reply
with the number. We *could* check with `==`, but the point of rubric-based RLAIF
is to route reward through an LLM grader — so the grader is asked *"Does the
chatbot get the answer N?"* and returns `<score>1</score>` / `<score>0</score>`.

We use **larger operands than the reference** (up to 5 digits by default) so a
small local policy is wrong often enough at step 0 to leave a clearly visible
learning curve. It's the cleanest task for a demo: reward is unambiguous, the
signal is dense, and the behavior change (verbose/wrong → terse correct integer)
is easy to eyeball.

## Run it on the A100

Target host: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
(torch 2.8.0 + CUDA 12.8, Python 3.11 preinstalled).

```bash
cd rubric

# Standard install — INTO the image's env so the preinstalled torch 2.8.0 is
# reused (no multi-GB reinstall, no CUDA-build mismatch). vLLM is included
# because GRPO generates many completions per step and colocated vLLM is the
# normal, fast way to do that — it accelerates the POLICY generation on this GPU
# (separate from the grader server, which is remote/OpenAI-compatible):
uv pip install --system -e '.[train,vllm]'

# Fallback only for machines that can't build vLLM: install the lighter stack and
# train with --no-use-vllm (plain transformers generation — much slower):
#   uv pip install --system -e '.[train]'   &&   python train.py ... --no-use-vllm

# --- or, isolated venv instead (uv pulls torch 2.8.0+cu128 fresh): ---
# uv sync --extra train --extra vllm      # then prefix the commands below with `uv run`

# 1) data
python tasks/addition.py                 # -> example_data/addition_{train,test}.jsonl

# 2) grader endpoint — run `vllm serve` where the grader model lives (this box,
#    another GPU, or another machine) and point the trainer at it. `vllm serve`
#    is a standalone tool; it does NOT require the `vllm` extra in this project.
#    IMPORTANT (single-GPU): `vllm serve` defaults to --gpu-memory-utilization 0.9,
#    which leaves no room for the colocated policy vLLM (vllm_gpu_mem: 0.3 in the
#    configs) — you'll OOM. Cap the grader so both fit: 0.35 + 0.30 leaves ~0.35
#    for the trainer itself. If the grader runs on a SEPARATE GPU/host, drop the
#    flag and let it use the default 0.9.
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8001 --gpu-memory-utilization 0.35 &
export GRADER_BASE_URL=http://localhost:8001/v1     # or http://<other-host>:8001/v1
export GRADER_MODEL=Qwen/Qwen3-4B-Instruct-2507
export GRADER_API_KEY=EMPTY

# 3) baseline behavior
python eval.py                            # base model, mean rubric score

# 4) train (config-driven; CLI flags still override individual values)
python train.py --config configs/addition.yaml
# no local vLLM installed? fall back to transformers generation:
#   python train.py --config configs/addition.yaml --no-use-vllm
# e.g. override on the fly:  python train.py --config configs/addition.yaml --max-steps 200

# 5) trained behavior
python eval.py --adapter outputs/rubric-grpo
```

See [.env.example](.env.example) for grader configuration.

### What "working" looks like

- **`reward`** in the TRL logs (or wandb, via `--wandb-project`) rises from a low
  baseline toward ~1.0 as the policy learns to emit the correct sum.
- **`loss`** is the GRPO policy-gradient surrogate; it's noisy and can be small or
  negative — read it alongside `reward` and `kl`, not on its own. The headline
  signal for "is it learning" is **reward going up**.
- **`eval.py`** shows the mean rubric score jump between base and trained, and the
  printed samples go from rambling/incorrect to a bare correct integer.
- **`[eval] capability=…`** lines during training (every `--eval-steps`) are the
  grader-free ground-truth accuracy on a held-out slice. Watch them against the
  reward: **reward rising while capability falls is reward hacking** — the policy
  is gaming the grader, not getting better. This is the phenomenon the recipe
  exists to expose, so don't judge a run by final reward alone.

## Notes & knobs

- **Experiment configs.** Put one YAML per experiment in [configs/](configs/) and
  launch with `--config`. Precedence is **CLI flag > config file > code default**,
  so you can pin an experiment in YAML and still tweak one value from the command
  line. Keys map 1:1 to `train.py` flags (underscores); a `grader:` block
  configures the grader endpoint and overrides the env vars.
  [configs/realistic_template.yaml](configs/realistic_template.yaml) is a starting
  point for a real task (KL on, longer completions, stronger grader).
- **Two models, two jobs.** The **policy** (trained here) is generated *locally* —
  colocated vLLM (`vllm_mode="colocate"`) accelerates that on the training GPU, and
  it's the standard/fast path (transformers via `--no-use-vllm` is a slow fallback).
  The **grader** is a *separate* model on a remote/OpenAI-compatible server. They
  never share weights or hardware — don't confuse "the remote vLLM" (grader) with
  "local vLLM" (policy generation).
- **Single-GPU memory.** Colocated vLLM is capped at `--vllm-gpu-mem 0.3` of VRAM;
  the rest is LoRA training. Since the grader is remote, the whole GPU is free for
  the policy — bump the fraction for a bigger generation batch, or drop it on OOM.
- **Grader latency = GPU idle.** The step is `generate → grade → update`, and the
  policy update needs the batch's rewards, so the grader sits on the critical path:
  while it runs, the training GPU is idle. TRL's reward interface is synchronous, so
  hiding that latency behind the *next* batch's generation would require a custom
  training loop — deliberately out of scope here. Instead the recipe shrinks the
  idle window: the grader dedups identical completions (big win on short arithmetic
  groups) and holds one persistent HTTP connection pool across steps (no per-step
  TCP/TLS handshakes). To shrink it further, keep the grader **local/colocated**
  rather than a remote 35B, set `GRADER_ENABLE_THINKING=false` for a short verdict,
  and raise `GRADER_MAX_CONCURRENCY` until the grader server saturates. The
  per-step `[grader]` log line reports call count and dedup savings so you can see
  the window.
- **Group size.** `--num-generations G` is the GRPO group. `per_device_batch *
  grad_accum` must be divisible by `G` (the script checks and explains if not).
- **KL.** `--beta 0.0` matches the reference (no KL penalty). Raise it (e.g. `0.02`)
  if the policy drifts or degenerates.
- **Grader outage → resume, don't restart.** The grader defaults to `on_error="raise"`
  (a persistent 429/5xx aborts the step rather than feeding a corrupted 0 reward). The
  trainer checkpoints every `save_steps`, so recover with
  `python train.py --config … --resume-from-checkpoint` (bare flag = latest checkpoint,
  or pass a specific `outputs/…/checkpoint-N`). A failure at step 180/200 costs you the
  last few steps, not the whole run.
- **Grader ≠ ground truth.** For addition the grader is near-perfect, which is why
  it's a good *demo*. On real tasks the rubric moves subjectivity from "is this
  good?" to "are these the right criteria?" — smaller, but not zero. Audit grader
  rationales and watch for reward hacking.
- **Harder/easier.** `tasks/addition.py --max-operand 999` reproduces the
  reference's easy setting; raise it for more headroom.
