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

# Install INTO the image's env so the preinstalled torch 2.8.0 is reused
# (no multi-GB reinstall, no CUDA-build mismatch):
uv pip install --system -e '.[train]'

# --- or, isolated venv instead (uv pulls torch 2.8.0+cu128 fresh): ---
# uv sync --extra train      # then prefix the commands below with `uv run`

# 1) data
python tasks/addition.py                 # -> example_data/addition_{train,test}.jsonl

# 2) grader endpoint (a second GPU/host, or CPU for a tiny model)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 &
export GRADER_BASE_URL=http://localhost:8001/v1
export GRADER_MODEL=Qwen/Qwen2.5-7B-Instruct
export GRADER_API_KEY=EMPTY

# 3) baseline behavior
python eval.py                            # base model, mean rubric score

# 4) train (config-driven; CLI flags still override individual values)
python train.py --config configs/addition.yaml
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

## Notes & knobs

- **Experiment configs.** Put one YAML per experiment in [configs/](configs/) and
  launch with `--config`. Precedence is **CLI flag > config file > code default**,
  so you can pin an experiment in YAML and still tweak one value from the command
  line. Keys map 1:1 to `train.py` flags (underscores); a `grader:` block
  configures the grader endpoint and overrides the env vars.
  [configs/realistic_template.yaml](configs/realistic_template.yaml) is a starting
  point for a real task (KL on, longer completions, stronger grader).
- **Single-GPU memory.** vLLM runs *colocated* (`vllm_mode="colocate"`) and is
  capped at `--vllm-gpu-mem 0.3` of VRAM; the rest is for LoRA training. Drop the
  fraction if you hit OOM, or move generation to a second GPU.
- **Group size.** `--num-generations G` is the GRPO group. `per_device_batch *
  grad_accum` must be divisible by `G` (the script checks and explains if not).
- **KL.** `--beta 0.0` matches the reference (no KL penalty). Raise it (e.g. `0.02`)
  if the policy drifts or degenerates.
- **Grader ≠ ground truth.** For addition the grader is near-perfect, which is why
  it's a good *demo*. On real tasks the rubric moves subjectivity from "is this
  good?" to "are these the right criteria?" — smaller, but not zero. Audit grader
  rationales and watch for reward hacking.
- **Harder/easier.** `tasks/addition.py --max-operand 999` reproduces the
  reference's easy setting; raise it for more headroom.
