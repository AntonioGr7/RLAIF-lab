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

## Add your own task — a worked example

Nothing in the training/grader code is task-specific: a task is just a `tasks/*.py`
that writes a jsonl of `RubricDatapoint`s, plus a config YAML. To show the full loop
on something different from the shipped arithmetic/RAG tasks, we'll build
**product-review sentiment classification with a grounded justification**: given a
review, the policy must answer `POSITIVE` or `NEGATIVE` and justify in one line by
quoting the review. It's a good example because it has two criteria (so it exercises
the merged multi-rubric grader call) *and* a mechanical ground truth (the gold label),
which is what lets you watch for reward hacking — here, a policy that writes a
confident, grader-pleasing justification while quietly always guessing the majority
label.

**1. Model the datapoint as `(conversation, rubric_items, meta)`.** The conversation
is the prompt the policy continues; the rubric items are what the grader scores; `meta`
is ground truth shown to *neither* policy nor grader — it exists only so eval can check
the model independently. Two rubrics, both on the default `<score>`/0–1 contract so
they auto-merge into one grader call:

```python
# tasks/sentiment.py  (mirror the sys.path shim + save_jsonl pattern from the others)
from data import Rubric, RubricDatapoint, save_jsonl

SYSTEM = ("Classify the review sentiment. Reply on one line: the word POSITIVE or "
          "NEGATIVE, then a short reason that quotes the review.")

def make_datapoint(review: str, gold_label: str) -> RubricDatapoint:
    return RubricDatapoint(
        convo=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": review}],
        rubric_items=[
            Rubric(rubric_str=("Award 1 if the stated sentiment label is correct for "
                               "this review, else 0."),
                   grader_output_format_instruction="Output <score>1</score> or <score>0</score>."),
            Rubric(rubric_str=("Award 1 only if the justification quotes or paraphrases "
                               "specific wording from the review (not a generic reason)."),
                   grader_output_format_instruction="Output <score>1</score> or <score>0</score>."),
        ],
        meta={"gold_label": gold_label},   # ground truth — eval only, never shown
    )
```

Note the first rubric is **reference-free**: the grader judges correctness from the
review itself, and the gold label stays in `meta`. Build train/test jsonl from any
labelled source (e.g. `datasets.load_dataset("stanfordnlp/sst2")`, mapping label
1→POSITIVE / 0→NEGATIVE) exactly as `tasks/rag_groundedness.py` does, writing
`example_data/sentiment_{train,test}.jsonl`.

**2. Teach eval its ground truth** so the periodic capability watch works. Add one
branch to `capability_score` in [eval.py](eval.py) for your `meta` shape:

```python
if isinstance(meta0.get("gold_label"), str):
    hits = sum(dp.meta["gold_label"].lower() in a.lower() for dp, a in zip(dps, completions))
    return hits / len(dps)
```

Without this the run still trains, but the `[eval] capability=…` line is skipped —
and that line is the whole point on a task where the grader can be gamed.

**3. Write `configs/sentiment.yaml`.** Keys map 1:1 to `train.py` flags. Give the
completion enough room for a label + one-line reason, and point the grader at your
endpoint:

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
train_jsonl: example_data/sentiment_train.jsonl
test_jsonl:  example_data/sentiment_test.jsonl   # used by eval + the periodic capability check
output_dir:  outputs/sentiment-grpo
num_generations: 8
per_device_batch: 16
grad_accum: 4
max_completion_length: 64            # label + a one-line quote
mask_truncated_completions: true     # don't let the length cap masquerade as reward
loss_type: dr_grpo
scale_rewards: none
beta: 0.0
max_steps: 200
eval_steps: 20                       # watch capability vs reward
grader:
  model: Qwen/Qwen3-4B-Instruct-2507
  enable_thinking: false
```

**4. Run the loop.** Same three commands as the shipped tasks:

```bash
python tasks/sentiment.py                                   # -> example_data/sentiment_{train,test}.jsonl
python eval.py --config configs/sentiment.yaml              # baseline label accuracy + grader score
python train.py --config configs/sentiment.yaml             # train; watch [grader] reward vs [eval] capability
python eval.py --config configs/sentiment.yaml --adapter outputs/sentiment-grpo
```

**What to watch.** If the `[grader]` reward climbs while `[eval] capability` (label
accuracy) stays flat or falls, the policy is gaming the justification rubric rather
than classifying better — exactly the failure this recipe is built to surface. The fix
is on the *rubric* side (tighten the correctness criterion, add a criterion that
penalises hedging), not the RL side. That is the whole workflow: the grader turns
"is this good?" into "are these the right criteria?", and the independent capability
metric keeps that honest.
