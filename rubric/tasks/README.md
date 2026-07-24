# tasks/

One dataset generator per task. Each script builds a jsonl of `RubricDatapoint`s
(see [../data.py](../data.py)) that `train.py` / `eval.py` consume via the
`train_jsonl` / `test_jsonl` paths in a [config](../configs).

- [addition.py](addition.py) — the `a + b` demo task (verifiable, single criterion).
- [rag_groundedness.py](rag_groundedness.py) — RAG groundedness + faithful
  abstention from SQuAD v2 (reference-free, two criteria, answerable/unanswerable
  mix). A realistic example; see [../configs/rag_groundedness.yaml](../configs/rag_groundedness.yaml).

## Adding a task

1. Copy `addition.py` to `tasks/<your_task>.py`.
2. Change `convo` (the prompt the policy continues) and `rubric_items` (the
   criteria the grader scores). Use several `Rubric(...)` items for
   multi-criterion grading — the reward is their mean.
3. Write to a new jsonl (e.g. `example_data/<your_task>_{train,test}.jsonl`).
4. Point a config at it: `train_jsonl: example_data/<your_task>_train.jsonl`
   (copy [../configs/realistic_template.yaml](../configs/realistic_template.yaml)).

The rubric *text* is the thing you're optimizing toward — it lives here in the
generator (baked into each datapoint's `rubric_str`), not in the training code.
Regenerate the jsonl after editing it.
