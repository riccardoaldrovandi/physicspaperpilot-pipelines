# Pipeline A — answering from the paper

The model reads the paper and answers the benchmark questions directly. This is the reference
condition: Pipeline B is measured against it.

## How it works

For every paper and every question, one request goes to the model:

1. **The paper.** Its text plus its figures as images (up to 12). From the PDF, the pages that hold
   figures or tables are rendered as images. From the LaTeX source, the figure files are used, in
   the order the source includes them. If the text is too long for the context window, the middle
   is cut and the start and end are kept.
2. **The question**, inside the benchmark's own prompt (`03_pilot_run/prompt_for_tested_models.md`).
   The model answers with `ANSWER:` and `EVIDENCE:` (the section headings it relied on).

The paper part is identical for all the questions of a paper. A rerun skips the questions that
already have an answer and retries the ones that failed.

## Files

| File | What it does |
|---|---|
| `answer_from_paper.py` | Answers the questions of all papers (or `--papers`), writes the raw answers (`.jsonl`) and the benchmark's predictions CSV |
| `run_pipeline_a.sbatch` | Leonardo job: starts the model, runs every config and seed in parallel, records energy use |

It relies on `common/` for the paper inputs, the model client and the cluster scripts.

## Running it

On Leonardo, from the repository root (one-time setup in the main README):

```bash
sbatch --export=ALL,MODEL=qwen38 pipeline_a/run_pipeline_a.sbatch
sbatch --export=ALL,MODEL=mistral35,CONFIGS=nothink pipeline_a/run_pipeline_a.sbatch
sbatch --export=ALL,MODEL=gemma4,SOURCE=latex pipeline_a/run_pipeline_a.sbatch
```

`MODEL` is `qwen38`, `gemma4`, `nemotron35` or `mistral35`. `CONFIGS` defaults to `think nothink`
(Mistral: `nothink` only), `SEEDS` to `0 1 2`. Other options are listed in `common/cluster/job.sh`.

Without SLURM, with a vLLM server already running and the variables of `common/cluster/models.sh`
set:

```bash
python pipeline_a/answer_from_paper.py --benchmark-dir benchmark/PhysicsPaperPilotEval_v0.3.2 \
    --output runs/pdf/qwen38_nothink_s0_A.jsonl --label "qwen38|nothink|s0|A_direct_pdf"
```

## Output

In `runs/<pdf|latex>/`:

- `<model>_<config>_s<seed>_A.jsonl` — one line per question: the raw answer, the parsed answer and
  evidence, the input hash, the number of images, whether the text was truncated, the sampling settings.
- `<model>_<config>_s<seed>_A.csv` — the same answers in the benchmark's predictions format, ready
  for `evaluation/`.
