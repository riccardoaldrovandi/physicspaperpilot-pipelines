# Pipeline B — answering from a structured summary

The model first turns the paper into a structured JSON summary, then answers the questions reading
only that JSON. Comparing it with Pipeline A shows how much information the summary keeps.

## How it works

**Step B1 — the summary** (`build_json.py`). One request per paper, with the same inputs as
Pipeline A (text and figures). The system prompt (`prompt.py`, version 2.1) asks for a JSON with a
fixed schema: summary, problem, motivation, contributions, method, data and setup, key results with
values and units, equations, tables, figures with their curves, limitations, future work. The answer
is decoded against that schema, placeholder values ("None mentioned") are normalized, and the JSON
is saved together with the exact input the model saw. Answers that are not valid JSON are kept
apart as rejected.

**Step B2 — the answers** (`answer_from_json.py`). One request per question: the benchmark's JSON
prompt (`03_pilot_run/prompt_json_only.md`) with the paper's JSON and the question. The model
returns its answer, the JSON fields it used, and whether it abstained ("NOT RECOVERABLE FROM
STRUCTURED SUMMARY") because the JSON lacks the information.

**Condition C — fixed reader.** The same B2 script, with one reader (Qwen3.8) answering from the
JSONs of every model. It separates the quality of a summary from the skill of its reader.

## Files

| File | What it does |
|---|---|
| `prompt.py` | The v2.1 prompt, the JSON schema built from its field layout, the rules for empty values |
| `build_json.py` | Step B1: one JSON per paper |
| `answer_from_json.py` | Step B2 (and C): answers from a JSON file |
| `run_pipeline_b.sbatch` | Leonardo job: starts the model, runs B1 then B2 for every config and seed, or C |

It relies on `common/` for the paper inputs, the model client and the cluster scripts.

## Running it

On Leonardo, from the repository root:

```bash
sbatch --export=ALL,MODEL=qwen38 pipeline_b/run_pipeline_b.sbatch                 # B1 then B2
sbatch --export=ALL,MODEL=mistral35,CONFIGS=nothink pipeline_b/run_pipeline_b.sbatch
# once all four models have their JSONs:
sbatch --export=ALL,MODEL=qwen38,STAGES=C,CONFIGS=nothink pipeline_b/run_pipeline_b.sbatch
```

`STAGES` is `B1 B2` (default), `B1`, `B2` or `C`. With `SOURCE=latex` the summaries are built from
the LaTeX sources. Other options are listed in `common/cluster/job.sh`.

Without SLURM, with a vLLM server running:

```bash
python pipeline_b/build_json.py --output runs/pdf/qwen38_nothink_s0_B1.jsonl
python pipeline_b/answer_from_json.py --benchmark-dir benchmark/PhysicsPaperPilotEval_v0.3.2 \
    --json-input runs/pdf/qwen38_nothink_s0_B1.jsonl --output runs/pdf/qwen38_nothink_s0_B2.jsonl \
    --label "qwen38|nothink|s0|B_json_same_model"
```

## Output

In `runs/<pdf|latex>/`:

- `<model>_<config>_s<seed>_B1.jsonl` — one JSON per paper, with the input and settings that produced
  it; `..._B1.rejected.jsonl` for the invalid ones.
- `<model>_<config>_s<seed>_B2.jsonl` / `.csv` — the answers, raw and in the benchmark's format;
  `..._B2.evidence_fields.csv` — fields used and abstentions.
- `C_<reader>_<config>_from_<summarizer>_s<seed>.*` — the same for condition C.
