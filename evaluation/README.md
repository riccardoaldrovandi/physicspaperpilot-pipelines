# Evaluation

Scores the answers of both pipelines and the JSON summaries of Pipeline B, and builds the result
tables. Everything runs on any machine (a GPU makes BERTScore much faster); copy `runs/` and
`energy_logs/` back from the cluster first.

## Order

From the repository root, for the PDF runs (`--runs-dir runs/latex` for the LaTeX ones):

```bash
pip install -r requirements-evaluation.txt
python evaluation/score_runs.py                         # official metrics of the benchmark
USE_TF=0 python evaluation/bertscore_analysis.py         # BERTScore, rescaled, paired differences
python evaluation/json_checks.py                         # Level 1 / Level 2 of the JSONs
USE_TF=0 python evaluation/json_semantic.py              # gold key points inside the JSONs
python evaluation/make_tables.py --pdf                  # all tables, Markdown and PDF
```

Each script reads `runs/pdf/` and writes into a subfolder of it. `make_tables.py` needs the
outputs of the four scripts before it.

## Scripts

| Script | Output (in `runs/pdf/`) | What it measures |
|---|---|---|
| `score_runs.py` | `scores/` | Runs the benchmark's own scorer (`evaluate_v3.py`) on every answer file: MCQ accuracy, token F1, ROUGE-L, key-point coverage, numeric targets, evidence-section F1. Means and SD over seeds; retention of Pipeline B with respect to A |
| `bertscore_analysis.py` | `bertscore/` | BERTScore precision, recall and F1 with two encoders (roberta-large, deberta-xlarge-mnli), raw and rescaled with the published baseline; abstentions counted apart; paired differences B − A and retention with bootstrap confidence intervals |
| `json_checks.py` | `json_checks/` | Level 1: is the JSON well formed (schema, empty fields, repetition, copied schema text). Level 2: are the numbers, references and datasets it cites present in the paper |
| `json_semantic.py` | `json_semantic/` | Which gold key points are present in the JSON (lexical rule of the benchmark, and BERTScore against each passage), with a control on another paper's key points; where information is lost (summary or reading); stability over seeds; density |
| `make_tables.py` | `tables/` | `TABLES.md` and `TABLES_by_question_type.md` (titles and tables only), `--pdf` for PDFs |
| `review_sheet.py` | `evaluation/human_review/` | `make`: a blind review sheet (Excel) of the A and B answers of one model; `score`: reads the filled sheet back |
| `review_correlation.py` | `bertscore/` | Spearman correlation between the human review and every automatic metric, per answer and paired A − B |
| `energy_report.py` | any folder | Energy, time and CO2 of the jobs, from `energy_logs/<job id>/` (see below) |
| `common.py` | | Shared helpers: file names of the runs, CSV I/O, model names |

`human_review/revisione_qwen38_nothink_s0.xlsx` is the filled review of Qwen3.8 (nothink, seed 0):
60 answers, A and B mixed blind. It is used by `review_correlation.py` and to calibrate the BERTScore
threshold of `json_semantic.py`.

## Energy report

Every cluster job writes `energy_logs/<job id>/`: one line per model call (time, tokens, images,
stage, paper, question), GPU energy and power every second from NVML, and CodeCarbon's CPU, GPU and
RAM energy with the CO2 estimate.

```bash
python evaluation/energy_report.py energy_logs/<job A> energy_logs/<job B> --out-dir energy_logs/report \
    --runs-dir runs/pdf [--pue 1.1] [--carbon-intensity 250]
```

It gives energy per job (start-up, busy, idle; GPU-hours allocated and used), per stage and per
call (Wh, Wh per 1,000 tokens, latency, throughput), per question and per paper, and the number of
questions per paper above which Pipeline B uses less energy than A. When several calls run at the
same time, the energy of each second is shared among them; the shares add up exactly to the
measured total.

## Notes

- **BERTScore on the raw scale** puts every pair of English sentences around 0.8–0.9: compare
  rescaled values, and prefer paired differences to absolute scores. Recall follows the human
  judgement better than F1 and precision.
- **The private answer key** (`benchmark/.../02_private_answer_key/`) is read here only, never by the
  pipelines.
- **Evidence F1 of Pipeline B** is near zero by construction: its answers cite JSON fields, not the
  paper's section headings.
