# PhysicsPaperPilot pipelines

Two ways for a language model to answer questions about physics papers, and the tools to compare
them on the PhysicsPaperPilot benchmark (15 papers on fast calorimeter simulation, 45 questions with
a private answer key - TBE).

- **Pipeline A** — the model reads the paper (text and figures) and answers each question.
- **Pipeline B** — the model first writes a structured JSON summary of the paper, then answers each
  question from the JSON alone.

The question behind the comparison: how much of what is needed to answer survives the structured
summary, and at what cost.

## Layout

```
benchmark/     the PhysicsPaperPilot package, v0.3.2: questions, prompts, private answer key, scorer
papers/pdf/    the 15 papers, in the versions listed in the package's papers.csv
common/        what both pipelines share
  llm_client.py        client for the model served by vLLM (sampling, context fitting, call log)
  paper_inputs.py      the text and figure images of a paper, from its PDF or its LaTeX source
  benchmark_io.py      questions, prompts and answer files in the benchmark's format
  fetch_latex_sources.py   downloads the arXiv LaTeX sources and converts their figures
  cluster/             Leonardo (CINECA): vLLM runtimes, the four models, shared job logic
  energy/              energy accounting of the jobs (GPU monitor + CodeCarbon)
pipeline_a/    answering from the paper                          -> pipeline_a/README.md
pipeline_b/    structured summary, then answering from it        -> pipeline_b/README.md
evaluation/    scoring, BERTScore, JSON checks, energy, tables   -> evaluation/README.md
```

Outputs go to `runs/<pdf|latex>/`, `energy_logs/` and `logs/` (not tracked by git).

## Models

| `MODEL` | Model | vLLM | GPUs | Configs |
|---|---|---|---|---|
| `qwen38` | Qwen3.8-27B | 0.19.1 | 2 (TP) | think, nothink |
| `gemma4` | Gemma 4 31B | 0.19.1 | 4 (TP) | think, nothink |
| `nemotron35` | Nemotron 3.5 Super VL 120B (FP8) | 0.32 | 4 (TP + EP) | think, nothink |
| `mistral35` | Mistral Medium 3.5 128B (FP8) | 0.32 | 4 (TP) | nothink; think with `MISTRAL35_EAGLE=1` (to be validated) |
| `llama4scout` | Llama 4 Scout 109B (FP8, RedHatAI) | 0.32 | 4 (TP) | nothink |
| `deepseekv4fv` | DeepSeek V4-Flash-Vision-Exp 305B | 0.32 | 4 (TP) | think, nothink (to be validated) |
| `mimo26flash` | MiMo-V2.6-Flash-RL 309B (FP8) | 0.32 | 4 (TP) | think, nothink (to be validated) |

All run with a 128k-token context, up to 32k output tokens and the sampling recommended by each
model card (`common/cluster/models.sh`). Each config runs with seeds 0, 1 and 2.

The A100 nodes have NVIDIA driver 535 (CUDA 12): vLLM 0.32 (`-cu129` image) is the newest release
that runs there, since 0.33 and later ship CUDA 13 only. A new model is added to `models.sh` and
checked with the smoke test before any benchmark job:

```bash
sbatch --export=ALL,MODEL=llama4scout,CONFIGS=nothink common/cluster/smoke_test.sbatch
sbatch --export=ALL,MODEL=mistral35,MISTRAL35_EAGLE=1,CONFIGS="think nothink" common/cluster/smoke_test.sbatch
```

It checks text, image reading, guided JSON, reasoning and decoding speed on long answers, then runs
one paper through Pipeline A and step B1.

## Running on Leonardo

Once, on a login node, from the repository root:

```bash
bash common/cluster/setup_runtime.sh vllm019             # Qwen3.8, Gemma 4
sbatch -p lrd_all_serial --time=02:00:00 --mem=30G -c 8 -o logs/setup_%j.out \
    --wrap "bash common/cluster/setup_runtime.sh vllm032"   # Nemotron 3.5, Mistral Medium 3.5
MODEL=qwen38 bash common/cluster/download_model.sh       # once per model (+ MISTRAL35_EAGLE=1 for mistral35 think)
bash common/energy/setup_venv_energy.sh                  # energy monitor (optional)
pip install -r requirements-pipelines.txt                # in the Python env of the jobs ($WORK/venv_vllm)
python common/fetch_latex_sources.py                     # only for SOURCE=latex
```

Then, with `export SBATCH_ACCOUNT=<your account>`:

```bash
sbatch --export=ALL,MODEL=qwen38 pipeline_a/run_pipeline_a.sbatch
sbatch --export=ALL,MODEL=qwen38 pipeline_b/run_pipeline_b.sbatch
```

Each job serves one model and runs every config and seed; resubmitting resumes where it stopped.
`SOURCE=latex` uses the LaTeX sources instead of the PDFs. Copy `runs/` and `energy_logs/` back and
follow `evaluation/README.md`.

## Benchmark version

`benchmark/` holds v0.3.2 of the package. Compared with the original v0.3, v0.3.1 corrected
evidence labels, two key points, one question and the scorer (`CHANGELOG_v0.3.1.md`); v0.3.2 widens
the accepted evidence of the 15 multiple-choice questions to the Introduction and the section that
presents the method (`CHANGELOG_v0.3.2.md`). Questions and gold answers are unchanged.

The answer key in `benchmark/.../02_private_answer_key/` is for scoring only: the pipelines never
read it, and it must not be given to a model.
