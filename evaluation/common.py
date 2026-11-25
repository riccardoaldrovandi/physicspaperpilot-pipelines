"""Helpers shared by the evaluation scripts: file names of the runs, CSV I/O, model names and the
package's scorer."""
import csv
import importlib.util
import math
import os
import re
import subprocess
import sys

CONDITION = {"A": "A_direct_pdf", "B2": "B_json_same_model", "C": "C_fixed_reader_json"}
MODELS = {"qwen38": "Qwen3.8-27B", "gemma4": "Gemma 4 31B", "nemotron35": "Nemotron 3.5", "mistral35": "Mistral Medium 3.5"}
CONFIGS = ["nothink", "think"]
NAME_AB = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<config>think|nothink)_s(?P<seed>\d+)_(?P<cond>A|B2)$")
NAME_C = re.compile(r"^C_(?P<model>[a-z0-9]+)_(?P<config>think|nothink)_from_(?P<summ>[a-z0-9]+)_s(?P<seed>\d+)$")
KEY_FILE = os.path.join("02_private_answer_key", "answer_key_private.csv")


def model_order(m):
    return list(MODELS).index(m) if m in MODELS else len(MODELS)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] {path} ({len(rows)} rows)")


def num(value):
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def describe(path):
    """Condition, reader, summarizer, config and seed of an answers file, from its name:
    <model>_<config>_s<seed>_A / _B2, or C_<reader>_<config>_from_<summarizer>_s<seed>."""
    stem = os.path.basename(path)[:-4]
    m = NAME_AB.match(stem)
    if m:
        return {"condition": CONDITION[m["cond"]], "reader": m["model"], "summarizer": m["model"] if m["cond"] == "B2" else "",
                "config": m["config"], "seed": int(m["seed"])}
    m = NAME_C.match(stem)
    if m:
        return {"condition": CONDITION["C"], "reader": m["model"], "summarizer": m["summ"], "config": m["config"], "seed": int(m["seed"])}
    return None


def score(csv_path, benchmark_dir, scores_dir, rescore=False, bert_model=None):
    """The package's evaluate_v3.py on one predictions CSV, cached in scores_dir/<stem>/ (redone when
    the CSV or the answer key is newer than the cached result)."""
    out = os.path.join(scores_dir, os.path.basename(csv_path)[:-4])
    marker = os.path.join(out, "summary_by_paper.csv")
    newest = max(os.path.getmtime(csv_path), os.path.getmtime(os.path.join(benchmark_dir, KEY_FILE)))
    if rescore or not os.path.exists(marker) or os.path.getmtime(marker) < newest:
        cmd = [sys.executable, "-B", os.path.join(benchmark_dir, "scripts", "evaluate_v3.py"), "--input", csv_path, "--outdir", out]
        if bert_model:
            cmd += ["--with-bertscore", "--bert-model", bert_model]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "USE_TF": "0"})
        if proc.returncode:
            print(proc.stdout, proc.stderr, file=sys.stderr)
            raise SystemExit(f"evaluate_v3.py failed on {csv_path}")
    return out


def load_evaluator(benchmark_dir):
    """The package's evaluate_v3.py as a module (its normalization, key-point rule, macro average)."""
    spec = importlib.util.spec_from_file_location("ppp_evaluate_v3", os.path.join(benchmark_dir, "scripts", "evaluate_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = previous
    return mod
