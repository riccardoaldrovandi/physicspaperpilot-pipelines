"""Reading the PhysicsPaperPilot package and writing answers in its format, for both pipelines.

From the package (--benchmark-dir), unchanged: the questions (questions_public.csv), the prompts
(the ```text block of 03_pilot_run/*.md), the CSV format (predictions_template.csv) and the MCQ
letter parser of scripts/evaluate_v3.py. The private answer key is never read here.

Answers are appended to a raw JSONL file, one line per question (a rerun skips the questions
already answered and retries the failed ones); the predictions CSV is rebuilt from it.
"""
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys

QUESTIONS_FILE = os.path.join("01_questions_for_models", "questions_public.csv")
TEMPLATE_FILE = os.path.join("03_pilot_run", "predictions_template.csv")
EVALUATOR_FILE = os.path.join("scripts", "evaluate_v3.py")
NOT_RECOVERABLE = "NOT RECOVERABLE FROM STRUCTURED SUMMARY"


def read_prompt(benchmark_dir: str, relative: str) -> str:
    """The ```text block of a prompt file of the package, verbatim."""
    with open(os.path.join(benchmark_dir, relative), encoding="utf-8") as f:
        m = re.search(r"```text\n(.*?)\n```", f.read(), re.S)
    if not m:
        raise ValueError(f"No ```text block in {relative}")
    return m.group(1)


def load_questions(benchmark_dir: str) -> list:
    with open(os.path.join(benchmark_dir, QUESTIONS_FILE), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_of(data) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()[:16]


def done_questions(raw_path: str) -> set:
    """Questions already answered without error (failed calls are retried on a rerun)."""
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_record(raw_path: str, record: dict) -> None:
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def base_record(condition: str, label: str, q: dict, model_name: str, settings: dict) -> dict:
    return {"condition": condition, "label": label, "model": model_name, "question_id": q["question_id"],
            "arxiv_id": q["arxiv_id"], "question_type": q["question_type"], "generation_settings": settings}


def _evaluator(benchmark_dir: str):
    """The package's evaluate_v3.py as a module (for its MCQ parser), without writing __pycache__."""
    spec = importlib.util.spec_from_file_location("ppp_evaluate_v3", os.path.join(benchmark_dir, EVALUATOR_FILE))
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def write_predictions(benchmark_dir: str, raw_path: str, csv_path: str) -> int:
    """The predictions CSV (package format) from the raw JSONL, last answer per question; for
    answers from a JSON also <csv>.evidence_fields.csv (evidence_fields, abstained, not_recoverable)."""
    with open(os.path.join(benchmark_dir, TEMPLATE_FILE), encoding="utf-8-sig", newline="") as f:
        fields = next(csv.reader(f))
    parse_mcq = _evaluator(benchmark_dir).parse_mcq
    latest = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            latest[rec["question_id"]] = rec
    rows, extra = [], []
    for qid, rec in latest.items():
        answer = rec.get("answer", "")
        choice = rec.get("mcq_choice", "")
        if rec["question_type"] == "multiple_choice" and not choice:
            choice = parse_mcq(answer, "")
        row = {"model": rec["label"], "question_id": qid, "model_answer": answer,
               "mcq_choice": choice if rec["question_type"] == "multiple_choice" else "",
               "cited_evidence": rec.get("cited_evidence", "")}
        rows.append({k: row.get(k, "") for k in fields})
        if rec["condition"] == "json":
            extra.append({"model": rec["label"], "question_id": qid,
                          "evidence_fields": "; ".join(rec.get("evidence_fields", [])),
                          "abstained": int(bool(rec.get("abstained", False))),
                          "not_recoverable": int(answer.strip().upper() == NOT_RECOVERABLE),
                          "error": rec.get("error", "")})
    rows.sort(key=lambda r: r["question_id"])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if extra:
        extra.sort(key=lambda r: r["question_id"])
        with open(csv_path[:-4] + ".evidence_fields.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(extra[0]))
            w.writeheader()
            w.writerows(extra)
    return len(rows)
