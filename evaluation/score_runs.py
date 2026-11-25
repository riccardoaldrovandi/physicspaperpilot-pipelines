"""Scores the answers of both pipelines with the package's evaluate_v3.py and compares the conditions.

Each predictions CSV of --runs-dir is scored on its own by the package's scripts/evaluate_v3.py,
unchanged (never mix conditions in one run). Then, from its outputs:
  summary.csv            per condition / reader / summarizer / config: every metric, mean and SD over
                         seeds (answered_questions out of 45; abstained / not_recoverable for JSON answers)
  by_category.csv        per question category
  retention_by_paper.csv Score(JSON read by R) / Score(PDF read by R) x 100 per metric and paper, same
                         reader, config and seed; empty when the PDF score is 0; json_missing=1 = no JSON
  retention_summary.csv  per condition / reader / config and metric: ratio of the mean scores and mean of
                         the per-paper ratios, on the papers with a JSON (plus the version with missing = 0)
Outputs in <runs-dir>/scores/. --with-bertscore adds evaluate_v3's BERTScore (scores_bertscore/).
  python evaluation/score_runs.py --runs-dir runs/pdf
"""
import argparse
import glob
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITION, describe, num, read_csv, score, write_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs/pdf")
    ap.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2")
    ap.add_argument("--rescore", action="store_true", help="Rerun evaluate_v3.py even if its outputs are up to date")
    ap.add_argument("--with-bertscore", action="store_true", help="Add evaluate_v3's BERTScore (outputs in scores_bertscore/)")
    ap.add_argument("--bert-model", default="roberta-large")
    args = ap.parse_args()
    scores_dir = os.path.join(args.runs_dir, "scores_bertscore" if args.with_bertscore else "scores")
    os.makedirs(scores_dir, exist_ok=True)

    runs = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*.csv"))):
        if path.endswith(".evidence_fields.csv"):
            continue
        meta = describe(path)
        if not meta:
            print(f"[~] Not a benchmark answer file, skipped: {path}")
            continue
        meta["dir"] = score(path, args.benchmark_dir, scores_dir, args.rescore, args.bert_model if args.with_bertscore else None)
        extra = path[:-4] + ".evidence_fields.csv"
        meta["extra"] = read_csv(extra) if os.path.exists(extra) else []
        runs.append(meta)
    if not runs:
        raise SystemExit(f"No answer CSV in {args.runs_dir}")
    print(f"[~] {len(runs)} answer file(s) scored with {os.path.join(args.benchmark_dir, 'scripts', 'evaluate_v3.py')}")

    group_key = ("condition", "reader", "summarizer", "config")
    by_paper = []        # one row per run and paper
    by_model = defaultdict(list)
    for r in runs:
        for row in read_csv(os.path.join(r["dir"], "summary_by_paper.csv")):
            by_paper.append({**{k: r[k] for k in group_key}, "seed": r["seed"], **row})
        model_row = read_csv(os.path.join(r["dir"], "summary_by_model.csv"))[0]
        # evaluate_v3 averages over the answered questions only: a paper without JSON just drops out.
        model_row["answered_questions"] = len(read_csv(os.path.join(r["dir"], "scored_answers.csv")))
        if r["extra"]:
            n = len(r["extra"])
            model_row["abstained_percent"] = 100 * sum(int(x["abstained"]) for x in r["extra"]) / n
            model_row["not_recoverable_percent"] = 100 * sum(int(x["not_recoverable"]) for x in r["extra"]) / n
            model_row["answer_errors"] = sum(1 for x in r["extra"] if x["error"])
        by_model[tuple(r[k] for k in group_key)].append(model_row)
    metrics = [k for k in by_paper[0] if k.endswith("_percent")]

    # summary.csv: mean and sd over seeds of evaluate_v3's macro (per paper, then across papers) scores.
    summary = []
    for key, rows in sorted(by_model.items()):
        item = dict(zip(group_key, key), n_seeds=len(rows))
        for m in metrics + ["answered_questions", "abstained_percent", "not_recoverable_percent"]:
            vals = [v for v in (num(x.get(m)) for x in rows) if v is not None]
            if vals:
                item[m] = round(statistics.mean(vals), 2)
                item[m.replace("_percent", "") + "_sd"] = round(statistics.stdev(vals), 2) if len(vals) > 1 else ""
        summary.append(item)
    write_csv(os.path.join(scores_dir, "summary.csv"), summary)

    # by_category.csv: answer-level means per question category, pooled over seeds.
    cats = defaultdict(list)
    for r in runs:
        for row in read_csv(os.path.join(r["dir"], "scored_answers.csv")):
            cats[tuple(r[k] for k in group_key) + (row["question_type"], row["category"])].append(row)
    by_cat = []
    for key, rows in sorted(cats.items()):
        item = dict(zip(group_key + ("question_type", "category"), key), n_answers=len(rows))
        for field in ["mcq_correct", "keypoint_coverage", "token_f1", "rougeL_f1", "numeric_target_coverage", "evidence_section_f1",
                      "bertscore_f1"]:
            vals = [v for v in (num(x.get(field)) for x in rows) if v is not None]
            if vals:
                item[field + "_percent"] = round(100 * statistics.mean(vals), 2)
        by_cat.append(item)
    write_csv(os.path.join(scores_dir, "by_category.csv"), by_cat)

    # Retention: JSON condition over the same reader's PDF score, same config, seed and paper.
    # A paper with no JSON (rejected in B1) has no answers, so evaluate_v3 leaves it out of
    # summary_by_paper: it is added here as json_missing=1 with a JSON score of 0.
    pdf_scores = {(r["reader"], r["config"], r["seed"], r["arxiv_id"]): r for r in by_paper if r["condition"] == CONDITION["A"]}
    json_runs = defaultdict(dict)
    for r in by_paper:
        if r["condition"] != CONDITION["A"]:
            json_runs[tuple(r[k] for k in group_key) + (r["seed"],)][r["arxiv_id"]] = r
    retention = []
    for run_key, papers in sorted(json_runs.items()):
        run = dict(zip(group_key + ("seed",), run_key))
        for (reader, config, seed, aid), base in sorted(pdf_scores.items()):
            if (reader, config, seed) != (run["reader"], run["config"], run["seed"]):
                continue
            r = papers.get(aid)
            for m in metrics:
                a = num(base.get(m))
                b = num(r.get(m)) if r else 0.0
                if a is None or b is None:
                    continue
                retention.append({**run, "arxiv_id": aid, "metric": m[:-8], "json_missing": int(r is None),
                                  "pdf_score": a, "json_score": b, "retention_percent": round(100 * b / a, 2) if a > 0 else ""})
    write_csv(os.path.join(scores_dir, "retention_by_paper.csv"), retention)

    # Two views, never combined: the papers with a JSON only (information lost inside the JSON), and
    # every paper with the missing JSONs at 0 (the pipeline as a whole, B1 rejections included).
    agg = defaultdict(list)
    for x in retention:
        agg[tuple(x[k] for k in group_key) + (x["metric"],)].append(x)
    ret_summary = []
    for key, all_rows in sorted(agg.items()):
        rows = [x for x in all_rows if not x["json_missing"]]
        ratios = [x["retention_percent"] for x in rows if x["retention_percent"] != ""]
        mean_pdf = statistics.mean(x["pdf_score"] for x in rows) if rows else 0
        mean_json = statistics.mean(x["json_score"] for x in rows) if rows else 0
        all_pdf = statistics.mean(x["pdf_score"] for x in all_rows)
        all_json = statistics.mean(x["json_score"] for x in all_rows)
        ret_summary.append({**dict(zip(group_key + ("metric",), key)),
                            "pdf_score_mean": round(mean_pdf, 2), "json_score_mean": round(mean_json, 2),
                            "retention_of_means_percent": round(100 * mean_json / mean_pdf, 2) if mean_pdf > 0 else "",
                            "mean_paper_retention_percent": round(statistics.mean(ratios), 2) if ratios else "",
                            "n_paper_seed_pairs": len(rows), "n_with_nonzero_pdf_score": len(ratios),
                            "n_json_missing": len(all_rows) - len(rows),
                            "json_missing_percent": round(100 * (len(all_rows) - len(rows)) / len(all_rows), 2),
                            "retention_missing_as_zero_percent": round(100 * all_json / all_pdf, 2) if all_pdf > 0 else ""})
    write_csv(os.path.join(scores_dir, "retention_summary.csv"), ret_summary)


if __name__ == "__main__":
    main()
