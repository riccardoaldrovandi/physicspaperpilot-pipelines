"""Spearman correlation between the manual review and the automatic metrics.

Human targets, from docs/benchmark/revisione_<reader>_<config>_s<seed>.xlsx (review_sheet.py):
  - judgement as an ordinal scale: corretta 2, parziale 1, sbagliata 0, astensione 0;
  - human key-point coverage (KP coperti / KP totali).
Automatic metrics: evaluate_v3 key-point coverage and token F1 (hidden 'Chiave' sheet), and BERTScore
P/R/F of roberta-large and deberta-xlarge-mnli, raw and rescaled (bertscore_analysis.py answers.csv).
Rescaling is affine with one b per encoder and metric, so its Spearman equals the raw one; both are
computed as a check.

1. Per answer: Spearman over all answers, and over the answered ones (abstentions excluded).
2. Paired: per question, Δ = A − B of the human judgement against Δ of the metric (this is how the
   metrics are used in the analysis), plus the share of questions where the human sees a difference
   and the metric moves in the same direction.
95% CIs: percentile bootstrap resampling questions (both answers of a question together), fixed seed.
Usage, from the repository root:
  python evaluation/review_correlation.py evaluation/human_review/revisione_qwen38_nothink_s0.xlsx
"""
import argparse
import os
import random
import re
import sys

from openpyxl import load_workbook
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_csv, write_csv  # noqa: E402

ORDINAL = {"corretta": 2, "parziale": 1, "sbagliata": 0, "astensione": 0}
COND = {"A": "A_direct_pdf", "B2": "B_json_same_model"}
METRICS = ["kp_auto", "token_f1"] + [f"{e}_{m}{s}" for e in ("roberta", "deberta") for m in "FRP" for s in ("", "_rs")]


def load(xlsx, answers_csv):
    m = re.search(r"revisione_(\w+?)_(think|nothink)_s(\d+)\.xlsx$", os.path.basename(xlsx))
    if not m:
        raise SystemExit(f"Unexpected file name: {xlsx}")
    reader, config, seed = m.groups()
    wb = load_workbook(xlsx, data_only=True)
    key = {r[0]: r for r in wb["Chiave"].iter_rows(min_row=2, values_only=True) if r[0] is not None}
    bert = {(r["condition"], r["question_id"]): r for r in read_csv(answers_csv)
            if r["reader"] == reader and r["config"] == config and r["seed"] == seed
            and r["condition"] in COND.values() and r["summarizer"] in ("", reader)}
    rows = []
    for r in wb["Revisione"].iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        k = key[r[0]]
        judgement = str(r[11] or "").strip().lower()
        if judgement not in ORDINAL:
            raise SystemExit(f"Row {r[0]}: missing or invalid judgement {r[11]!r}")
        covered = [s.strip() for s in str(r[12] or "").split(",") if s.strip()]
        if not all(s.isdigit() for s in covered):
            raise SystemExit(f"Row {r[0]}: KP coperti must be integers separated by commas, got {r[12]!r}")
        b = bert[(COND[k[2]], r[1])]
        if b["answer"].strip() != (r[10] or "").strip():
            raise SystemExit(f"Row {r[0]}: answer text differs from {answers_csv}")
        row = {"riga": r[0], "question_id": r[1], "question_type": r[3], "condition": k[2],
               "judgement": judgement, "human_ord": ORDINAL[judgement], "abstained": int(judgement == "astensione"),
               "human_kp": len(covered) / int(k[5]), "kp_auto": float(k[3]), "token_f1": float(k[4])}
        for mname in METRICS[2:]:
            row[mname] = float(b[mname])
        rows.append(row)
    return rows, (reader, config, seed)


def rho(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def boot_ci(clusters, stat, n_boot, seed):
    rng = random.Random(seed)
    keys = sorted(clusters)
    vals = []
    for _ in range(n_boot):
        sample = [x for _ in keys for x in clusters[rng.choice(keys)]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xlsx")
    ap.add_argument("--answers", default="runs/pdf/bertscore/answers.csv")
    ap.add_argument("--out-dir", default="runs/pdf/bertscore")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows, (reader, config, seed) = load(args.xlsx, args.answers)
    print(f"[~] {len(rows)} judged answers ({reader} {config} s{seed}), "
          f"{sum(r['abstained'] for r in rows)} abstentions, {len({r['question_id'] for r in rows})} questions")

    out = []
    for target in ("human_ord", "human_kp"):
        for scope in ("all", "answered"):
            sub = [r for r in rows if scope == "all" or not r["abstained"]]
            clusters = {}
            for r in sub:
                clusters.setdefault(r["question_id"], []).append(r)
            for mname in METRICS:
                stat = lambda s: rho([r[target] for r in s], [r[mname] for r in s])
                lo, hi = boot_ci(clusters, stat, args.n_boot, args.seed)
                out.append({"analysis": "per_answer", "target": target, "scope": scope, "metric": mname,
                            "n": len(sub), "spearman": round(stat(sub), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})

    # Paired A - B per question.
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question_id"], {})[r["condition"]] = r
    pairs = [(q, d["A"], d["B2"]) for q, d in sorted(by_q.items()) if "A" in d and "B2" in d]
    for target in ("human_ord", "human_kp"):
        for mname in METRICS:
            diffs = [(a[target] - b[target], a[mname] - b[mname]) for _, a, b in pairs]
            stat = lambda s: rho([h for h, _ in s], [m for _, m in s])
            lo, hi = boot_ci({i: [d] for i, d in enumerate(diffs)}, stat, args.n_boot, args.seed)
            seen = [(h, m) for h, m in diffs if h != 0]
            agree = sum(1 for h, m in seen if (h > 0) == (m > 0) and m != 0)
            out.append({"analysis": "paired_A_minus_B", "target": target, "scope": "all", "metric": mname,
                        "n": len(diffs), "spearman": round(stat(diffs), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                        "human_differs": len(seen), "same_direction": agree})

    # Check: rescaled and raw give the same Spearman.
    idx = {(o["analysis"], o["target"], o["scope"], o["metric"]): o["spearman"] for o in out}
    worst = max(abs(v - idx[k[:3] + (k[3][:-3],)]) for k, v in idx.items() if k[3].endswith("_rs"))
    print(f"[check] rescaled vs raw Spearman: max |diff| {worst:.2e} -> {'OK' if worst < 1e-9 else 'FAIL'}")

    if worst >= 1e-9:
        raise SystemExit("[!] Check failed: rescaled and raw Spearman differ.")

    stem = os.path.splitext(os.path.basename(args.xlsx))[0]
    write_csv(os.path.join(args.out_dir, f"human_correlation_{stem}.csv"), out)
    write_csv(os.path.join(args.out_dir, f"human_review_rows_{stem}.csv"), rows)


if __name__ == "__main__":
    main()
