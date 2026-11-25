"""BERTScore of the answers, with the corrections needed to read it (short and open answers).

1. Per-answer P / R / F1 from the package's evaluate_v3.py (--with-bertscore), once per encoder:
   roberta-large (the package default) and microsoft/deberta-xlarge-mnli.
2. Baseline rescaling (x - b) / (1 - b), b from bert_score's own rescale_baseline file for the
   encoder's default layer: 0 = as similar as two random sentences. Affine, applied to stored values.
3. Abstentions ("NOT RECOVERABLE FROM STRUCTURED SUMMARY" or "Not specified in the paper" alone) are
   counted apart: abstention rate, and BERTScore of the answered questions only.
4. Paired differences JSON - PDF (same reader and config, per question, mean over seeds, macro over
   papers) with a 95% bootstrap CI resampling papers; retention = ratio of rescaled means.
5. Agreement of the two encoders (Spearman per answer).
Outputs in <runs-dir>/bertscore/: answers.csv, summary.csv, paired_deltas.csv, retention_rescaled.csv,
abstention.csv, encoder_agreement.csv, verification.txt. Needs bert-score and a GPU for speed:
  USE_TF=0 HF_HUB_OFFLINE=1 python evaluation/bertscore_analysis.py --runs-dir runs/pdf
"""
import argparse
import glob
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITION, MODELS, describe, model_order, num, read_csv, score, write_csv  # noqa: E402

ENCODERS = {"roberta-large": "roberta", "microsoft/deberta-xlarge-mnli": "deberta"}
MARKER_B = "NOT RECOVERABLE FROM STRUCTURED SUMMARY"
MARKER_A = "not specified in the paper"
QTYPES = {"short_answer": "short", "open_ended": "open"}


def is_abstention(text):
    """The whole answer is the abstention marker (an answer that adds content is an answer)."""
    t = re.sub(r"[\s.:;,\-'\"]+", " ", text).strip()
    return t.upper() == MARKER_B or t.lower() == MARKER_A


def baseline(encoder):
    """b for P, R, F of the encoder's default layer, from bert_score's rescale_baseline/en/<encoder>.tsv."""
    import bert_score
    from bert_score.utils import model2layers
    layer = model2layers[encoder]
    path = os.path.join(os.path.dirname(bert_score.__file__), "rescale_baseline", "en", encoder + ".tsv")
    for row in read_csv(path):
        if int(row["LAYER"]) == layer:
            return {"P": float(row["P"]), "R": float(row["R"]), "F": float(row["F"])}, layer, path
    raise SystemExit(f"No baseline for layer {layer} in {path}")


def rescale(x, b):
    return (x - b) / (1 - b)


def load_answers(bench_dir, benchmark_dir, out_dir, encoders):
    """One row per (run, question) with P/R/F raw and rescaled per encoder; short and open answers only."""
    runs = []
    for path in sorted(glob.glob(os.path.join(bench_dir, "*.csv"))):
        if path.endswith(".evidence_fields.csv"):
            continue
        meta = describe(path)
        if meta:
            runs.append((path, meta))
    base = {e: baseline(e) for e in encoders}
    rows = {}
    for enc in encoders:
        scores_dir = os.path.join(out_dir, "scores_" + ENCODERS.get(enc, enc.replace("/", "_")))
        os.makedirs(scores_dir, exist_ok=True)
        print(f"[~] {enc}: {len(runs)} answer files through evaluate_v3 -> {scores_dir}")
        for path, meta in runs:
            d = score(path, benchmark_dir, scores_dir, False, enc)
            for r in read_csv(os.path.join(d, "scored_answers.csv")):
                if r["question_type"] not in QTYPES:
                    continue
                k = (meta["condition"], meta["reader"], meta["summarizer"], meta["config"], meta["seed"], r["question_id"])
                row = rows.setdefault(k, {
                    "condition": meta["condition"], "reader": meta["reader"], "summarizer": meta["summarizer"],
                    "config": meta["config"], "seed": meta["seed"], "question_id": r["question_id"],
                    "arxiv_id": r["arxiv_id"], "question_type": r["question_type"],
                    "abstained": int(is_abstention(r["model_answer"])), "answer": r["model_answer"]})
                if row["answer"] != r["model_answer"]:
                    raise SystemExit(f"Answer text differs between encoders for {k}")
                tag = ENCODERS.get(enc, enc)
                b = base[enc][0]
                for m, col in (("P", "bertscore_precision"), ("R", "bertscore_recall"), ("F", "bertscore_f1")):
                    v = num(r[col])
                    if v is None:
                        raise SystemExit(f"Missing {col} for {k} ({enc})")
                    row[f"{tag}_{m}"] = v
                    row[f"{tag}_{m}_rs"] = rescale(v, b[m])
    return list(rows.values()), base


# ---------------------------------------------------------------- statistics
def paper_macro(values_by_q, q_paper):
    """Mean over papers of the mean over that paper's questions."""
    by_p = defaultdict(list)
    for q, v in values_by_q.items():
        by_p[q_paper[q]].append(v)
    return statistics.mean(statistics.mean(v) for v in by_p.values()), by_p


def bootstrap(by_paper_pairs, stat, n_boot, seed):
    """Percentile 95% CI of stat(list of per-paper items), resampling papers with replacement."""
    papers = sorted(by_paper_pairs)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [by_paper_pairs[rng.choice(papers)] for _ in papers]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None, None, 0
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))], len(vals)


def per_question(rows, value_key, answered_only):
    """question_id -> mean over seeds of value_key (answered seeds only if answered_only)."""
    acc = defaultdict(list)
    for r in rows:
        if answered_only and r["abstained"]:
            continue
        acc[r["question_id"]].append(r[value_key])
    return {q: statistics.mean(v) for q, v in acc.items()}


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["condition"], r["reader"], r["summarizer"], r["config"], r["question_type"])].append(r)
    return g


def comparisons(g):
    """(json group key, pdf group key): B vs A of the same reader/config; C vs A of the C reader/config."""
    out = []
    for key in g:
        cond, reader, summ, config, qt = key
        if cond == CONDITION["A"]:
            continue
        a = (CONDITION["A"], reader, "", config, qt)
        if a in g:
            out.append((key, a))
    return sorted(out, key=lambda k: (k[0][0] != CONDITION["B2"], model_order(k[0][1]), k[0][3],
                                      model_order(k[0][2]), k[0][4] != "short_answer"))


def label(key):
    cond, reader, summ, config, qt = key
    name = MODELS.get(reader, reader)
    if cond == CONDITION["C"]:
        name += f" reads {MODELS.get(summ, summ)}"
    return name, config, {"A_direct_pdf": "A", "B_json_same_model": "B", "C_fixed_reader_json": "C"}[cond], QTYPES[qt]


def analyse(rows, base, encoders, n_boot, seed):
    q_paper = {r["question_id"]: r["arxiv_id"] for r in rows}
    g = groups(rows)
    tags = [ENCODERS.get(e, e) for e in encoders]

    # Per-condition summary: abstention rate, means over all answers and over answered ones.
    summary = []
    for key in sorted(g, key=lambda k: (model_order(k[1]), k[3], k[0], model_order(k[2]), k[4] != "short_answer")):
        rs = g[key]
        name, config, cond, qt = label(key)
        item = {"reader": name, "config": config, "condition": cond, "question_type": qt, "n_answers": len(rs),
                "n_questions": len({r["question_id"] for r in rs}), "n_papers": len({r["arxiv_id"] for r in rs}),
                "n_seeds": len({r["seed"] for r in rs}), "abstained": sum(r["abstained"] for r in rs)}
        ab = per_question(rs, "abstained", False)
        item["abstention_percent"] = round(100 * paper_macro(ab, q_paper)[0], 2)
        for t in tags:
            for m in ("P", "R", "F"):
                for suffix in ("", "_rs"):
                    vk = f"{t}_{m}{suffix}"
                    item[vk + "_all"] = round(100 * paper_macro(per_question(rs, vk, False), q_paper)[0], 2)
                    pq = per_question(rs, vk, True)
                    item[vk + "_answered"] = round(100 * paper_macro(pq, q_paper)[0], 2) if pq else ""
        summary.append(item)

    # Paired differences and retention, JSON condition vs PDF condition.
    paired, retention, abst = [], [], []
    for jkey, akey in comparisons(g):
        name, config, cond, qt = label(jkey)
        common = {"reader": name, "config": config, "condition": cond, "question_type": qt}
        # Abstention rate difference (all questions).
        ja, aa = per_question(g[jkey], "abstained", False), per_question(g[akey], "abstained", False)
        qs = sorted(set(ja) & set(aa))
        bp = defaultdict(list)
        for q in qs:
            bp[q_paper[q]].append(ja[q] - aa[q])
        stat = lambda s: statistics.mean(statistics.mean(x) for x in s)
        lo, hi, _ = bootstrap(bp, stat, n_boot, seed)
        abst.append({**common, "n_questions": len(qs), "n_papers": len(bp),
                     "json_abstention_percent": round(100 * paper_macro({q: ja[q] for q in qs}, q_paper)[0], 2),
                     "pdf_abstention_percent": round(100 * paper_macro({q: aa[q] for q in qs}, q_paper)[0], 2),
                     "delta_pp": round(100 * stat(list(bp.values())), 2),
                     "ci_low_pp": round(100 * lo, 2), "ci_high_pp": round(100 * hi, 2)})
        for t in tags:
            for m in ("P", "R", "F"):
                for scope in ("answered", "all"):
                    for suffix in ("", "_rs"):
                        vk = f"{t}_{m}{suffix}"
                        jq = per_question(g[jkey], vk, scope == "answered")
                        aq = per_question(g[akey], vk, scope == "answered")
                        qs = sorted(set(jq) & set(aq))
                        bp = defaultdict(list)
                        for q in qs:
                            bp[q_paper[q]].append((jq[q], aq[q]))
                        dstat = lambda s: statistics.mean(statistics.mean(j - a for j, a in x) for x in s)
                        lo, hi, _ = bootstrap(bp, dstat, n_boot, seed)
                        jm = statistics.mean(statistics.mean(j for j, _ in x) for x in bp.values())
                        am = statistics.mean(statistics.mean(a for _, a in x) for x in bp.values())
                        paired.append({**common, "encoder": t, "metric": m, "scale": "rescaled" if suffix else "raw",
                                       "scope": scope, "n_questions": len(qs), "n_papers": len(bp),
                                       "json_mean": round(100 * jm, 2), "pdf_mean": round(100 * am, 2),
                                       "delta": round(100 * dstat(list(bp.values())), 2),
                                       "ci_low": round(100 * lo, 2), "ci_high": round(100 * hi, 2)})
                        if suffix and scope == "answered":
                            def rstat(s):
                                a = statistics.mean(statistics.mean(a for _, a in x) for x in s)
                                j = statistics.mean(statistics.mean(j for j, _ in x) for x in s)
                                return j / a if a > 0 else None
                            alo, ahi, _ = bootstrap(bp, lambda s: statistics.mean(statistics.mean(a for _, a in x) for x in s),
                                                    n_boot, seed)
                            rlo, rhi, nv = bootstrap(bp, rstat, n_boot, seed)
                            retention.append({**common, "encoder": t, "metric": m, "n_questions": len(qs), "n_papers": len(bp),
                                              "pdf_mean_rs": round(100 * am, 2), "json_mean_rs": round(100 * jm, 2),
                                              "retention_percent": round(100 * jm / am, 1) if am > 0 else "",
                                              "ci_low": round(100 * rlo, 1) if rlo is not None else "",
                                              "ci_high": round(100 * rhi, 1) if rhi is not None else "",
                                              "valid_resamples": nv, "pdf_mean_ci_low": round(100 * alo, 2),
                                              "unstable": int(alo <= 0 or nv < n_boot)})
    return summary, paired, retention, abst


def rnd(v, digits=3):
    return "" if v is None else round(v, digits)


def spearman(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None   # undefined: too few answers, or constant scores
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


# ---------------------------------------------------------------- checks
def verify(rows, base, bench_dir, encoders, paired, gold, n_sample, seed):
    """Cross-checks written to verification.txt; any failure stops the script."""
    out, ok = [], True
    tags = [ENCODERS.get(e, e) for e in encoders]
    # (a) roberta values equal those of the official scores_bertscore/ run.
    if "roberta" in tags and os.path.isdir(os.path.join(bench_dir, "scores_bertscore")):
        off = {}
        for d in glob.glob(os.path.join(bench_dir, "scores_bertscore", "*", "scored_answers.csv")):
            meta = describe(os.path.join(bench_dir, os.path.basename(os.path.dirname(d)) + ".csv"))
            for r in read_csv(d):
                if r["question_type"] in QTYPES:
                    off[(meta["condition"], meta["reader"], meta["summarizer"], meta["config"], meta["seed"], r["question_id"])] = float(r["bertscore_f1"])
        diffs = [abs(off[k] - r["roberta_F"]) for r in rows
                 for k in [(r["condition"], r["reader"], r["summarizer"], r["config"], r["seed"], r["question_id"])] if k in off]
        good = len(diffs) == len(rows) and max(diffs) < 1e-6
        ok &= good
        out.append(f"(a) roberta F1 vs scores_bertscore/: {len(diffs)}/{len(rows)} answers matched, max |diff| {max(diffs):.2e} -> {'OK' if good else 'FAIL'}")
    # (b) raw F1 'all' paper-macro means equal the official summary.csv (mean over seeds of per-seed macro).
    official = os.path.join(bench_dir, "scores_bertscore", "summary.csv")
    if "roberta" in tags and os.path.exists(official):
        off = {(r["condition"], r["reader"], r["summarizer"], r["config"]): r for r in read_csv(official)}
        q_paper = {r["question_id"]: r["arxiv_id"] for r in rows}
        worst = 0
        for key, rs in groups(rows).items():
            cond, reader, summ, config, qt = key
            per_seed = [paper_macro({r["question_id"]: r["roberta_F"] for r in rs if r["seed"] == s}, q_paper)[0]
                        for s in sorted({r["seed"] for r in rs})]
            worst = max(worst, abs(100 * statistics.mean(per_seed) - float(off[(cond, reader, summ, config)][f"{QTYPES[qt]}_bertscore_f1_percent"])))
        good = worst < 0.01
        ok &= good
        out.append(f"(b) per-seed macro F1 means vs scores_bertscore/summary.csv: max |diff| {worst:.4f} points -> {'OK' if good else 'FAIL'}")
    else:
        out.append("(a)/(b) skipped: no scores_bertscore/ (python evaluation/score_runs.py --with-bertscore)")
    # (c) rescaled delta = raw delta / (1 - b), answered scope.
    worst = 0
    idx = {(p["reader"], p["config"], p["condition"], p["question_type"], p["encoder"], p["metric"], p["scope"], p["scale"]): p for p in paired}
    for (rd, cf, cd, qt, enc, m, sc, scale), p in idx.items():
        if scale != "raw":
            continue
        b = base[[e for e in encoders if ENCODERS.get(e, e) == enc][0]][0][m]
        q = idx[(rd, cf, cd, qt, enc, m, sc, "rescaled")]
        worst = max(worst, abs(p["delta"] / (1 - b) - q["delta"]))
    good = worst < 0.05   # both sides rounded to 0.01 before the check
    ok &= good
    out.append(f"(c) rescaled delta = raw delta / (1 - b): max |diff| {worst:.4f} points (rounding) -> {'OK' if good else 'FAIL'}")
    # (d) direct bert_score recomputation of a random sample, per encoder.
    from bert_score import score as bscore
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_sample, len(rows)))
    for enc in encoders:
        t = ENCODERS.get(enc, enc)
        _, _, F = bscore([r["answer"] for r in sample], [gold[r["question_id"]] for r in sample], lang="en",
                         model_type=enc, verbose=False, batch_size=8)
        d = max(abs(float(F[i]) - r[f"{t}_F"]) for i, r in enumerate(sample))
        good = d < 1e-4
        ok &= good
        out.append(f"(d) {enc}: direct bert_score on {len(sample)} random answers, max |diff| {d:.2e} -> {'OK' if good else 'FAIL'}")
    # (e) abstention rule vs the abstained flag of the B/C runs.
    flags = {}
    for f in glob.glob(os.path.join(bench_dir, "*.evidence_fields.csv")):
        meta = describe(f.replace(".evidence_fields", ""))
        for r in read_csv(f):
            flags[(meta["condition"], meta["reader"], meta["summarizer"], meta["config"], meta["seed"], r["question_id"])] = int(r["abstained"])
    mism = [r for r in rows if (k := (r["condition"], r["reader"], r["summarizer"], r["config"], r["seed"], r["question_id"])) in flags
            and flags[k] != r["abstained"]]
    out.append(f"(e) abstention rule vs 'abstained' flag (B/C, short+open): {len(mism)} disagreement(s): "
               + "; ".join(f"{r['reader']} {r['config']} s{r['seed']} {r['question_id']} flag={1 - r['abstained']} text={r['answer'][:80]!r}" for r in mism))
    return ok, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs/pdf")
    ap.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2")
    ap.add_argument("--out-dir", default=None, help="Default: <runs-dir>/bertscore")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-sample", type=int, default=100)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.runs_dir, "bertscore")
    os.makedirs(out_dir, exist_ok=True)

    rows, base = load_answers(args.runs_dir, args.benchmark_dir, out_dir, args.encoders)
    tags = [ENCODERS.get(e, e) for e in args.encoders]
    write_csv(os.path.join(out_dir, "answers.csv"), [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    summary, paired, retention, abst = analyse(rows, base, args.encoders, args.n_boot, args.seed)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)
    write_csv(os.path.join(out_dir, "paired_deltas.csv"), paired)
    write_csv(os.path.join(out_dir, "retention_rescaled.csv"), retention)
    write_csv(os.path.join(out_dir, "abstention.csv"), abst)

    agree = []
    if len(tags) == 2:
        for qt in QTYPES:
            sub = [r for r in rows if r["question_type"] == qt]
            for m in ("P", "R", "F"):
                agree.append({"question_type": QTYPES[qt], "metric": m, "n_answers": len(sub),
                              "spearman_all": rnd(spearman([r[f"{tags[0]}_{m}"] for r in sub], [r[f"{tags[1]}_{m}"] for r in sub])),
                              "spearman_answered": rnd(spearman([r[f"{tags[0]}_{m}"] for r in sub if not r["abstained"]],
                                                                [r[f"{tags[1]}_{m}"] for r in sub if not r["abstained"]]))})
        write_csv(os.path.join(out_dir, "encoder_agreement.csv"), agree)

    key = read_csv(os.path.join(args.benchmark_dir, "02_private_answer_key", "answer_key_private.csv"))
    gold = {r["question_id"]: r["gold_answer"] for r in key}
    ok, lines = verify(rows, base, args.runs_dir, args.encoders, paired, gold, args.verify_sample, args.seed)
    with open(os.path.join(out_dir, "verification.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        raise SystemExit("[!] Verification failed: see verification.txt")


if __name__ == "__main__":
    main()
