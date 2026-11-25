"""All result tables, as Markdown (titles and tables only), optionally also as PDF.

Reads the outputs of the other evaluation scripts in --runs-dir: scores/ (score_runs.py),
bertscore/ (bertscore_analysis.py), json_checks/ (json_checks.py), json_semantic/ (json_semantic.py).
Writes to <runs-dir>/tables/:
  TABLES.md                   Pipeline A and B (official metrics, BERTScore, evidence F1 by question
                              type), B vs A (retention, paired differences), structured JSON
  TABLES_by_question_type.md  A and B separately for MCQ, short-answer and open questions
Evidence F1 by question type is evaluate_v3's own macro restricted to one question type, per seed,
then mean and SD over seeds; the values that also exist in scores/summary.csv are checked against it.
  python evaluation/make_tables.py --runs-dir runs/pdf [--pdf]
"""
import argparse
import glob
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITION, MODELS, describe, load_evaluator, model_order, read_csv  # noqa: E402

A, B = CONDITION["A"], CONDITION["B2"]
CONFIGS = ["nothink", "think"]
QTYPES = [("multiple_choice", "MCQ"), ("short_answer", "Short"), ("open_ended", "Open")]


def keys_of(rows, reader="reader"):
    ks = sorted({(r[reader], r["config"]) for r in rows}, key=lambda k: (model_order(k[0]), CONFIGS.index(k[1])))
    return ks


def name(m):
    return MODELS.get(m, m)


def ms(mean, sd):
    if mean in ("", None):
        return "–"
    return f"{float(mean):.1f} ± {float(sd):.1f}" if sd not in ("", None) else f"{float(mean):.1f}"


def fx(v, digits):
    """A number with fixed digits, '–' when the cell is empty (nothing to measure)."""
    return "–" if v in ("", None) else f"{float(v):.{digits}f}"


def ci(v, lo, hi, fmt="{:.1f}", sign=False):
    f = (lambda x: f"{float(x):+.1f}") if sign else (lambda x: fmt.format(float(x)))
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def boot_mean(by_paper, n_boot, seed):
    papers = sorted(by_paper)
    rng = random.Random(seed)
    vals = sorted(statistics.mean(by_paper[rng.choice(papers)] for _ in papers) for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


NR = "NOT RECOVERABLE FROM STRUCTURED SUMMARY"
EV = [("evidence_section_precision", "Evidence precision"), ("evidence_section_recall", "Evidence recall"),
      ("evidence_section_f1", "Evidence F1")]
BERT = [("deberta_F_rs", "BERTScore F1 rescaled"), ("deberta_R_rs", "BERTScore recall rescaled")]
SPEC = {
    "multiple_choice": ("MCQ questions", [("mcq_correct", "MCQ accuracy")] + EV),
    "short_answer": ("short-answer questions", [("exact_match", "Exact match"), ("token_f1", "Token F1"),
                                                 ("keypoint_coverage", "Key-point coverage"),
                                                 ("numeric_target_coverage", "Numeric target coverage")] + BERT + EV),
    "open_ended": ("open questions", [("token_f1", "Token F1"), ("rougeL_f1", "ROUGE-L F1"),
                                      ("keypoint_coverage", "Key-point coverage")] + BERT + EV),
}
# Column of scores/summary.csv holding the same value (mean over seeds), for the check.
CHECK = {("multiple_choice", "mcq_correct"): "mcq_accuracy", ("short_answer", "exact_match"): "short_exact_match",
         ("short_answer", "token_f1"): "short_token_f1", ("short_answer", "keypoint_coverage"): "short_keypoint_coverage",
         ("short_answer", "numeric_target_coverage"): "numeric_target_coverage",
         ("short_answer", "evidence_section_f1"): "short_evidence_section_f1", ("open_ended", "token_f1"): "open_token_f1",
         ("open_ended", "rougeL_f1"): "open_rougeL_f1", ("open_ended", "keypoint_coverage"): "open_keypoint_coverage",
         ("open_ended", "evidence_section_f1"): "open_evidence_section_f1"}


def ms_seeds(vals):
    if not vals:
        return "–"
    return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"


def overall_tables(args, ev):
    bd = args.runs_dir
    md = [f"# {pkg_name(args)} — tables", ""]

    summary = read_csv(os.path.join(bd, "scores", "summary.csv"))
    sidx = {(r["condition"], r["reader"], r["config"]): r for r in summary if r["condition"] in (A, B)}
    bs = read_csv(os.path.join(bd, "bertscore", "summary.csv"))
    cond_letter = {A: "A", B: "B"}
    bidx = {(r["condition"], r["reader"], r["config"], r["question_type"]): r for r in bs}
    rev = {v: k for k, v in MODELS.items()}

    # Evidence F1 per question type, per seed (evaluate_v3 macro), then mean / SD over seeds.
    ev_by = defaultdict(lambda: defaultdict(list))
    for d in glob.glob(os.path.join(bd, "scores", "*", "scored_answers.csv")):
        meta = describe(os.path.join(bd, os.path.basename(os.path.dirname(d)) + ".csv"))
        if not meta or meta["condition"] not in (A, B):
            continue
        rows = read_csv(d)
        key = (meta["condition"], meta["reader"], meta["config"])
        for qt, _ in QTYPES:
            v = ev.macro_over_papers(rows, "evidence_section_f1", qt)
            if v is not None:
                ev_by[key][qt].append(100 * v)
        ev_by[key]["all"].append(100 * ev.macro_over_papers(rows, "evidence_section_f1"))
    for key, d in ev_by.items():
        for qt, col in (("short_answer", "short_evidence_section_f1_percent"), ("open_ended", "open_evidence_section_f1_percent"),
                        ("all", "all_evidence_section_f1_percent")):
            if d[qt] and abs(statistics.mean(d[qt]) - float(sidx[key][col])) > 0.01:
                raise SystemExit(f"[!] Evidence F1 {qt} of {key} differs from scores/summary.csv")
    print("[check] evidence F1 short / open / all = scores/summary.csv: OK")

    for cond in (A, B):
        P = "Pipeline A (PDF)" if cond == A else "Pipeline B (own JSON)"
        keys = [k for k in keys_of(summary) if (cond, *k) in sidx]
        hdr = ["Model", "Config", "MCQ accuracy", "Short exact match", "Short token F1", "Short key-point coverage",
               "Open token F1", "Open ROUGE-L F1", "Open key-point coverage", "Numeric target coverage", "Answered questions"]
        cols = ["mcq_accuracy", "short_exact_match", "short_token_f1", "short_keypoint_coverage", "open_token_f1", "open_rougeL_f1",
                "open_keypoint_coverage", "numeric_target_coverage"]
        rows = []
        for k in keys:
            r = sidx[(cond, *k)]
            row = [name(k[0]), k[1]] + [ms(r.get(c + "_percent"), r.get(c + "_sd")) for c in cols] + [ms(r["answered_questions"], r.get("answered_questions_sd"))]
            if cond == B:
                row += [ms(r.get("abstained_percent"), r.get("abstained_sd")), ms(r.get("not_recoverable_percent"), r.get("not_recoverable_sd"))]
            rows.append(row)
        if cond == B:
            hdr += ["Abstained %", "Not recoverable %"]
        md.append(table(f"{P} — official metrics (evaluate_v3, %, mean ± SD over 3 seeds)", hdr, rows))
        for qt, label in (("short", "short answers"), ("open", "open answers")):
            rows = []
            for k in keys:
                r = bidx.get((cond_letter[cond], name(k[0]), k[1], qt))
                if r is None:   # no answer of this type in the runs
                    continue
                for t in ("roberta", "deberta"):
                    rows.append([name(k[0]), k[1], f"{t}-{'large' if t == 'roberta' else 'xlarge-mnli'}", f"{float(r['abstention_percent']):.1f}"]
                                + [f"{float(r[f'{t}_{m}_answered']):.1f}" for m in "PRF"]
                                + [f"{float(r[f'{t}_{m}_rs_answered']):.1f}" for m in "PRF"]
                                + [f"{float(r[f'{t}_F_rs_all']):.1f}"])
            md.append(table(f"{P} — BERTScore, {label} (answered questions; raw and baseline-rescaled)",
                            ["Model", "Config", "Encoder", "Abstained %", "P raw", "R raw", "F1 raw", "P rescaled", "R rescaled",
                             "F1 rescaled", "F1 rescaled incl. abstentions"], rows))
        rows = []
        for k in keys:
            d = ev_by[(cond, *k)]
            rows.append([name(k[0]), k[1]] + [ms_seeds(d[qt]) for qt, _ in QTYPES] + [ms_seeds(d["all"])])
        md.append(table(f"{P} — evidence-section F1 by question type (%, mean ± SD over 3 seeds)",
                        ["Model", "Config", "MCQ questions", "Short-answer questions", "Open questions", "All questions"], rows))

    # ---- Pipeline B vs A
    ret = read_csv(os.path.join(bd, "scores", "retention_summary.csv"))
    rmetrics = ["mcq_accuracy", "short_exact_match", "short_token_f1", "short_keypoint_coverage", "open_token_f1", "open_rougeL_f1",
                "open_keypoint_coverage", "numeric_target_coverage", "short_evidence_section_f1", "open_evidence_section_f1",
                "all_evidence_section_f1"]
    ridx = {(r["reader"], r["config"], r["metric"]): r for r in ret if r["condition"] == B}
    rows = []
    for k in keys_of([r for r in ret if r["condition"] == B]):
        rows.append([name(k[0]), k[1]] + [(lambda v: "–" if v in ("", None) else f"{float(v):.1f}")(ridx.get((*k, m), {}).get("retention_of_means_percent"))
                                          for m in rmetrics])
    md.append(table("Pipeline B vs A — retention of the official metrics (B mean / A mean × 100, %)",
                    ["Model", "Config", "MCQ accuracy", "Short exact match", "Short token F1", "Short key-point coverage", "Open token F1",
                     "Open ROUGE-L F1", "Open key-point coverage", "Numeric target coverage", "Short evidence F1", "Open evidence F1",
                     "All evidence F1"], rows))
    rr = read_csv(os.path.join(bd, "bertscore", "retention_rescaled.csv"))
    pd = read_csv(os.path.join(bd, "bertscore", "paired_deltas.csv"))
    rri = {(r["reader"], r["config"], r["question_type"], r["encoder"], r["metric"]): r for r in rr if r["condition"] == "B"}
    pdi = {(r["reader"], r["config"], r["question_type"], r["encoder"], r["metric"]): r for r in pd
           if r["condition"] == "B" and r["scale"] == "rescaled" and r["scope"] == "answered"}
    bkeys = [(rev[r], c) for r, c in sorted({(r["reader"], r["config"]) for r in rr if r["condition"] == "B"},
                                            key=lambda x: (model_order(rev[x[0]]), CONFIGS.index(x[1])))]
    for qt, label in (("short", "short answers"), ("open", "open answers")):
        rows = []
        for m, c in bkeys:
            row = [name(m), c]
            for t in ("roberta", "deberta"):
                for met in ("F", "R"):
                    x = rri.get((name(m), c, qt, t, met))
                    row.append("–" if x is None or x["retention_percent"] == "" else
                               ci(x["retention_percent"], x["ci_low"], x["ci_high"]) + (" †" if x["unstable"] == "1" else ""))
            rows.append(row)
        md.append(table(f"Pipeline B vs A — BERTScore retention, {label} (baseline-rescaled, answered questions, %, 95% CI)",
                        ["Model", "Config", "roberta F1", "roberta recall", "deberta F1", "deberta recall"], rows))
    for qt, label in (("short", "short answers"), ("open", "open answers")):
        rows = []
        for m, c in bkeys:
            if (name(m), c, qt, "roberta", "F") not in pdi:
                continue
            row = [name(m), c, pdi[(name(m), c, qt, "roberta", "F")]["n_questions"]]
            for t in ("roberta", "deberta"):
                for met in ("F", "R", "P"):
                    x = pdi[(name(m), c, qt, t, met)]
                    row.append(ci(x["delta"], x["ci_low"], x["ci_high"], sign=True))
            rows.append(row)
        md.append(table(f"Pipeline B vs A — paired differences B − A, {label} (BERTScore rescaled, answered questions, points, 95% CI)",
                        ["Model", "Config", "Questions", "roberta ΔF1", "roberta Δrecall", "roberta Δprecision", "deberta ΔF1",
                         "deberta Δrecall", "deberta Δprecision"], rows))

    # ---- JSON (B1): Level 1 and Level 2 (evaluation/json_checks.py)
    js = read_csv(os.path.join(bd, "json_checks", "summary.csv"))
    md.append(table("Structured JSON (B1) — Level 1 and Level 2",
                    ["Model", "Config", "JSONs", "Level 1 passed", "Empty fields %", "4-gram repetition", "Compression ratio",
                     "Level 2 [95% CI]", "Numbers grounded", "References grounded", "Datasets grounded",
                     "Ungrounded numbers per JSON", "Ungrounded references per JSON"],
                    [[name(r["model"]), r["config"], r["jsons"], f"{r['level1_passed']}/{r['jsons']}", r["empty_fields_percent"],
                      fx(r["repeated_ngram_ratio"], 3), fx(r["compression_ratio"], 3),
                      (ci(r["level2"], r["level2_ci_low"], r["level2_ci_high"], fmt="{:.3f}") if r["level2"] != "" else "–"),
                      fx(r["numbers_grounded"], 3), fx(r["references_grounded"], 3), fx(r["datasets_grounded"], 3),
                      fx(r["ungrounded_numbers_per_json"], 2), fx(r["ungrounded_references_per_json"], 2)] for r in js]))

    # ---- JSON semantic
    jd = os.path.join(bd, "json_semantic")
    cov = read_csv(os.path.join(jd, "json_coverage.csv"))
    corder = lambda r: (model_order(r["model"]), CONFIGS.index(r["config"]))
    cov.sort(key=corder)
    md.append(table("Structured JSON (B1) — gold key points present in the JSON (%, 95% CI)",
                    ["Model", "Config", "Lexical (per passage)", "Lexical (whole JSON)", "BERTScore roberta", "BERTScore deberta",
                     "AUC own vs other paper, roberta", "AUC own vs other paper, deberta", "Best-passage recall rescaled, roberta",
                     "Best-passage recall rescaled, deberta", "Pipeline A answers, lexical", "Pipeline B answers, lexical"],
                    [[name(r["model"]), r["config"], r["json_lex"], r["json_lex_whole"], r["json_roberta_spec"], r["json_deberta_spec"],
                      r["auc_own_vs_other_roberta"], r["auc_own_vs_other_deberta"], r["json_roberta_R_rs_mean"], r["json_deberta_R_rs_mean"],
                      r["answer_A_lex"], r["answer_B2_lex"]] for r in cov]))
    md.append(table("Structured JSON (B1) — control: key points of another paper found in the JSON (%)",
                    ["Model", "Config", "Lexical", "roberta, answer threshold", "deberta, answer threshold",
                     "roberta, JSON threshold", "deberta, JSON threshold"],
                    [[name(r["model"]), r["config"], r["control_other_paper_lex"], r["control_other_paper_roberta_present"],
                      r["control_other_paper_deberta_present"], r["control_other_paper_roberta_spec"], r["control_other_paper_deberta_spec"]]
                     for r in cov]))
    dec = [r for r in read_csv(os.path.join(jd, "decomposition.csv")) if r["method"] == "lex" and r["condition"] == "B"]
    dec.sort(key=lambda r: (model_order(r["reader"]), CONFIGS.index(r["config"])))
    md.append(table("Structured JSON (B1) → Pipeline B answer — where the gold key points are lost (lexical, % of key points)",
                    ["Model", "Config", "Recovered", "Lost in reading (B2)", "Lost in summary (B1)", "In answer, not in JSON"],
                    [[name(r["reader"]), r["config"], r["recovered"], r["lost_in_reading"], r["lost_in_summary"], r["not_from_json"]] for r in dec]))
    kpj = read_csv(os.path.join(jd, "kp_json.csv"))
    field_of = lambda f: re.split(r"[.\[]", f)[0]
    tot = Counter(field_of(r["deberta_field"]) for r in kpj if r["deberta_spec"] == "1")
    top = [f for f, _ in tot.most_common(7)]
    rows = []
    for k in sorted({(r["model"], r["config"]) for r in kpj}, key=lambda k: (model_order(k[0]), CONFIGS.index(k[1]))):
        sub = [field_of(r["deberta_field"]) for r in kpj if (r["model"], r["config"]) == k and r["deberta_spec"] == "1"]
        c = Counter(sub)
        pct = lambda n: f"{100 * n / len(sub):.1f}" if sub else "–"   # no key point found in this model's JSONs
        rows.append([name(k[0]), k[1], len(sub)] + [pct(c[f]) for f in top] + [pct(sum(v for f, v in c.items() if f not in top))])
    md.append(table("Structured JSON (B1) — fields holding the present key points (BERTScore deberta, % of present key points)",
                    ["Model", "Config", "Present (key point × seed)"] + top + ["other fields"], rows))
    st = read_csv(os.path.join(jd, "stability_density.csv"))
    st.sort(key=corder)
    md.append(table("Structured JSON (B1) — stability over seeds and density",
                    ["Model", "Config", "Same presence in 3 seeds, lexical %", "Same presence in 3 seeds, deberta %",
                     "Numbers shared between seeds (Jaccard)", "JSON words (mean)", "Key points present per 1,000 words (deberta)"],
                    [[name(r["model"]), r["config"], r["lex_same_in_3_seeds_pct"], r["deberta_spec_same_in_3_seeds_pct"],
                      r["numbers_jaccard_between_seeds"], r["json_words_mean"], r["kp_present_per_1k_words_deberta"]] for r in st]))

    return md


def question_type_tables(args, ev):
    bd = args.runs_dir
    qtype = {r["question_id"]: r["question_type"] for r in read_csv(os.path.join(args.benchmark_dir, "01_questions_for_models", "questions_public.csv"))}

    vals = defaultdict(list)       # (cond, reader, config, qtype, metric) -> per-seed values
    for d in sorted(glob.glob(os.path.join(bd, "scores", "*", "scored_answers.csv"))):
        run = os.path.basename(os.path.dirname(d))
        meta = describe(os.path.join(bd, run + ".csv"))
        if not meta or meta["condition"] not in (A, B):
            continue
        rows = read_csv(d)
        for qt, (_, cols) in SPEC.items():
            for m, _ in cols:
                if m.startswith("deberta"):
                    continue
                v = ev.macro_over_papers(rows, m, qt)
                if v is not None:
                    vals[(meta["condition"], meta["reader"], meta["config"], qt, m)].append(100 * v)
        if meta["condition"] == B:
            extra = read_csv(os.path.join(bd, run + ".evidence_fields.csv"))
            answers = {r["question_id"]: r["model_answer"] for r in read_csv(os.path.join(bd, run + ".csv"))}
            for qt in SPEC:
                sub = [x for x in extra if qtype[x["question_id"]] == qt]
                if not sub:
                    continue
                vals[(B, meta["reader"], meta["config"], qt, "abstained")].append(100 * sum(int(x["abstained"]) for x in sub) / len(sub))
                vals[(B, meta["reader"], meta["config"], qt, "not_recoverable")].append(
                    100 * sum(answers.get(x["question_id"], "").strip().upper() == NR for x in sub) / len(sub))

    # BERTScore deberta rescaled, answered questions, macro per seed.
    rev = {"A_direct_pdf": A, "B_json_same_model": B}
    by_run = defaultdict(list)
    for r in read_csv(os.path.join(bd, "bertscore", "answers.csv")):
        if r["condition"] in rev and r["abstained"] == "0":
            by_run[(rev[r["condition"]], r["reader"], r["config"], r["seed"], r["question_type"])].append(r)
    for (cond, reader, config, seed, qt), rows in by_run.items():
        for m, _ in BERT:
            vals[(cond, reader, config, qt, m)].append(100 * ev.macro_over_papers(
                [{"arxiv_id": x["arxiv_id"], "question_type": qt, m: x[m]} for x in rows], m, qt))

    summary = {(r["condition"], r["reader"], r["config"]): r for r in read_csv(os.path.join(bd, "scores", "summary.csv"))}
    for (cond, reader, config, qt, m), v in vals.items():
        col = CHECK.get((qt, m))
        if col and abs(statistics.mean(v) - float(summary[(cond, reader, config)][col + "_percent"])) > 0.01:
            raise SystemExit(f"[!] {cond} {reader} {config} {qt} {m}: {statistics.mean(v):.2f} != summary {col}")
    print("[check] values shared with scores/summary.csv: OK")

    keys = sorted({(k[1], k[2]) for k in vals}, key=lambda k: (model_order(k[0]), CONFIGS.index(k[1])))
    md = [f"# {pkg_name(args)} — Pipeline A and B by question type", ""]
    for qt, (label, cols) in SPEC.items():
        for cond, pname in ((A, "Pipeline A (PDF)"), (B, "Pipeline B (own JSON)")):
            extra = []
            if cond == B:
                extra = [("abstained", "Abstained %")] + ([("not_recoverable", "Not recoverable %")] if qt == "short_answer" else [])
            allcols = cols + extra
            md.append(f"### {pname} — {label} (%, mean ± SD over 3 seeds)")
            md.append("")
            md.append("| Model | Config | " + " | ".join(c for _, c in allcols) + " |")
            md.append("|" + "---|" * (len(allcols) + 2))
            for reader, config in keys:
                if not vals.get((cond, reader, config, qt, cols[0][0])):
                    continue
                md.append(f"| {MODELS.get(reader, reader)} | {config} | " + " | ".join(ms_seeds(vals.get((cond, reader, config, qt, m), []))
                                                                              for m, _ in allcols) + " |")
            md.append("")
    return md


def pkg_name(args):
    return os.path.basename(os.path.normpath(args.benchmark_dir)).replace("PhysicsPaperPilotEval_", "PhysicsPaperPilot ")


def to_pdf(md_path, landscape=True):
    """Markdown to PDF with markdown-it-py + WeasyPrint (optional dependencies)."""
    from markdown_it import MarkdownIt
    from weasyprint import HTML
    html = MarkdownIt("commonmark").enable("table").render(open(md_path, encoding="utf-8").read())
    css = ("@page { size: A4 landscape; margin: 12mm 10mm; } body { font-family: 'DejaVu Sans', sans-serif; font-size: 7.4pt; }"
           " h1 { font-size: 15pt; } h3 { font-size: 9.5pt; margin: 12pt 0 4pt; } table { border-collapse: collapse; width: 100%;"
           " page-break-inside: avoid; } th, td { border: 1px solid #c8c8c8; padding: 2pt 3pt; text-align: left; } th { background: #eef1f5; }")
    HTML(string=f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>").write_pdf(md_path[:-3] + ".pdf")
    print(f"[+] {md_path[:-3]}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs/pdf")
    ap.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--pdf", action="store_true", help="Also write PDFs (needs markdown-it-py and weasyprint)")
    args = ap.parse_args()
    ev = load_evaluator(args.benchmark_dir)
    out_dir = os.path.join(args.runs_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)
    for name, md in (("TABLES.md", overall_tables(args, ev)), ("TABLES_by_question_type.md", question_type_tables(args, ev))):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"[+] {path}: {sum(1 for x in md if x.startswith('###'))} tables")
        if args.pdf:
            to_pdf(path)


if __name__ == "__main__":
    main()
