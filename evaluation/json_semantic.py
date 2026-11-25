"""Does the structured JSON (B1) contain the gold key points of the benchmark?

For every B1 JSON of --runs-dir and every gold key point of its paper (keypoints_lexicon.csv of the
private key, short and open questions):
  1. the JSON is split into passages with their path (string fields into sentences, list items,
     dict sub-fields, key_results / tables / figures items, figure curves on their own);
  2. lexical presence: the package's key-point rule (an accepted phrase, or at least 85% of the anchor
     tokens) applied to each passage ("lex"), and for reference to the whole JSON at once ("lex_whole",
     much more lenient on a long text);
  3. BERTScore recall of the key point (reference) against each passage (candidate), best passage, per
     encoder, raw and rescaled. The maximum over ~70 passages is far higher than the score of one
     answer: "spec" presence = maximum above the 95th percentile of the maxima reached by ANOTHER
     paper's key points in the same JSON (95% specificity); "present" = above the threshold that best
     matches the manual key-point labels of the answers (review sheet), shown for the control only;
  4. control: the key points of another paper searched in each JSON (should be ~0);
  5. per key point, JSON vs Pipeline B answer with the same lexical rule: recovered / lost in reading /
     lost in summary / in the answer but not in the JSON;
  6. where the present key points are (field), stability over seeds and density of the JSON.
Means: per question, per paper, then over papers (evaluate_v3's macro); 95% bootstrap CI over papers.
Outputs in <runs-dir>/json_semantic/; bertscore_cache_<encoder>.jsonl makes reruns fast.
  USE_TF=0 HF_HUB_OFFLINE=1 python evaluation/json_semantic.py --runs-dir runs/pdf
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bertscore_analysis import ENCODERS, baseline  # noqa: E402
from common import describe, read_csv, write_csv  # noqa: E402

NAME_B1 = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<config>think|nothink)_s(?P<seed>\d+)_B1$")
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
QTYPES = ("short_answer", "open_ended")
SKIP_FIELDS = {"visual_analysis_findings"}
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$\\])")


def evaluator(benchmark_dir):
    spec = importlib.util.spec_from_file_location("ppp_eval_v3", os.path.join(benchmark_dir, "scripts", "evaluate_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- passages
def _text(v):
    if isinstance(v, dict):
        return "; ".join(t for t in (_text(x) for x in v.values()) if t)
    if isinstance(v, list):
        return "; ".join(t for t in (_text(x) for x in v) if t)
    return "" if v is None else " ".join(str(v).split())


def _sentences(text):
    out = []
    for s in SENT_RE.split(text):
        s = s.strip()
        if not s:
            continue
        if out and len(s.split()) < 4:
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def passages(parsed):
    """[(path, text)] of a v2.1 JSON; empty values and the PySR field are left out."""
    out = []

    def add(path, text):
        text = " ".join(str(text).split())
        if text and re.search(r"[A-Za-z0-9]", text) and text.lower() not in ("not specified", "n/a", "none"):
            out.append((path, text))

    for k, v in parsed.items():
        if k in SKIP_FIELDS or v in (None, "", [], {}):
            continue
        if isinstance(v, str):
            for s in _sentences(v):
                add(k, s)
        elif isinstance(v, dict):
            for sub, sv in v.items():
                if isinstance(sv, list):
                    for i, item in enumerate(sv):
                        add(f"{k}.{sub}[{i}]", _text(item))
                else:
                    for s in _sentences(_text(sv)):
                        add(f"{k}.{sub}", s)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    curves = item.get("curves") if isinstance(item.get("curves"), list) else []
                    add(f"{k}[{i}]", _text({kk: vv for kk, vv in item.items() if kk != "curves"}))
                    for j, c in enumerate(curves):
                        add(f"{k}[{i}].curves[{j}]", _text(c))
                else:
                    add(f"{k}[{i}]", _text(item))
    return out


# ---------------------------------------------------------------- lexical rule (evaluate_v3)
def lex_hit(ev, text_norm, kp):
    if any(ev.phrase_present(text_norm, a) for a in kp["aliases"]):
        return True
    anchors = list(dict.fromkeys(t for t in ev.tokenize(kp["key_point"]) if t not in ev.STOPWORDS and len(t) > 1))
    if len(anchors) >= 3:
        words = set(text_norm.split())
        return sum(1 for t in anchors if t in words) / len(anchors) >= 0.85
    return False


# ---------------------------------------------------------------- BERTScore
class Scorer:
    CHUNK = 4000

    def __init__(self, encoder, batch_size, cache_path=None):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(model_type=encoder, lang="en", batch_size=batch_size)
        # BERTScorer.score() has its own batch_size (default 64) that ignores the constructor's one.
        self.batch_size = batch_size
        self.cache, self.cache_path = {}, cache_path
        # Scores already computed by an earlier run (same encoder, same texts) are reused.
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    c, r, v = json.loads(line)
                    self.cache[(c, r)] = v
            print(f"    {len(self.cache)} pairs from {cache_path}", flush=True)

    def recall(self, pairs):
        """pairs: [(candidate, reference)] -> recall of the reference covered by the candidate."""
        todo = list(dict.fromkeys(p for p in pairs if p not in self.cache))
        # bert_score keeps the embeddings of every sentence of one call in memory: blocks of CHUNK pairs.
        for i in range(0, len(todo), self.CHUNK):
            block = todo[i:i + self.CHUNK]
            _, R, _ = self.scorer.score([c for c, _ in block], [r for _, r in block], batch_size=self.batch_size)
            for p, v in zip(block, R.tolist()):
                self.cache[p] = v
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    for p in block:
                        f.write(json.dumps([p[0], p[1], self.cache[p]], ensure_ascii=False) + "\n")
            if len(todo) > self.CHUNK:
                print(f"    {min(i + self.CHUNK, len(todo))}/{len(todo)} pairs", flush=True)
        return [self.cache[p] for p in pairs]


# ---------------------------------------------------------------- statistics
def macro(values_by_q, q_paper):
    by_p = defaultdict(list)
    for q, v in values_by_q.items():
        by_p[q_paper[q]].append(v)
    return statistics.mean(statistics.mean(v) for v in by_p.values()), by_p


def boot(by_p, n_boot, seed):
    papers = sorted(by_p)
    rng = random.Random(seed)
    vals = sorted(statistics.mean(statistics.mean(by_p[rng.choice(papers)]) for _ in papers) for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def best_tau(scores, labels):
    cands = sorted(set(scores))
    return max(((kappa([s >= t for s in scores], labels), t) for t in cands), key=lambda x: (x[0], -x[1]))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs/pdf")
    ap.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2")
    ap.add_argument("--review-xlsx", default="evaluation/human_review/revisione_qwen38_nothink_s0.xlsx")
    ap.add_argument("--out-dir", default=None, help="Default: <runs-dir>/json_semantic")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-seed0", action="store_true", help="Quick test: seed 0 JSONs and answers only")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.runs_dir, "json_semantic")
    os.makedirs(out_dir, exist_ok=True)
    ev = evaluator(args.benchmark_dir)
    tags = [ENCODERS.get(e, e) for e in args.encoders]

    questions = {r["question_id"]: r for r in read_csv(os.path.join(args.benchmark_dir, "01_questions_for_models", "questions_public.csv"))}
    kps = defaultdict(list)
    for r in read_csv(os.path.join(args.benchmark_dir, "02_private_answer_key", "keypoints_lexicon.csv")):
        q = questions[r["question_id"]]
        if q["question_type"] in QTYPES:
            kps[q["arxiv_id"]].append({"qid": r["question_id"], "kp_id": r["keypoint_id"], "key_point": r["key_point"],
                                       "aliases": json.loads(r["accepted_phrases_json"] or "[]")})
    q_paper = {kp["qid"]: a for a, lst in kps.items() for kp in lst}
    print(f"[~] {sum(len(v) for v in kps.values())} key points, {len(q_paper)} questions, {len(kps)} papers")

    # ---- JSONs
    jsons, n_bad = [], 0
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*_B1.jsonl"))):
        m = NAME_B1.match(os.path.basename(path)[:-6])
        if not m:
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            aid = (ARXIV_RE.search(rec.get("source_file", "")) or [None, ""])[1]
            try:
                parsed = json.loads([x["content"] for x in rec["messages"] if x["role"] == "assistant"][-1])
            except (json.JSONDecodeError, IndexError, KeyError):
                n_bad += 1
                continue
            if args.only_seed0 and m["seed"] != "0":
                continue
            jsons.append({"model": m["model"], "config": m["config"], "seed": int(m["seed"]), "arxiv_id": aid,
                          "passages": passages(parsed), "text": _text(parsed)})
    print(f"[~] {len(jsons)} JSONs ({n_bad} unreadable), {statistics.mean(len(j['passages']) for j in jsons):.0f} passages on average")

    # ---- answers (A, B2, C)
    answers = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*.csv"))):
        if path.endswith(".evidence_fields.csv"):
            continue
        meta = describe(path)
        if not meta or (args.only_seed0 and meta["seed"] != 0):
            continue
        for r in read_csv(path):
            if r["question_id"] in q_paper:
                answers.append({**meta, "qid": r["question_id"], "answer": r["model_answer"] or ""})

    # ---- lexical presence; control: the key points of ANOTHER paper searched in each JSON
    papers = sorted(kps)
    other = {p: papers[(i + 1) % len(papers)] for i, p in enumerate(papers)}
    rows_json, rows_ans, rows_ctrl = [], [], []
    for j in jsons:
        norm_whole = ev.normalize_text(j["text"])
        norm_pass = [(p, ev.normalize_text(t)) for p, t in j["passages"]]
        for kp in kps[j["arxiv_id"]]:
            where = next((p for p, t in norm_pass if lex_hit(ev, t, kp)), "")
            rows_json.append({"model": j["model"], "config": j["config"], "seed": j["seed"], "arxiv_id": j["arxiv_id"],
                              "qid": kp["qid"], "kp_id": kp["kp_id"], "lex": int(bool(where)), "lex_field": where,
                              "lex_whole": int(lex_hit(ev, norm_whole, kp)), "_j": j, "_kp": kp})
        for kp in kps[other[j["arxiv_id"]]]:
            rows_ctrl.append({"model": j["model"], "config": j["config"], "seed": j["seed"], "arxiv_id": j["arxiv_id"],
                              "qid": kp["qid"], "kp_id": kp["kp_id"],
                              "lex": int(any(lex_hit(ev, t, kp) for _, t in norm_pass)), "_j": j, "_kp": kp})
    kp_by_q = defaultdict(list)
    for lst in kps.values():
        for kp in lst:
            kp_by_q[kp["qid"]].append(kp)
    for a in answers:
        norm = ev.normalize_text(a["answer"])
        for kp in kp_by_q[a["qid"]]:
            rows_ans.append({"condition": a["condition"], "reader": a["reader"], "summarizer": a["summarizer"],
                             "config": a["config"], "seed": a["seed"], "arxiv_id": q_paper[a["qid"]], "qid": a["qid"],
                             "kp_id": kp["kp_id"], "lex": int(lex_hit(ev, norm, kp)), "_a": a, "_kp": kp})

    # ---- BERTScore recall
    base = {e: baseline(e)[0]["R"] for e in args.encoders}
    for enc, tag in zip(args.encoders, tags):
        print(f"[~] {enc}: JSON passages and answers vs key points...")
        # deberta-xlarge needs smaller batches on an 8 GB GPU (long figure passages).
        bs = args.batch_size if tag != "deberta" else min(args.batch_size, 8)
        sc = Scorer(enc, bs, os.path.join(out_dir, f"bertscore_cache_{tag}.jsonl"))
        pairs = [(t, r["_kp"]["key_point"]) for r in rows_json + rows_ctrl for _, t in r["_j"]["passages"]]
        pairs += [(r["_a"]["answer"] or ".", r["_kp"]["key_point"]) for r in rows_ans]
        print(f"    {len(set(pairs))} distinct pairs")
        sc.recall(pairs)
        for r in rows_json + rows_ctrl:
            vals = sc.recall([(t, r["_kp"]["key_point"]) for _, t in r["_j"]["passages"]]) or [0.0]
            i = max(range(len(vals)), key=vals.__getitem__)
            r[f"{tag}_R"] = vals[i]
            r[f"{tag}_R_rs"] = (vals[i] - base[enc]) / (1 - base[enc])
            r[f"{tag}_field"] = r["_j"]["passages"][i][0] if r["_j"]["passages"] else ""
        for r in rows_ans:
            v = sc.recall([(r["_a"]["answer"] or ".", r["_kp"]["key_point"])])[0]
            r[f"{tag}_R"], r[f"{tag}_R_rs"] = v, (v - base[enc]) / (1 - base[enc])
        del sc
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    # ---- calibration on the manual labels
    from openpyxl import load_workbook
    wb = load_workbook(args.review_xlsx, data_only=True)
    key = {r[0]: r for r in wb["Chiave"].iter_rows(min_row=2, values_only=True) if r[0] is not None}
    m = re.search(r"revisione_(\w+?)_(think|nothink)_s(\d+)\.xlsx$", args.review_xlsx)
    rev_model, rev_cfg, rev_seed = m.group(1), m.group(2), int(m.group(3))
    cond_name = {"A": "A_direct_pdf", "B2": "B_json_same_model"}
    idx = {(r["condition"], r["qid"], r["kp_id"]): r for r in rows_ans
           if r["reader"] == rev_model and r["config"] == rev_cfg and r["seed"] == rev_seed and r["condition"] in cond_name.values()}
    labels = []
    for r in wb["Revisione"].iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        cond = cond_name[key[r[0]][2]]
        covered = {int(x) for x in str(r[12] or "").split(",") if x.strip().isdigit()}
        for kp in kp_by_q[r[1]]:
            n = int(kp["kp_id"].rsplit("KP", 1)[1])
            a = idx.get((cond, r[1], kp["kp_id"]))
            if a is None:   # that answer is not in --runs-dir
                continue
            labels.append({"qid": r[1], "condition": cond, "kp_id": kp["kp_id"], "human": int(n in covered), "lex": a["lex"],
                           **{f"{t}_R": a[f"{t}_R"] for t in tags}})
    if not labels:
        raise SystemExit(f"[!] None of the answers labelled in {args.review_xlsx} ({rev_model} {rev_cfg} s{rev_seed}, A and B2) "
                         f"is in {args.runs_dir}: the BERTScore thresholds cannot be calibrated.")
    print(f"[~] {len(labels)} manual key-point labels for the calibration")
    human = [x["human"] for x in labels]
    calib, tau = [], {}
    calib.append({"method": "lexical (evaluate_v3 rule)", "n": len(labels), "human_present": sum(human),
                  "accuracy": round(sum(x["lex"] == x["human"] for x in labels) / len(labels), 3),
                  "kappa": round(kappa([x["lex"] for x in labels], human), 3), "kappa_cv": "", "auc": "", "tau": ""})
    for t in tags:
        s = [x[f"{t}_R"] for x in labels]
        tau[t] = best_tau(s, human)
        cv_pred = [None] * len(labels)
        for q in sorted({x["qid"] for x in labels}):
            train = [i for i, x in enumerate(labels) if x["qid"] != q]
            tq = best_tau([s[i] for i in train], [human[i] for i in train])
            for i, x in enumerate(labels):
                if x["qid"] == q:
                    cv_pred[i] = int(s[i] >= tq)
        calib.append({"method": f"BERTScore recall {t} >= tau", "n": len(labels), "human_present": sum(human),
                      "accuracy": round(sum((s[i] >= tau[t]) == human[i] for i in range(len(labels))) / len(labels), 3),
                      "kappa": round(kappa([int(v >= tau[t]) for v in s], human), 3),
                      "kappa_cv": round(kappa(cv_pred, human), 3), "auc": round(auc(s, human), 3), "tau": round(tau[t], 4)})
    write_csv(os.path.join(out_dir, "calibration_labels.csv"), labels)
    write_csv(os.path.join(out_dir, "calibration.csv"), calib)

    for r in rows_json + rows_ans + rows_ctrl:
        for t in tags:
            r[f"{t}_present"] = int(r[f"{t}_R"] >= tau[t])
    # The maximum over ~100 passages is far higher than the score of one answer, so the threshold
    # calibrated on answers says "present" for most key points of ANOTHER paper too. For the JSON the
    # threshold is set on that control instead: 95th percentile of the other paper's maxima, i.e.
    # 5% of another paper's key points pass ("spec": 95% specificity). "dec" = what the decomposition
    # uses: spec on the JSON, the human-calibrated tau on the answers.
    tau_json = {}
    for t in tags:
        ctrl = sorted(r[f"{t}_R"] for r in rows_ctrl)
        tau_json[t] = ctrl[int(0.95 * len(ctrl))]
        for r in rows_json + rows_ctrl:
            r[f"{t}_spec"] = r[f"{t}_dec"] = int(r[f"{t}_R"] >= tau_json[t])
        for r in rows_ans:
            r[f"{t}_dec"] = r[f"{t}_present"]
    calib += [{"method": f"JSON threshold {t} (95% specificity on another paper's key points)", "tau": round(tau_json[t], 4),
               "n": len(rows_ctrl)} for t in tags]
    write_csv(os.path.join(out_dir, "calibration.csv"), calib)

    # ---- coverage of the gold in the JSON, per model / config
    def cover(rows, col, keyf):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[keyf(r)][r["qid"]].append(r[col])
        out = {}
        for k, per_q in by.items():
            mean_q = {q: statistics.mean(v) for q, v in per_q.items()}
            val, by_p = macro(mean_q, q_paper)
            lo, hi = boot(by_p, args.n_boot, args.seed)
            out[k] = (100 * val, 100 * lo, 100 * hi)
        return out
    mk = lambda r: (r["model"], r["config"])
    cov = {c: cover(rows_json, c, mk) for c in ["lex", "lex_whole"] + [f"{t}_present" for t in tags] + [f"{t}_spec" for t in tags]
           + [f"{t}_R_rs" for t in tags]}
    ans_A = cover([r for r in rows_ans if r["condition"] == "A_direct_pdf"], "lex", lambda r: (r["reader"], r["config"]))
    ans_B = cover([r for r in rows_ans if r["condition"] == "B_json_same_model"], "lex", lambda r: (r["reader"], r["config"]))
    t_main = tags[-1]
    coverage = []
    for k in sorted(cov["lex"]):
        f = lambda c: f"{cov[c][k][0]:.1f} [{cov[c][k][1]:.1f}, {cov[c][k][2]:.1f}]"
        coverage.append({"model": k[0], "config": k[1], "json_lex": f("lex"), "json_lex_whole": f"{cov['lex_whole'][k][0]:.1f}",
                         **{f"json_{t}_spec": f(f"{t}_spec") for t in tags},
                         **{f"json_{t}_present_answer_tau": f"{cov[f'{t}_present'][k][0]:.1f}" for t in tags},
                         **{f"auc_own_vs_other_{t}": round(auc([r[f"{t}_R"] for r in rows_json if mk(r) == k] + [r[f"{t}_R"] for r in rows_ctrl if mk(r) == k],
                                                              [1] * sum(1 for r in rows_json if mk(r) == k) + [0] * sum(1 for r in rows_ctrl if mk(r) == k)), 3)
                            for t in tags},
                         **{f"json_{t}_R_rs_mean": f"{cov[f'{t}_R_rs'][k][0]:.1f}" for t in tags},
                         **{f"control_other_paper_{c}": f"{100 * statistics.mean(r[c] for r in rows_ctrl if (r['model'], r['config']) == k):.1f}"
                            for c in ["lex"] + [f"{t}_present" for t in tags] + [f"{t}_spec" for t in tags]},
                         "answer_A_lex": f"{ans_A[k][0]:.1f}" if k in ans_A else "", "answer_B2_lex": f"{ans_B[k][0]:.1f}" if k in ans_B else ""})
    write_csv(os.path.join(out_dir, "json_coverage.csv"), coverage)

    # ---- decomposition JSON vs answer (B2: own JSON; C: Qwen reader on each summarizer's JSON)
    jidx = {(r["model"], r["config"], r["seed"], r["kp_id"]): r for r in rows_json}
    cats = ["recovered", "lost_in_reading", "lost_in_summary", "not_from_json"]
    decomp = []
    for method in ["lex"] + [f"{t}_dec" for t in tags]:
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows_ans:
            if r["condition"] == "A_direct_pdf":
                continue
            j = jidx.get((r["summarizer"], r["config"], r["seed"], r["kp_id"]))
            if j is None:
                continue
            cat = {(1, 1): "recovered", (1, 0): "lost_in_reading", (0, 0): "lost_in_summary", (0, 1): "not_from_json"}[(j[method], r[method])]
            label = (r["condition"][0], r["reader"], r["summarizer"], r["config"])
            for c in cats:
                groups[label][c][r["qid"]].append(int(cat == c))
        for label in sorted(groups):
            row = {"method": method, "condition": label[0], "reader": label[1], "summarizer": label[2], "config": label[3]}
            for c in cats:
                val, by_p = macro({q: statistics.mean(v) for q, v in groups[label][c].items()}, q_paper)
                row[c] = round(100 * val, 1)
            decomp.append(row)
    write_csv(os.path.join(out_dir, "decomposition.csv"), decomp)

    # ---- where the key points are found (best passage of the present ones), stability, density
    where = []
    for (mdl, cfg) in sorted(cov["lex"]):
        sub = [r for r in rows_json if (r["model"], r["config"]) == (mdl, cfg) and r[f"{t_main}_spec"]]
        cnt = Counter(re.split(r"[.\[]", r[f"{t_main}_field"])[0] for r in sub)
        where.append({"model": mdl, "config": cfg, "present": len(sub),
                      **{f: round(100 * n / len(sub), 1) for f, n in cnt.most_common(6)}})
    write_csv(os.path.join(out_dir, "where_found.csv"), where)
    stab = []
    num_re = re.compile(r"\d+(?:\.\d+)?")
    for (mdl, cfg) in sorted(cov["lex"]):
        sub = [r for r in rows_json if (r["model"], r["config"]) == (mdl, cfg)]
        row = {"model": mdl, "config": cfg}
        for method in ["lex", f"{t_main}_spec"]:
            by_kp = defaultdict(list)
            for r in sub:
                by_kp[r["kp_id"]].append(r[method])
            full = [v for v in by_kp.values() if len(v) == 3]
            row[f"{method}_same_in_3_seeds_pct"] = round(100 * sum(len(set(v)) == 1 for v in full) / len(full), 1) if full else ""
        js = [j for j in jsons if (j["model"], j["config"]) == (mdl, cfg)]
        by_paper = defaultdict(list)
        for j in js:
            by_paper[j["arxiv_id"]].append(set(num_re.findall(j["text"])))
        jac = [len(a & b) / len(a | b) for sets in by_paper.values() for i, a in enumerate(sets) for b in sets[i + 1:] if a | b]
        row["numbers_jaccard_between_seeds"] = round(statistics.mean(jac), 3) if jac else ""
        words = statistics.mean(len(j["text"].split()) for j in js)
        row["json_words_mean"] = round(words)
        present = sum(r[f"{t_main}_spec"] for r in sub) / len(js)
        row[f"kp_present_per_1k_words_{t_main}"] = round(1000 * present / words, 2)
        stab.append(row)
    write_csv(os.path.join(out_dir, "stability_density.csv"), stab)

    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    write_csv(os.path.join(out_dir, "kp_json.csv"), strip(rows_json))
    write_csv(os.path.join(out_dir, "kp_answers.csv"), strip(rows_ans))
    write_csv(os.path.join(out_dir, "kp_control_other_paper.csv"), strip(rows_ctrl))

    def show(title, rows, cols):
        print(f"\n## {title}\n")
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    show("Calibration on the manual key-point labels", calib, ["method", "n", "human_present", "accuracy", "kappa", "kappa_cv", "auc", "tau"])
    show("Gold key points present in the JSON (%)", coverage, ["model", "config", "json_lex", "json_lex_whole"] + [f"json_{t}_spec" for t in tags]
         + [f"auc_own_vs_other_{t}" for t in tags] + [f"json_{t}_R_rs_mean" for t in tags] + ["answer_A_lex", "answer_B2_lex"])
    show("Control: key points of another paper found in the JSON (%; should be ~0)", coverage,
         ["model", "config"] + [f"control_other_paper_{c}" for c in ["lex"] + [f"{t}_present" for t in tags] + [f"{t}_spec" for t in tags]])
    # BERTScore decomposition (in decomposition.csv) mixes two thresholds (JSON: maximum over passages,
    # 95% specificity; answer: human-calibrated), which inflates "not_from_json": only the lexical one,
    # the same rule on both sides, is shown.
    for method in ["lex"]:
        show(f"Decomposition JSON vs answer ({method}, % of key points)", [d for d in decomp if d["method"] == method],
             ["condition", "reader", "summarizer", "config"] + cats)
    show(f"Where the present key points are ({t_main} spec, % by field)", where, list(dict.fromkeys(k for r in where for k in r)))
    show("Stability over seeds and density", stab, list(dict.fromkeys(k for r in stab for k in r)))


if __name__ == "__main__":
    main()
