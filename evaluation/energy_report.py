"""Energy, time and emissions report of pipeline jobs, from the logs of common/energy/energy.sh.

For every job directory (energy_logs/<job id>, pulled from Leonardo):
  - total GPU energy from the NVML energy counters (or, where a GPU has none, power integrated over
    time), split into start-up (job start -> server ready), busy (at least one LLM call in flight)
    and idle; node energy (CPU + GPU + RAM) and CO2-equivalent from CodeCarbon;
  - every interval between two GPU samples is shared among the calls in flight, in proportion to
    their overlap with it ("time" attribution, default) or to overlap x tokens per second of the call
    ("tokens"); the part of the interval with no call in flight is idle / start-up energy. The sum over
    the calls is therefore exactly the busy energy of the job (checked).
Outputs in --out-dir:
  calls.csv        one row per call: labels, latency, tokens, concurrency, attributed GPU energy, CO2;
  jobs.csv         one row per job: wall time, GPU-hours allocated and busy, energy by phase, average
                   and peak power, utilization, peak memory, CodeCarbon totals, CO2 with and without PUE;
  by_stage.csv     per model / source / config / seed / stage: calls, errors, latency mean / p50 / p95,
                   tokens, throughput, GPU Wh, Wh per call, Wh per 1k tokens, CO2;
  per_unit.csv     energy per question (A, B2, C) and per paper (B1);
  breakeven.csv    Pipeline A vs B: questions per paper above which B (one B1 JSON + cheap B2
                   questions) costs less GPU energy than A, next to the quality of A and B if
                   --runs-dir has the scores of evaluation/score_runs.py;
  report.md        the main tables.
CO2: kg = kWh x intensity x PUE. Intensity (g/kWh): --carbon-intensity, else the one CodeCarbon used
for the job (emissions / energy). PUE: --pue (default 1.0 = IT equipment only).
Usage:
  python evaluation/energy_report.py energy_logs/64700001 energy_logs/64700002 \\
      --out-dir energy_logs/report --pue 1.1 --runs-dir runs/pdf
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
J_PER_WH = 3600.0


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] {path} ({len(rows)} rows)")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---------------------------------------------------------------- loading
def load_job(d):
    job = {"dir": d, "id": os.path.basename(os.path.normpath(d))}
    meta_path = os.path.join(d, "job.json")
    job["meta"] = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    gpus_path = os.path.join(d, "gpus.json")
    job["gpus"] = json.load(open(gpus_path)) if os.path.exists(gpus_path) else {"gpus": []}
    job["events"] = []
    if os.path.exists(os.path.join(d, "events.tsv")):
        for line in open(os.path.join(d, "events.tsv"), encoding="utf-8"):
            t, _, text = line.rstrip("\n").partition("\t")
            if fnum(t) is not None:
                job["events"].append((float(t), text))
    job["calls"] = []
    for path in sorted(glob.glob(os.path.join(d, "calls", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fnum(c.get("t_start")) is not None and fnum(c.get("t_end")) is not None:
                c["job"] = job["id"]
                job["calls"].append(c)
    job["calls"].sort(key=lambda c: c["t_start"])
    job["samples"] = read_csv(os.path.join(d, "gpu_samples.csv"))
    job["emissions"] = read_csv(os.path.join(d, "emissions.csv"))
    return job


def energy_series(samples):
    """[(t, cumulative GPU energy over all GPUs in J)] and per-GPU stats. Uses the energy counter of a
    GPU when it has one, else trapezoidal integration of its power."""
    by_gpu = defaultdict(list)
    for s in samples:
        t = fnum(s["t"])
        if t is not None:
            by_gpu[s["gpu"]].append((t, fnum(s.get("energy_mj")), fnum(s.get("power_w")), s))
    per_gpu_cum, stats = {}, {}
    for g, rows in by_gpu.items():
        rows.sort(key=lambda r: r[0])
        use_counter = all(r[1] is not None for r in rows)
        cum, acc = [], 0.0
        for i, (t, e, p, _) in enumerate(rows):
            if use_counter:
                acc = (e - rows[0][1]) / 1000.0
            elif i > 0:
                p0, p1 = rows[i - 1][2], p
                if p0 is not None and p1 is not None:
                    acc += (p0 + p1) / 2 * (t - rows[i - 1][0])
            cum.append((t, acc))
        per_gpu_cum[g] = cum
        powers = [r[2] for r in rows if r[2] is not None]
        utils = [fnum(r[3].get("util_gpu")) for r in rows if fnum(r[3].get("util_gpu")) is not None]
        mems = [fnum(r[3].get("mem_used_mib")) for r in rows if fnum(r[3].get("mem_used_mib")) is not None]
        stats[g] = {"counter": use_counter, "energy_j": acc, "power_mean_w": statistics.mean(powers) if powers else None,
                    "power_peak_w": max(powers) if powers else None, "util_mean": statistics.mean(utils) if utils else None,
                    "mem_peak_mib": max(mems) if mems else None}
    # Common time grid: the sample times of the first GPU (all GPUs are sampled in the same loop).
    if not per_gpu_cum:
        return [], stats
    grid = [t for t, _ in next(iter(per_gpu_cum.values()))]
    total = []
    for k, t in enumerate(grid):
        total.append((t, sum(cum[min(k, len(cum) - 1)][1] for cum in per_gpu_cum.values())))
    return total, stats


# ---------------------------------------------------------------- attribution
def attribute(job, mode):
    """Shares the GPU energy of every sample interval among the calls in flight; returns phase totals (J)."""
    series = job["series"]
    calls = job["calls"]
    for c in calls:
        c["gpu_j"] = 0.0
        c["_conc"] = 0.0
        dur = max(c["t_end"] - c["t_start"], 1e-6)
        toks = (fnum(c.get("prompt_tokens")) or 0) + (fnum(c.get("completion_tokens")) or 0)
        c["_rate"] = toks / dur if toks else 1.0
    ready = next((t for t, e in job["events"] if e.startswith("server_ready")), None)
    phases = {"startup_j": 0.0, "busy_j": 0.0, "idle_j": 0.0, "busy_s": 0.0}
    j = 0
    active = []
    for k in range(1, len(series)):
        a, b = series[k - 1][0], series[k][0]
        de = series[k][1] - series[k - 1][1]
        length = b - a
        if length <= 0:
            continue
        while j < len(calls) and calls[j]["t_start"] < b:
            active.append(calls[j])
            j += 1
        active = [c for c in active if c["t_end"] > a]
        overlaps = [(c, min(b, c["t_end"]) - max(a, c["t_start"])) for c in active]
        overlaps = [(c, o) for c, o in overlaps if o > 0]
        # Part of the interval covered by at least one call.
        spans = sorted((max(a, c["t_start"]), min(b, c["t_end"])) for c, _ in overlaps)
        covered, cur_s, cur_e = 0.0, None, None
        for s, e in spans:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            covered += cur_e - cur_s
        busy_e = de * covered / length
        rest = de - busy_e
        if ready is not None and b <= ready:
            phases["startup_j"] += rest
        elif ready is not None and a < ready:
            phases["startup_j"] += rest * (ready - a) / length
            phases["idle_j"] += rest * (b - ready) / length
        else:
            phases["idle_j"] += rest
        phases["busy_j"] += busy_e
        phases["busy_s"] += covered
        if overlaps:
            weights = [o * (c["_rate"] if mode == "tokens" else 1.0) for c, o in overlaps]
            wsum = sum(weights)
            for (c, o), w in zip(overlaps, weights):
                c["gpu_j"] += busy_e * w / wsum
                c["_conc"] += o * len(overlaps)
    for c in calls:
        dur = max(c["t_end"] - c["t_start"], 1e-6)
        c["concurrency_mean"] = round(c.pop("_conc") / dur, 2)
        c.pop("_rate", None)
    return phases


# ---------------------------------------------------------------- report
def intensity_of(job, override):
    if override is not None:
        return override, "--carbon-intensity"
    em = job["emissions"]
    if em:
        last = em[-1]
        e_kwh, co2_kg = fnum(last.get("energy_consumed")), fnum(last.get("emissions"))
        if e_kwh and co2_kg is not None:
            return 1000.0 * co2_kg / e_kwh, f"CodeCarbon ({last.get('country_name') or last.get('country_iso_code', '')})"
    return None, "none"


def label(c, key):
    v = c.get(key)
    return "" if v is None else str(v)


def summarize(calls, keys, intensity_by_job, pue):
    groups = defaultdict(list)
    for c in calls:
        groups[tuple(label(c, k) for k in keys)].append(c)
    out = []
    for gk, cs in sorted(groups.items()):
        ok = [c for c in cs if c.get("status") == "ok"]
        lat = [c["latency_s"] for c in cs]
        ptok = sum(fnum(c.get("prompt_tokens")) or 0 for c in ok)
        ctok = sum(fnum(c.get("completion_tokens")) or 0 for c in ok)
        wh = sum(c["gpu_j"] for c in cs) / J_PER_WH
        # g: kWh x g/kWh x PUE
        co2 = sum(c["gpu_j"] / J_PER_WH / 1000 * (intensity_by_job.get(c["job"]) or 0) * pue for c in cs)
        row = dict(zip(keys, gk))
        row.update({"calls": len(cs), "errors": len(cs) - len(ok),
                    "finish_length": sum(1 for c in ok if c.get("finish_reason") == "length"),
                    "latency_mean_s": round(statistics.mean(lat), 2), "latency_p50_s": round(pct(lat, 0.5), 2),
                    "latency_p95_s": round(pct(lat, 0.95), 2), "latency_sum_h": round(sum(lat) / 3600, 3),
                    "prompt_tokens": int(ptok), "completion_tokens": int(ctok),
                    "reasoning_tokens": int(sum(fnum(c.get("reasoning_tokens")) or 0 for c in ok)),
                    "images": int(sum(fnum(c.get("n_images")) or 0 for c in cs)),
                    "completion_tokens_per_s_per_call": round(statistics.mean([(fnum(c.get("completion_tokens")) or 0) / max(c["latency_s"], 1e-6) for c in ok]), 1) if ok else "",
                    "concurrency_mean": round(statistics.mean(c["concurrency_mean"] for c in cs), 2),
                    "gpu_wh": round(wh, 3), "gpu_wh_per_call": round(wh / len(cs), 4),
                    "gpu_wh_per_1k_tokens": round(1000 * wh / (ptok + ctok), 4) if ptok + ctok else "",
                    "co2_g": round(co2, 6) if intensity_by_job else ""})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job_dirs", nargs="+", help="energy_logs/<job id> directories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attribution", choices=["time", "tokens"], default="time")
    ap.add_argument("--pue", type=float, default=1.0, help="Power usage effectiveness of the data centre (1.0 = IT only)")
    ap.add_argument("--carbon-intensity", type=float, default=None, help="gCO2e/kWh; default: CodeCarbon's value for the job")
    ap.add_argument("--runs-dir", default=None, help="Runs with scores/summary.csv (evaluation/score_runs.py) for the quality columns")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs, all_calls, intensity_by_job, job_rows = [], [], {}, []
    for d in args.job_dirs:
        job = load_job(d)
        job["series"], gstats = energy_series(job["samples"])
        phases = attribute(job, args.attribution)
        total_j = job["series"][-1][1] if job["series"] else 0.0
        attributed = sum(c["gpu_j"] for c in job["calls"])
        if job["series"] and abs(attributed - phases["busy_j"]) > 1e-6 * max(1.0, total_j):
            raise SystemExit(f"[!] {job['id']}: attributed {attributed:.3f} J != busy {phases['busy_j']:.3f} J")
        if job["series"] and abs(phases["startup_j"] + phases["busy_j"] + phases["idle_j"] - total_j) > 1e-6 * max(1.0, total_j):
            raise SystemExit(f"[!] {job['id']}: phases do not add up to the total GPU energy")
        intensity, source = intensity_of(job, args.carbon_intensity)
        if intensity is not None:
            intensity_by_job[job["id"]] = intensity
        ev = dict((e.split()[0], t) for t, e in job["events"])
        t0 = ev.get("job_start", job["series"][0][0] if job["series"] else None)
        t1 = ev.get("job_end", job["series"][-1][0] if job["series"] else None)
        wall_h = (t1 - t0) / 3600 if t0 and t1 else None
        n_gpu = len(gstats) or len(job["gpus"].get("gpus", []))
        em = job["emissions"][-1] if job["emissions"] else {}
        node_kwh = fnum(em.get("energy_consumed"))
        row = {"job": job["id"], "project": job["meta"].get("project", ""), "nodes": job["meta"].get("nodes", ""),
               "gpus": n_gpu, "gpu_model": (job["gpus"].get("gpus") or [{}])[0].get("name", ""),
               "energy_counter": all(s["counter"] for s in gstats.values()) if gstats else "",
               "calls": len(job["calls"]), "call_errors": sum(1 for c in job["calls"] if c.get("status") != "ok"),
               "wall_h": round(wall_h, 3) if wall_h else "",
               "startup_h": round((ev["server_ready"] - t0) / 3600, 3) if "server_ready" in ev and t0 else "",
               "busy_h": round(phases["busy_s"] / 3600, 3),
               "gpu_hours_allocated": round(n_gpu * wall_h, 2) if wall_h else "",
               "gpu_hours_busy": round(n_gpu * phases["busy_s"] / 3600, 2),
               "gpu_kwh_total": round(total_j / J_PER_WH / 1000, 4),
               "gpu_kwh_startup": round(phases["startup_j"] / J_PER_WH / 1000, 4),
               "gpu_kwh_busy": round(phases["busy_j"] / J_PER_WH / 1000, 4),
               "gpu_kwh_idle": round(phases["idle_j"] / J_PER_WH / 1000, 4),
               "gpu_power_mean_w": round(sum(s["power_mean_w"] or 0 for s in gstats.values()), 1),
               "gpu_power_peak_w_per_gpu": round(max((s["power_peak_w"] or 0) for s in gstats.values()), 1) if gstats else "",
               "gpu_util_mean_pct": round(statistics.mean(s["util_mean"] for s in gstats.values() if s["util_mean"] is not None), 1) if gstats else "",
               "gpu_mem_peak_gib_per_gpu": round(max((s["mem_peak_mib"] or 0) for s in gstats.values()) / 1024, 1) if gstats else "",
               "node_kwh_codecarbon": round(node_kwh, 4) if node_kwh is not None else "",
               "cpu_kwh_codecarbon": round(fnum(em.get("cpu_energy")) or 0, 4) if em else "",
               "gpu_kwh_codecarbon": round(fnum(em.get("gpu_energy")) or 0, 4) if em else "",
               "ram_kwh_codecarbon": round(fnum(em.get("ram_energy")) or 0, 4) if em else "",
               "cpu_model": em.get("cpu_model", ""), "carbon_intensity_g_per_kwh": round(intensity, 1) if intensity else "",
               "carbon_intensity_source": source, "pue": args.pue,
               "co2_kg_gpu_nvml": round(total_j / J_PER_WH / 1000 * intensity / 1000 * args.pue, 4) if intensity else "",
               "co2_kg_node_codecarbon": round(node_kwh * intensity / 1000 * args.pue, 4) if intensity and node_kwh is not None else ""}
        job_rows.append(row)
        jobs.append(job)
        all_calls.extend(job["calls"])
        print(f"[~] {job['id']}: {len(job['calls'])} calls, GPU {row['gpu_kwh_total']} kWh "
              f"(start-up {row['gpu_kwh_startup']}, busy {row['gpu_kwh_busy']}, idle {row['gpu_kwh_idle']})")

    for c in all_calls:
        c["arxiv_id"] = (ARXIV_ID_RE.search(str(c.get("paper", ""))) or [None, ""])[1]
        c["gpu_wh"] = round(c["gpu_j"] / J_PER_WH, 5)
        # kWh x g/kWh = g
        c["co2_g"] = round(c["gpu_j"] / J_PER_WH / 1000 * intensity_by_job.get(c["job"], 0) * args.pue, 8) if intensity_by_job else ""
    call_fields = ["job", "model_key", "source", "config", "seed", "stage", "run", "arxiv_id", "question_id", "question_type",
                   "t_start", "t_end", "latency_s", "status", "finish_reason", "prompt_tokens", "completion_tokens",
                   "reasoning_tokens", "reasoning_chars", "n_images", "guided_json", "concurrency_mean", "gpu_wh", "co2_g", "error"]
    write_csv(os.path.join(args.out_dir, "calls.csv"), [{k: c.get(k, "") for k in call_fields} for c in all_calls])
    write_csv(os.path.join(args.out_dir, "jobs.csv"), job_rows)
    keys = ["model_key", "source", "config", "seed", "stage"]
    by_stage = summarize(all_calls, keys, intensity_by_job, args.pue)
    write_csv(os.path.join(args.out_dir, "by_stage.csv"), by_stage)

    # Energy per unit: per question for A / B2 / C, per paper for B1 (mean over units, each unit = sum of its calls).
    units = defaultdict(lambda: defaultdict(float))
    for c in all_calls:
        st = c.get("stage", "")
        unit = c.get("arxiv_id", "") if st == "B1" else c.get("question_id", "")
        units[tuple(label(c, k) for k in keys)][unit] += c["gpu_j"] / J_PER_WH
    per_unit = []
    for gk, u in sorted(units.items()):
        vals = list(u.values())
        per_unit.append({**dict(zip(keys, gk)), "unit": "paper" if gk[-1] == "B1" else "question", "units": len(vals),
                         "gpu_wh_per_unit_mean": round(statistics.mean(vals), 4), "gpu_wh_per_unit_p95": round(pct(vals, 0.95), 4)})
    write_csv(os.path.join(args.out_dir, "per_unit.csv"), per_unit)

    # Break-even A vs B per model / source / config / seed.
    pu = {tuple(r[k] for k in keys): r["gpu_wh_per_unit_mean"] for r in per_unit}
    quality = {}
    if args.runs_dir:
        for r in read_csv(os.path.join(args.runs_dir, "scores", "summary.csv")):
            quality[(r["reader"], r["config"], r["condition"], r.get("summarizer", ""))] = r
    be = []
    for (m, src, cfg, seed, st), ea in sorted(pu.items()):
        if st != "A":
            continue
        e1, e2 = pu.get((m, src, cfg, seed, "B1")), pu.get((m, src, cfg, seed, "B2"))
        if e1 is None or e2 is None:
            continue
        diff = ea - e2
        row = {"model_key": m, "source": src, "config": cfg, "seed": seed, "A_wh_per_question": ea,
               "B1_wh_per_paper": e1, "B2_wh_per_question": e2,
               "breakeven_questions_per_paper": round(e1 / diff, 2) if diff > 0 else "never (B2 >= A)",
               "A_wh_3_questions": round(3 * ea, 4), "B_wh_3_questions": round(e1 + 3 * e2, 4),
               "B_over_A_at_3_questions": round((e1 + 3 * e2) / (3 * ea), 3) if ea else ""}
        qa, qb = quality.get((m, cfg, "A_direct_pdf", "")), quality.get((m, cfg, "B_json_same_model", m))
        if qa and qb:
            for col in ("short_keypoint_coverage_percent", "open_keypoint_coverage_percent", "short_token_f1_percent"):
                row[f"A_{col}"], row[f"B_{col}"] = qa.get(col, ""), qb.get(col, "")
        be.append(row)
    write_csv(os.path.join(args.out_dir, "breakeven.csv"), be)

    def md(rows, cols):
        if not rows:
            return "_none_\n"
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
        return "\n".join(out) + "\n"
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Energy report\n\nAttribution: {args.attribution}; PUE {args.pue}; jobs: {', '.join(j['id'] for j in jobs)}.\n\n## Jobs\n\n")
        f.write(md(job_rows, ["job", "project", "gpus", "wall_h", "startup_h", "busy_h", "gpu_hours_allocated", "gpu_hours_busy",
                              "gpu_kwh_total", "gpu_kwh_startup", "gpu_kwh_busy", "gpu_kwh_idle", "gpu_power_mean_w",
                              "gpu_util_mean_pct", "node_kwh_codecarbon", "carbon_intensity_g_per_kwh", "co2_kg_node_codecarbon"]))
        f.write("\n## By stage\n\n")
        f.write(md(by_stage, keys + ["calls", "errors", "latency_mean_s", "latency_p95_s", "prompt_tokens", "completion_tokens",
                                     "concurrency_mean", "gpu_wh", "gpu_wh_per_call", "gpu_wh_per_1k_tokens", "co2_g"]))
        f.write("\n## Pipeline A vs B\n\n")
        f.write(md(be, ["model_key", "source", "config", "seed", "A_wh_per_question", "B1_wh_per_paper", "B2_wh_per_question",
                        "breakeven_questions_per_paper", "B_over_A_at_3_questions"]))
    print(f"[+] {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
