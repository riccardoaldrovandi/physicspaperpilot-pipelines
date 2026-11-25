"""GPU / node energy monitor for a whole job, run in the background next to the pipeline.

Writes to --out-dir:
  gpus.json         the GPUs seen by NVML (index, name, uuid, power limit, energy counter support);
  gpu_samples.csv   every --interval seconds, per GPU: cumulative energy counter (mJ, NVML
                    nvmlDeviceGetTotalEnergyConsumption, Volta and later), power (W), GPU and
                    memory utilization (%), memory used (MiB), SM clock (MHz), temperature (C);
  vllm_metrics.csv  with --metrics-url, every --metrics-interval seconds, the vLLM server gauges
                    (requests running / waiting, KV-cache usage, prefix-cache counters);
  emissions.csv     CodeCarbon (offline, --country) for the whole run: CPU, GPU and RAM energy,
                    CO2-equivalent emissions and the carbon intensity it used (skipped with
                    --no-codecarbon or if CodeCarbon is not installed).
Stops on SIGTERM / SIGINT, flushing everything. Needs nvidia-ml-py (and codecarbon) only: it runs
in its own small environment (common/energy/setup_venv_energy.sh), not in the vLLM one.
"""
import signal

_stop = False


def _on_signal(*_):
    global _stop
    _stop = True


# Installed before anything slow (imports, CodeCarbon start-up): a job that ends early still gets
# its final samples written instead of the monitor being killed half-way.
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import pynvml  # noqa: E402

GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:gpu_cache_usage_perc",
          "vllm:kv_cache_usage_perc", "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_queries_total",
          "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
          "vllm:prompt_tokens_total", "vllm:generation_tokens_total")
def gpu_info(handles):
    out = []
    for i, h in enumerate(handles):
        info = {"index": i, "name": pynvml.nvmlDeviceGetName(h), "uuid": pynvml.nvmlDeviceGetUUID(h)}
        try:
            info["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        except pynvml.NVMLError:
            info["power_limit_w"] = None
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            info["energy_counter"] = True
        except pynvml.NVMLError:
            info["energy_counter"] = False
        out.append(info)
    return out


def sample(i, h, t):
    def get(fn, *a, scale=1.0):
        try:
            return round(fn(h, *a) * scale, 3)
        except pynvml.NVMLError:
            return ""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        ug, um = util.gpu, util.memory
    except pynvml.NVMLError:
        ug = um = ""
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        used, total = round(mem.used / 2**20), round(mem.total / 2**20)
    except pynvml.NVMLError:
        used = total = ""
    return {"t": round(t, 3), "gpu": i,
            "energy_mj": get(pynvml.nvmlDeviceGetTotalEnergyConsumption),
            "power_w": get(pynvml.nvmlDeviceGetPowerUsage, scale=1e-3),
            "util_gpu": ug, "util_mem": um, "mem_used_mib": used, "mem_total_mib": total,
            "sm_clock_mhz": get(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM),
            "temp_c": get(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU)}


def scrape(url, t):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(GAUGES):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{")[0]
        if name in GAUGES:
            rows.append({"t": round(t, 3), "metric": name, "value": value})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between GPU samples")
    ap.add_argument("--metrics-url", default="", help="vLLM Prometheus endpoint, e.g. http://localhost:8000/metrics")
    ap.add_argument("--metrics-url-file", default="", help="File whose first line is the metrics URL, read when it appears (the server starts after the monitor)")
    ap.add_argument("--metrics-interval", type=float, default=5.0)
    ap.add_argument("--country", default="ITA", help="ISO 3166-1 alpha-3 code for CodeCarbon's carbon intensity")
    ap.add_argument("--region", default=None, help="Optional CodeCarbon region")
    ap.add_argument("--codecarbon-interval", type=float, default=15.0)
    ap.add_argument("--project", default="paper_analyzer")
    ap.add_argument("--no-codecarbon", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    with open(os.path.join(args.out_dir, "gpus.json"), "w") as f:
        json.dump({"host": os.uname().nodename, "started": time.time(), "interval_s": args.interval,
                   "gpus": gpu_info(handles)}, f, indent=1)

    tracker = None
    if not args.no_codecarbon and not _stop:
        try:
            from codecarbon import OfflineEmissionsTracker
            tracker = OfflineEmissionsTracker(country_iso_code=args.country, region=args.region, output_dir=args.out_dir,
                                              project_name=args.project, measure_power_secs=args.codecarbon_interval,
                                              tracking_mode="machine", log_level="error",
                                              allow_multiple_runs=True)
            tracker.start()
        except Exception as e:
            print(f"[!] CodeCarbon not started ({type(e).__name__}: {e}); GPU samples only.", file=sys.stderr)
            tracker = None

    fields = list(sample(0, handles[0], time.time())) if handles else ["t"]
    gf = open(os.path.join(args.out_dir, "gpu_samples.csv"), "a", newline="")
    gw = csv.DictWriter(gf, fieldnames=fields)
    if gf.tell() == 0:
        gw.writeheader()
    mf = open(os.path.join(args.out_dir, "vllm_metrics.csv"), "a", newline="")
    mw = csv.DictWriter(mf, fieldnames=["t", "metric", "value"])
    if mf.tell() == 0:
        mw.writeheader()

    url, next_metrics, n = args.metrics_url, 0.0, 0
    try:
        while not _stop:
            t = time.time()
            for i, h in enumerate(handles):
                gw.writerow(sample(i, h, t))
            if not url and args.metrics_url_file and os.path.exists(args.metrics_url_file):
                with open(args.metrics_url_file) as f:
                    url = f.readline().strip()
            if url and t >= next_metrics:
                mw.writerows(scrape(url, t))
                next_metrics = t + args.metrics_interval
            n += 1
            if n % 30 == 0:
                gf.flush()
                mf.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t)))
    finally:
        for i, h in enumerate(handles):
            gw.writerow(sample(i, h, time.time()))
        gf.close()
        mf.close()
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as e:
                print(f"[!] CodeCarbon stop failed: {e}", file=sys.stderr)
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
