# Energy / usage accounting for a SLURM job. Sourced from the repository root by the sbatch scripts:
#   source common/energy/energy.sh
#   energy_start <project name>     # call log on, GPU monitor + CodeCarbon in the background
#   ... model_start ...
#   energy_server_ready             # marks the end of the start-up phase, adds the vLLM metrics URL
#   energy_event "<text>"           # any other mark (stage start / end ...)
# energy_stop runs by itself when the job ends (EXIT trap, kept together with the server's own trap).
#
# Outputs in $ENERGY_DIR (energy_logs/<job id>): job.json (SLURM allocation), events.tsv,
# calls/calls_<host>_<pid>.jsonl (one line per LLM call, common/llm_client.py), gpus.json,
# gpu_samples.csv, vllm_metrics.csv, emissions.csv (CodeCarbon), monitor.log.
# Aggregation, on the PC: python evaluation/energy_report.py energy_logs/<job id> ...
# ENERGY=0 turns everything off. Without $ENERGY_VENV (common/energy/setup_venv_energy.sh) only the
# call log and the events are written (no GPU energy).

ENERGY="${ENERGY:-1}"
ENERGY_DIR="${ENERGY_DIR:-energy_logs/${SLURM_JOB_ID:-local_$$}}"
ENERGY_VENV="${ENERGY_VENV:-${WORK:-.}/venv_energy}"
ENERGY_INTERVAL="${ENERGY_INTERVAL:-1}"
ENERGY_COUNTRY="${ENERGY_COUNTRY:-ITA}"
ENERGY_PID=""

energy_event() {
    [ "$ENERGY" = 1 ] || return 0
    printf '%s\t%s\n' "$(date +%s.%N)" "$*" >> "$ENERGY_DIR/events.tsv"
}

# Adds energy_stop in front of whatever EXIT trap is installed (the vLLM scripts install their own).
energy_trap() {
    [ "$ENERGY" = 1 ] || return 0
    local prev
    prev="$(trap -p EXIT | sed -e "s/^trap -- '//" -e "s/' EXIT\$//")"
    case "$prev" in *energy_stop*) return 0 ;; esac
    trap "energy_stop; ${prev}" EXIT
}

energy_start() {
    [ "$ENERGY" = 1 ] || return 0
    mkdir -p "$ENERGY_DIR/calls"
    ENERGY_DIR="$(cd "$ENERGY_DIR" && pwd)"
    export LLM_CALL_LOG_DIR="$ENERGY_DIR/calls"
    cat > "$ENERGY_DIR/job.json" <<EOF
{"project": "${1:-}", "job_id": "${SLURM_JOB_ID:-}", "job_name": "${SLURM_JOB_NAME:-}", "partition": "${SLURM_JOB_PARTITION:-}",
 "nodes": "${SLURM_JOB_NODELIST:-$(hostname)}", "gpus_on_node": "${SLURM_GPUS_ON_NODE:-}", "cpus_per_task": "${SLURM_CPUS_PER_TASK:-}",
 "mem_per_node_mb": "${SLURM_MEM_PER_NODE:-}", "submit_dir": "${SLURM_SUBMIT_DIR:-$PWD}", "start_epoch": $(date +%s)}
EOF
    energy_event job_start
    if [ -x "$ENERGY_VENV/bin/python" ]; then
        "$ENERGY_VENV/bin/python" common/energy/gpu_monitor.py --out-dir "$ENERGY_DIR" --interval "$ENERGY_INTERVAL" \
            --country "$ENERGY_COUNTRY" --metrics-url-file "$ENERGY_DIR/metrics_url" --project "${1:-paper_analyzer}" \
            > "$ENERGY_DIR/monitor.log" 2>&1 &
        ENERGY_PID=$!
        echo "[~] Energy monitor started (pid $ENERGY_PID) -> $ENERGY_DIR"
    else
        echo "[!] $ENERGY_VENV missing (common/energy/setup_venv_energy.sh): call log only, no GPU energy."
    fi
    energy_trap
}

# energy_server_ready [metrics url]: end of the model start-up; the monitor starts scraping vLLM.
energy_server_ready() {
    [ "$ENERGY" = 1 ] || return 0
    energy_event server_ready
    [ -n "${1:-}" ] && echo "$1" > "$ENERGY_DIR/metrics_url"
    energy_trap
}

energy_stop() {
    [ "$ENERGY" = 1 ] || return 0
    [ -n "${ENERGY_STOPPED:-}" ] && return 0
    ENERGY_STOPPED=1
    energy_event job_end
    if [ -n "$ENERGY_PID" ] && kill -0 "$ENERGY_PID" 2>/dev/null; then
        kill -TERM "$ENERGY_PID"
        local i
        for i in $(seq 60); do kill -0 "$ENERGY_PID" 2>/dev/null || break; sleep 1; done
    fi
    echo "[+] Energy logs in $ENERGY_DIR"
}
