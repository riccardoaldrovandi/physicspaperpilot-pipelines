# Common part of run_pipeline_a.sbatch and run_pipeline_b.sbatch. Sourced from the repository root.
#   job_setup <project>        checks, energy accounting, vLLM server, pipeline Python environment
#   run_parallel <function>    <function> <config> <seed> for every CONFIGS x SEEDS, MAX_PARALLEL at a time
#
# Variables (sbatch --export=ALL,...): MODEL (required), SOURCE (pdf | latex), CONFIGS (think nothink;
# mistral35: nothink only), SEEDS (0 1 2), MAX_PARALLEL (6), OUT_DIR (runs/<SOURCE>),
# BENCHMARK_DIR, PDF_DIR, LATEX_DIR, LATEX_FIGURES_DIR, PIPELINE_VENV ($WORK/venv_vllm: needs
# requests + pymupdf), ENERGY (1) and the overrides of models.sh.

MODEL="${MODEL:?set MODEL=qwen38|gemma4|nemotron35|mistral35}"
SOURCE="${SOURCE:-pdf}"
CONFIGS="${CONFIGS:-think nothink}"
SEEDS="${SEEDS:-0 1 2}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
OUT_DIR="${OUT_DIR:-runs/$SOURCE}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark/PhysicsPaperPilotEval_v0.3.2}"
PDF_DIR="${PDF_DIR:-papers/pdf}"
LATEX_DIR="${LATEX_DIR:-papers/latex}"
LATEX_FIGURES_DIR="${LATEX_FIGURES_DIR:-papers/latex_figures}"
PIPELINE_VENV="${PIPELINE_VENV:-${WORK:-}/venv_vllm}"
case "$SOURCE" in pdf) TAG="" ;; latex) TAG="|latex" ;; *) echo "[!] SOURCE=pdf|latex"; exit 1 ;; esac
# Every image of a request is attached: the server must accept them (vllm serve --limit-mm-per-prompt).
export MAX_IMAGES="${MAX_IMAGES:-12}" SINGLE_CALL_MAX_IMAGES="${MAX_IMAGES:-12}"
SOURCE_ARGS=(--source "$SOURCE" --pdf-dir "$PDF_DIR" --latex-dir "$LATEX_DIR" --figures-dir "$LATEX_FIGURES_DIR" --cache-dir "cache/$MODEL")

job_setup() {
    case " $CONFIGS " in *" think "*) [ "$MODEL" = mistral35 ] && { echo "[!] mistral35: CONFIGS=nothink only"; exit 1; } ;; esac
    [ -f "$BENCHMARK_DIR/01_questions_for_models/questions_public.csv" ] || { echo "[!] No benchmark package in '$BENCHMARK_DIR'"; exit 1; }
    if [ "$SOURCE" = pdf ]; then
        ls "$PDF_DIR"/*.pdf > /dev/null 2>&1 || { echo "[!] No PDF in '$PDF_DIR'"; exit 1; }
    else
        ls "$LATEX_DIR"/*.tar.gz > /dev/null 2>&1 && [ -d "$LATEX_FIGURES_DIR" ] \
            || { echo "[!] No LaTeX sources in '$LATEX_DIR': run common/fetch_latex_sources.py and copy papers/ here"; exit 1; }
    fi
    mkdir -p "$OUT_DIR" logs
    module purge; module load gcc/12.2.0
    source common/cluster/models.sh || exit 1
    source common/energy/energy.sh
    export LLM_CALL_MODEL_KEY="$MODEL" LLM_CALL_SOURCE="$SOURCE" LLM_CALL_OUT_DIR="$OUT_DIR"
    energy_start "$1"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    energy_event "server_start $MODEL"
    model_start || exit 1
    energy_server_ready "http://localhost:$PORT/metrics"
    source "$PIPELINE_VENV/bin/activate"
    export PYTHONUNBUFFERED=1
    echo "[~] $MODEL: source $SOURCE, configs $CONFIGS, seeds $SEEDS, benchmark $BENCHMARK_DIR -> $OUT_DIR/"
}

run_parallel() {
    local fn="$1" pids=() config seed n p
    for config in $CONFIGS; do
        for seed in $SEEDS; do
            while :; do
                n=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done
                [ "$n" -lt "$MAX_PARALLEL" ] && break
                sleep 20
            done
            echo "[~] $(date +%H:%M) start $config s$seed"
            energy_event "chain_start $config s$seed"
            "$fn" "$config" "$seed" & pids+=($!)
        done
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    energy_event chains_done
}

# step <stage> <config> <seed> <log name> <command...>: one pipeline command in a subshell with the
# model's sampling for <config>, its own log, and the labels of the energy call log.
step() {
    local stage="$1" config="$2" seed="$3" name="$4"; shift 4
    local log="logs/${name}_${SOURCE}_${SLURM_JOB_ID:-local}.log" status
    (
        model_env "$config" || exit 1
        export LOCAL_RUN_TAG="ppp $stage $config $SOURCE" LOCAL_SEED="$seed"
        export LLM_CALL_STAGE="$stage" LLM_CALL_CONFIG="$config" LLM_CALL_SEED="$seed" LLM_CALL_RUN="$name"
        "$@"
    ) > "$log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then echo "[+] $(date +%H:%M) $name: done ($log)"
    else echo "[!] $(date +%H:%M) $name: FAILED with exit code $status, see $log"; fi
    return "$status"
}
