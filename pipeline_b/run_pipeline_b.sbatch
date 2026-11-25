#!/bin/bash
#SBATCH --job-name=ppp_B
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ppp_B_%j.out
#SBATCH --error=logs/ppp_B_%j.err
# Pipeline B on Leonardo: one model per job, every config and seed. From the repository root, with
# SBATCH_ACCOUNT set in your shell:
#   sbatch --export=ALL,MODEL=qwen38 pipeline_b/run_pipeline_b.sbatch                 # B1 then B2
#   sbatch --export=ALL,MODEL=mistral35,CONFIGS=nothink pipeline_b/run_pipeline_b.sbatch
# and, once every model has its B1 JSONs, the fixed reader on all of them (condition C):
#   sbatch --export=ALL,MODEL=qwen38,STAGES=C,CONFIGS=nothink pipeline_b/run_pipeline_b.sbatch
# STAGES: "B1 B2" (default) | B1 | B2 | C. JSON_MODELS: summarizers read in stage C.
# Outputs: $OUT_DIR/<model>_<config>_s<seed>_B1.jsonl (+ .rejected.jsonl), <..>_B2.{jsonl,csv}
# (+ .evidence_fields.csv), C_<reader>_<config>_from_<summarizer>_s<seed>.*; energy_logs/<job id>/.
# Resubmit to resume. All variables: common/cluster/job.sh.

cd "$SLURM_SUBMIT_DIR"
source common/cluster/job.sh
STAGES="${STAGES:-B1 B2}"
JSON_MODELS="${JSON_MODELS:-qwen38 gemma4 nemotron35 mistral35}"
has() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
job_setup "ppp_B_${MODEL}_${SOURCE}"

chain() {
    local c="$1" s="$2" base="${MODEL}_$1_s$2" summ
    has B1 && step B1 "$c" "$s" "${base}_B1" python pipeline_b/build_json.py "${SOURCE_ARGS[@]}" --output "$OUT_DIR/${base}_B1.jsonl"
    has B2 && step B2 "$c" "$s" "${base}_B2" python pipeline_b/answer_from_json.py --benchmark-dir "$BENCHMARK_DIR" \
        --json-input "$OUT_DIR/${base}_B1.jsonl" --output "$OUT_DIR/${base}_B2.jsonl" --label "${MODEL}|$c|s$s|B_json_same_model${TAG}"
    if has C; then
        for summ in $JSON_MODELS; do
            [ -s "$OUT_DIR/${summ}_${c}_s${s}_B1.jsonl" ] || { echo "[!] C: no B1 JSONs of $summ ($c, s$s), skipped"; continue; }
            step C "$c" "$s" "C_${MODEL}_${c}_from_${summ}_s${s}" python pipeline_b/answer_from_json.py --benchmark-dir "$BENCHMARK_DIR" \
                --json-input "$OUT_DIR/${summ}_${c}_s${s}_B1.jsonl" --output "$OUT_DIR/C_${MODEL}_${c}_from_${summ}_s${s}.jsonl" \
                --label "R=${MODEL}|S=${summ}|$c|s$s|C_fixed_reader${TAG}"
        done
    fi
}
run_parallel chain
