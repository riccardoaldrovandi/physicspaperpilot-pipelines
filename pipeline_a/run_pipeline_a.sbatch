#!/bin/bash
#SBATCH --job-name=ppp_A
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ppp_A_%j.out
#SBATCH --error=logs/ppp_A_%j.err
# Pipeline A on Leonardo: one model per job, every config and seed. From the repository root, with
# SBATCH_ACCOUNT set in your shell:
#   sbatch --export=ALL,MODEL=qwen38 pipeline_a/run_pipeline_a.sbatch
#   sbatch --export=ALL,MODEL=mistral35,CONFIGS=nothink pipeline_a/run_pipeline_a.sbatch
#   sbatch --export=ALL,MODEL=gemma4,SOURCE=latex pipeline_a/run_pipeline_a.sbatch
# Outputs: $OUT_DIR/<model>_<config>_s<seed>_A.{jsonl,csv}; energy_logs/<job id>/. Resubmit to resume.
# All variables: common/cluster/job.sh.

cd "$SLURM_SUBMIT_DIR"
source common/cluster/job.sh
job_setup "ppp_A_${MODEL}_${SOURCE}"

chain() {
    local name="${MODEL}_$1_s$2_A"
    step A "$1" "$2" "$name" python pipeline_a/answer_from_paper.py --benchmark-dir "$BENCHMARK_DIR" "${SOURCE_ARGS[@]}" \
        --output "$OUT_DIR/$name.jsonl" --label "${MODEL}|$1|s$2|A_direct_pdf${TAG}"
}
run_parallel chain
ls -1 "$OUT_DIR"/"${MODEL}"_*_A.csv 2>/dev/null | while read -r f; do echo "  $f: $(($(wc -l < "$f") - 1)) answers"; done
