#!/bin/bash
#SBATCH --job-name=ppp_smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ppp_smoke_%j.out
#SBATCH --error=logs/ppp_smoke_%j.err
# Smoke test of one model before the pipelines: server start, the checks of smoke_test.py for each
# config, then one paper through Pipeline A and step B1. From the repository root:
#   sbatch --export=ALL,MODEL=llama4scout,CONFIGS=nothink common/cluster/smoke_test.sbatch
#   sbatch --export=ALL,MODEL=mistral35,MISTRAL35_EAGLE=1,CONFIGS="think nothink" common/cluster/smoke_test.sbatch
# Outputs: logs/smoke_<model>_<config>_<job>.json, runs/smoke/ (answers and JSON of SMOKE_PAPER).

cd "$SLURM_SUBMIT_DIR"
export OUT_DIR="${OUT_DIR:-runs/smoke}" SEEDS=0
SMOKE_PAPER="${SMOKE_PAPER:-2402.11575}"
source common/cluster/job.sh
job_setup "ppp_smoke_${MODEL}"

STATUS=0
for config in $CONFIGS; do
    echo; echo "################ $MODEL $config: checks ################"
    ( model_env "$config" && python common/cluster/smoke_test.py --out "logs/smoke_${MODEL}_${config}_${SLURM_JOB_ID}.json" \
        --paper "$PDF_DIR/${SMOKE_PAPER}*.pdf" ) || STATUS=1
    echo; echo "################ $MODEL $config: one paper through Pipeline A and B1 ################"
    step A "$config" 0 "smoke_${MODEL}_${config}_A" python pipeline_a/answer_from_paper.py --benchmark-dir "$BENCHMARK_DIR" \
        "${SOURCE_ARGS[@]}" --papers "$SMOKE_PAPER" --output "$OUT_DIR/${MODEL}_${config}_s0_A.jsonl" --label "${MODEL}|$config|smoke" || STATUS=1
    step B1 "$config" 0 "smoke_${MODEL}_${config}_B1" python pipeline_b/build_json.py "${SOURCE_ARGS[@]}" --papers "$SMOKE_PAPER" \
        --output "$OUT_DIR/${MODEL}_${config}_s0_B1.jsonl" || STATUS=1
    [ -s "$OUT_DIR/${MODEL}_${config}_s0_B1.jsonl" ] && echo "[+] B1: JSON written" || { echo "[!] B1: no valid JSON"; STATUS=1; }
done
echo; [ "$STATUS" = 0 ] && echo "[+] $MODEL ready for the pipelines" || echo "[!] $MODEL: at least one check failed, see above"
exit "$STATUS"
