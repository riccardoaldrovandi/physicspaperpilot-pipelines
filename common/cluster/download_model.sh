#!/bin/bash
# Downloads one of the four models to $MODELS_DIR. On a LOGIN node (compute nodes have no internet);
# resumable, just run it again. Gemma 4 is gated: accept its license and `hf auth login` first.
#   MODEL=qwen38 bash common/cluster/download_model.sh      # qwen38 | gemma4 | nemotron35 | mistral35
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/cluster/models.sh
HF_BIN="$(command -v hf || echo "$WORK_DIR/venv_vllm/bin/hf")"
[ -x "$HF_BIN" ] || { echo "[!] 'hf' CLI not found (pip install huggingface_hub)"; exit 1; }
mkdir -p "$MODEL_DIR"
case "$MODEL" in
    qwen38)     "$HF_BIN" download Qwen/Qwen3.8-27B --local-dir "$MODEL_DIR"; INDEX=model.safetensors.index.json ;;
    gemma4)     "$HF_BIN" download google/gemma-4-31B-it --local-dir "$MODEL_DIR"; INDEX=model.safetensors.index.json ;;
    nemotron35) "$HF_BIN" download RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8 --local-dir "$MODEL_DIR" --exclude "__pycache__/*"
                INDEX=model.safetensors.index.json ;;
    # Mistral-native format only (consolidated-*.safetensors + params.json + tekken.json).
    mistral35)  "$HF_BIN" download mistralai/Mistral-Medium-3.5-128B --local-dir "$MODEL_DIR" \
                    --include "consolidated*" "params.json" "tekken.json" "SYSTEM_PROMPT.txt" "generation_config.json"
                INDEX=consolidated.safetensors.index.json ;;
esac
# Every shard of the index must be on disk, otherwise vLLM fails late, on the compute node.
python3 - "$MODEL_DIR" "$INDEX" <<'PY'
import json, os, sys
d, index = sys.argv[1], sys.argv[2]
shards = sorted(set(json.load(open(os.path.join(d, index)))["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
print(f"[{'!' if missing else '+'}] {len(shards) - len(missing)}/{len(shards)} shards present")
sys.exit(1 if missing else 0)
PY
# vLLM 0.32 names Nemotron's embeddings differently from the FP8 checkpoint's ignore rule.
[ "$MODEL" = nemotron35 ] && python3 common/cluster/patch_nemotron_config.py "$MODEL_DIR"
echo "[+] $MODEL ready in $MODEL_DIR"
