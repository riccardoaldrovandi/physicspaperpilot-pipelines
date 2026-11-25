#!/bin/bash
# One-time setup of a vLLM runtime on Leonardo (see runtime.sh). From the repository root:
#   bash common/cluster/setup_runtime.sh vllm019     # on a login node: venv (glibc >= 2.31) or Apptainer image
#   sbatch -p lrd_all_serial --time=02:00:00 --mem=30G -c 8 -o logs/setup_%j.out \
#       --wrap "bash common/cluster/setup_runtime.sh vllm032"   # image build: too heavy for a login node
# Rerunning only repeats the checks.
set -euo pipefail
cd "$(dirname "$0")/../.."
source common/cluster/runtime.sh
WHICH="${1:?usage: setup_runtime.sh vllm019|vllm032}"

pull() {   # pull <image file> <docker tag>
    [ -n "$APPTAINER_BIN" ] || { echo "[!] apptainer/singularity not found on this node."; exit 1; }
    mkdir -p "$(dirname "$1")" "$WORK_DIR/.cache/apptainer" "$WORK_DIR/.cache/apptainer_tmp"
    # Image layers are several GB: cache and build dir off $HOME and /tmp.
    export APPTAINER_CACHEDIR="$WORK_DIR/.cache/apptainer" SINGULARITY_CACHEDIR="$WORK_DIR/.cache/apptainer"
    export APPTAINER_TMPDIR="$WORK_DIR/.cache/apptainer_tmp" SINGULARITY_TMPDIR="$WORK_DIR/.cache/apptainer_tmp"
    [ -f "$1" ] && echo "[~] $1 already there" || "$APPTAINER_BIN" pull "$1" "docker://vllm/vllm-openai:$2"
}

case "$WHICH" in
    vllm019)
        GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
        if [ "$(printf '%s\n' 2.31 "$GLIBC" | sort -V | head -1)" = 2.31 ] && [ "${MODE:-venv}" = venv ]; then
            set +u; module purge; module load gcc/12.2.0; set -u
            [ -d "$VENV_019" ] || "${PYTHON_BIN:-python3}" -m venv "$VENV_019"
            "$VENV_019/bin/pip" install -U pip
            # torch from the cu128 index: CUDA 12.8 runs on the nodes' 12.x driver; transformers pinned
            # near vLLM 0.19.1's release (an unpinned install takes an incompatible 5.x).
            "$VENV_019/bin/pip" install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
            "$VENV_019/bin/pip" install vllm==0.19.1 transformers==5.6.2
        else
            pull "$SIF_019" v0.19.1
        fi ;;
    vllm032) pull "$SIF_032" v0.32.0-cu129 ;;
    *) echo "[!] usage: setup_runtime.sh vllm019|vllm032"; exit 1 ;;
esac

use_runtime "$WHICH"
"${RUN[@]}" python3 -c "
import torch, transformers, vllm
print(f'[~] torch {torch.__version__} (CUDA {torch.version.cuda}) | vllm {vllm.__version__} | transformers {transformers.__version__}')
assert not (torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13), 'CUDA 13 runtime: does not run on the A100 driver'
"
"${RUN[@]}" vllm serve --help > /dev/null || { echo "[!] 'vllm serve' does not even import: see above."; exit 1; }
echo "[+] $WHICH runtime ready."
