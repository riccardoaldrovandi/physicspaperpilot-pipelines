# vLLM runtime and server helpers for SLURM jobs on Leonardo (CINECA). Sourced, not executed.
#
# Two runtimes, both outside the pipeline's own Python environment:
#   vllm019  vLLM 0.19.1 (Qwen3.8, Gemma 4): a venv ($WORK/venv_vllm_qwen38) or the Apptainer image
#            $WORK/containers/vllm-openai_v0.19.1.sif
#   vllm032  vLLM 0.32 (Nemotron 3.5, Mistral Medium 3.5): the image vllm-openai_v0.32.0-cu129.sif
#            (the -cu129 build runs on the A100 nodes' CUDA 12 driver)
# One-time setup: common/cluster/setup_runtime.sh.
#   use_runtime vllm019|vllm032    sets RUN=(...), the prefix that runs a command inside the runtime
#   cuda_check                     fails in seconds if the runtime cannot use the GPUs
#   wait_for_server / stop_server  around a server started in the background (SERVER_PID, PORT)

WORK_DIR="${WORK:?WORK is not set (it is on Leonardo)}"
MODELS_DIR="${MODELS_DIR:-$WORK_DIR/models}"
VENV_019="${VENV_019:-$WORK_DIR/venv_vllm_qwen38}"
SIF_019="${SIF_019:-$WORK_DIR/containers/vllm-openai_v0.19.1.sif}"
SIF_032="${SIF_032:-$WORK_DIR/containers/vllm-openai_v0.32.0-cu129.sif}"
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"

# Compile caches off $HOME (small quota); the PyTorch sampler needs no nvcc.
export VLLM_CACHE_ROOT="$WORK_DIR/.cache/vllm" TRITON_CACHE_DIR="$WORK_DIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/.cache/torchinductor" VLLM_USE_FLASHINFER_SAMPLER=0

use_runtime() {
    case "$1" in
        vllm019)
            if [ -x "$VENV_019/bin/vllm" ]; then RUN=(env "PATH=$VENV_019/bin:$PATH" "VIRTUAL_ENV=$VENV_019")
            elif [ -f "$SIF_019" ]; then RUN=("$APPTAINER_BIN" exec --nv --bind /leonardo_work "$SIF_019")
            else echo "[!] No vLLM 0.19.1 runtime: run common/cluster/setup_runtime.sh vllm019 on a login node." >&2; return 1
            fi ;;
        vllm032)
            [ -f "$SIF_032" ] || { echo "[!] No vLLM 0.32 image: run common/cluster/setup_runtime.sh vllm032." >&2; return 1; }
            export VLLM_NO_USAGE_STATS=1   # no internet on compute nodes
            # The -cu129 image ships a torchcodec built for CUDA 13, which crashes `vllm serve` at import
            # (it only decodes audio/video, never sent here): a stub that raises ImportError is bound over it.
            local stubs="$WORK_DIR/.cache/vllm32_stubs" bind
            if [ ! -f "$stubs/torchcodec/__init__.py" ]; then
                mkdir -p "$stubs/torchcodec"
                echo 'raise ImportError("torchcodec disabled: CUDA 13 build in the -cu129 image")' > "$stubs/torchcodec/__init__.py"
            fi
            bind="$stubs/torchcodec:/usr/local/lib/python3.12/dist-packages/torchcodec"
            case ",${APPTAINER_BIND:-}," in *",$bind,"*) ;; *) export APPTAINER_BIND="$bind${APPTAINER_BIND:+,$APPTAINER_BIND}" ;; esac
            case ",${SINGULARITY_BIND:-}," in *",$bind,"*) ;; *) export SINGULARITY_BIND="$bind${SINGULARITY_BIND:+,$SINGULARITY_BIND}" ;; esac
            RUN=("$APPTAINER_BIN" exec --nv --bind /leonardo_work "$SIF_032") ;;
        *) echo "[!] Unknown runtime '$1'" >&2; return 1 ;;
    esac
}

cuda_check() {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    "${RUN[@]}" python3 -c "
import torch
n = torch.cuda.device_count()
print(f'[~] torch {torch.__version__} CUDA {torch.version.cuda}, visible GPUs: {n}')
assert n >= 2, 'need at least 2 GPUs'
for i in range(n): torch.ones(1, device=f'cuda:{i}')
print('[+] CUDA works on all GPUs')
" || { echo "[!] CUDA check failed: nothing else was started."; return 1; }
}

# wait_for_server [max seconds]: polls /v1/models every 5 s; gives up at once if the server died.
wait_for_server() {
    local steps=$(( ${1:-900} / 5 ))
    for i in $(seq 1 "$steps"); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "[+] Server ready after $((i * 5))s."
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[!] Server process exited during startup: see $SERVER_LOG"; tail -40 "$SERVER_LOG"; return 1; }
        sleep 5
    done
    echo "[!] Server not ready after $((steps * 5))s: see $SERVER_LOG"
    tail -40 "$SERVER_LOG"
    return 1
}

# Kills the server's whole process group (it is started with setsid), tensor-parallel workers included.
stop_server() {
    [ -z "${SERVER_PID:-}" ] && return 0
    echo "[~] Stopping the vLLM server (process group $SERVER_PID)..."
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || pkill -TERM -P "$SERVER_PID" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}
