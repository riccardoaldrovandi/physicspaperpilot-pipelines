# The models of the benchmark: vLLM server settings and pipeline sampling, the same for both
# pipelines. Sourced from the repository root, with MODEL set
# (qwen38 | gemma4 | nemotron35 | mistral35 | llama4scout):
#   source common/cluster/models.sh || exit 1
#   model_start                 # vLLM server on the job's GPUs, 128k context (installs the stop trap)
#   model_env think|nothink     # environment read by common/llm_client.py
#
# Overrides: MODELS_DIR, <MODEL>_MODEL_DIR, MAX_IMAGES (images per request, 12 in the benchmark),
# GPU_MEM_UTIL, MAX_NUM_SEQS, <MODEL>_EXTRA_ARGS (extra `vllm serve` flags).
# MISTRAL35_EAGLE=1: Mistral Medium 3.5 with EAGLE speculative decoding (its own draft model), the
# condition for its "think" config: without it reasoning decoding slowed to ~2.5 tok/s on vLLM 0.32.

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"

QWEN38_MODEL_DIR="${QWEN38_MODEL_DIR:-$MODELS_DIR/qwen3.8-27b}"
GEMMA4_MODEL_DIR="${GEMMA4_MODEL_DIR:-$MODELS_DIR/gemma4-31b}"
NEMOTRON35_MODEL_DIR="${NEMOTRON35_MODEL_DIR:-$MODELS_DIR/nemotron-3.5-super-vl-120b-fp8}"
MISTRAL35_MODEL_DIR="${MISTRAL35_MODEL_DIR:-$MODELS_DIR/mistral-medium-3.5-128b}"
MISTRAL35_EAGLE_DIR="${MISTRAL35_EAGLE_DIR:-$MODELS_DIR/mistral-medium-3.5-128b-eagle}"
LLAMA4SCOUT_MODEL_DIR="${LLAMA4SCOUT_MODEL_DIR:-$MODELS_DIR/llama-4-scout-17b-16e-fp8}"
MAX_CONTEXT=131072

# Guided JSON (xgrammar) allows unlimited whitespace: Gemma 4 sometimes filled its whole budget with
# newlines after a valid JSON. Formatting only, so it is on for Gemma (and Mistral, same issue on
# Mistral Small 4). NOT for Nemotron: with it Nemotron wrote corrupted JSONs (34 of 90).
WS_GUARD='--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}'

case "$MODEL" in
    qwen38)     MODEL_DIR="$QWEN38_MODEL_DIR";     RUNTIME=vllm019 ;;
    gemma4)     MODEL_DIR="$GEMMA4_MODEL_DIR";     RUNTIME=vllm019; GEMMA4_EXTRA_ARGS="${GEMMA4_EXTRA_ARGS-$WS_GUARD}" ;;
    nemotron35) MODEL_DIR="$NEMOTRON35_MODEL_DIR"; RUNTIME=vllm032 ;;
    mistral35)  MODEL_DIR="$MISTRAL35_MODEL_DIR";  RUNTIME=vllm032 ;;
    llama4scout) MODEL_DIR="$LLAMA4SCOUT_MODEL_DIR"; RUNTIME=vllm032 ;;
    *) echo "[!] Unknown MODEL '$MODEL' (qwen38 | gemma4 | nemotron35 | mistral35 | llama4scout)"; return 1 ;;
esac

# The `vllm serve` flags of each model (the model directory is also the served model name).
server_args() {
    # JSON values without spaces: the flags are split on spaces in model_start.
    local mm="{\"image\":${MAX_IMAGES:-2},\"video\":0}" mm_image="{\"image\":${MAX_IMAGES:-2}}"
    case "$MODEL" in
        # 27B BF16 over 2 GPUs (tensor parallelism); --max-num-seqs 32: 256 made the sampler warm-up OOM.
        qwen38) echo "--tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85}
                      --max-num-seqs ${MAX_NUM_SEQS:-32} --reasoning-parser qwen3 --limit-mm-per-prompt $mm" ;;
        # 31B BF16: a 128k context needs 4 GPUs for the KV cache.
        gemma4) echo "--tensor-parallel-size ${GEMMA4_TP:-4} --dtype bfloat16 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85}
                      --max-num-seqs ${MAX_NUM_SEQS:-32} --reasoning-parser gemma4 --limit-mm-per-prompt $mm_image
                      ${GEMMA4_EXTRA_ARGS:-}" ;;
        # FP8 120B MoE over 4 GPUs; expert parallelism is required on A100 (FP8 Marlin MoE padding).
        nemotron35) echo "--trust-remote-code --tensor-parallel-size 4 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85}
                          --max-num-seqs ${MAX_NUM_SEQS:-32} --max-num-batched-tokens 16384 --mamba-ssm-cache-dtype float16
                          --enable-expert-parallel --reasoning-parser nemotron_v3 --chat-template-content-format string
                          --limit-mm-per-prompt $mm ${NEMOTRON35_EXTRA_ARGS:-}" ;;
        # FP8 128B dense, Mistral-native format, over 4 GPUs.
        mistral35) echo "--tokenizer-mode mistral --config-format mistral --load-format mistral --tensor-parallel-size 4
                         --gpu-memory-utilization ${GPU_MEM_UTIL:-0.80} --max-num-seqs ${MAX_NUM_SEQS:-32}
                         --max-num-batched-tokens 16384 --reasoning-parser mistral
                         --structured-outputs-config {\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}
                         --limit-mm-per-prompt $mm_image ${MISTRAL35_EXTRA_ARGS:-}"
                   # EAGLE draft settings from the vLLM recipe of the model.
                   [ "${MISTRAL35_EAGLE:-0}" = 1 ] && echo "--speculative-config {\"model\":\"$MISTRAL35_EAGLE_DIR\",\"num_speculative_tokens\":3,\"method\":\"eagle\",\"max_model_len\":\"65536\"}" ;;
        # Llama 4 Scout (109B MoE, 17B active), FP8 dynamic (RedHatAI): FP8 runs as W8A16 on A100.
        llama4scout) echo "--tensor-parallel-size 4 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} --max-num-seqs ${MAX_NUM_SEQS:-32}
                           --limit-mm-per-prompt $mm_image ${LLAMA4SCOUT_EXTRA_ARGS:-}" ;;
    esac
}

model_start() {
    use_runtime "$RUNTIME" || return 1
    cuda_check || return 1
    if [ "$RUNTIME" = vllm032 ]; then
        "${RUN[@]}" python3 -c "import vllm.multimodal.media, vllm.entrypoints.cli.main; print('[+] vllm serve imports fine')" \
            || { echo "[!] vLLM import check failed inside the container: nothing else was started."; return 1; }
    fi
    PORT="${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"
    mkdir -p logs
    SERVER_LOG="logs/${MODEL}_server_${SLURM_JOB_ID:-local}.log"
    local args
    args="$(server_args)"
    echo "[~] Starting $MODEL ($MODEL_DIR), port $PORT, log $SERVER_LOG"
    # setsid: own process group, so stop_server also kills the tensor-parallel workers.
    # $args is split on purpose (one flag per word; the JSON values contain no spaces).
    setsid "${RUN[@]}" vllm serve "$MODEL_DIR" --max-model-len "$MAX_CONTEXT" --port "$PORT" $args > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    trap stop_server EXIT
    # 130 GB of FP8 weights load much slower than the 50-60 GB of the others.
    wait_for_server "$([ "$RUNTIME" = vllm032 ] && echo 1800 || echo 900)"
}

# model_env <config>: what common/llm_client.py reads. 128k context, 32k output tokens (reasoning
# included), no repetition penalties; temperature and top_p/top_k from each model card.
model_env() {
    export LOCAL_MODEL_PATH="$MODEL_DIR" LOCAL_TEXT_SERVER_URL="http://localhost:$PORT/v1"
    export LOCAL_MAX_MODEL_LEN=$MAX_CONTEXT LOCAL_MAX_NEW_TOKENS=32768 \
           LOCAL_REPETITION_PENALTY=1.0 LOCAL_FREQUENCY_PENALTY=0 LOCAL_PRESENCE_PENALTY=0
    unset LOCAL_ENABLE_THINKING LOCAL_TOP_P LOCAL_TOP_K LOCAL_EXTRA_BODY
    case "$MODEL:$1" in
        qwen38:think)       export LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=20 ;;
        qwen38:nothink)     export LOCAL_ENABLE_THINKING=0 LOCAL_TEMPERATURE=0.7 LOCAL_TOP_P=0.8 LOCAL_TOP_K=20 ;;
        # Gemma 4 needs special tokens kept for the server to split its thought channel from the answer.
        gemma4:think)       export LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=64 \
                                   LOCAL_EXTRA_BODY='{"skip_special_tokens": false}' ;;
        gemma4:nothink)     export LOCAL_ENABLE_THINKING=0 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=64 \
                                   LOCAL_EXTRA_BODY='{"skip_special_tokens": false}' ;;
        nemotron35:think)   export LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 ;;
        nemotron35:nothink) export LOCAL_ENABLE_THINKING=0 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 ;;
        # Mistral switches reasoning per request (reasoning_effort); "think" only with MISTRAL35_EAGLE=1.
        mistral35:nothink)  export LOCAL_EXTRA_BODY='{"reasoning_effort": "none"}' LOCAL_TEMPERATURE=0.3 LOCAL_TOP_P=1.0 ;;
        mistral35:think)    export LOCAL_EXTRA_BODY='{"reasoning_effort": "high"}' LOCAL_TEMPERATURE=0.7 LOCAL_TOP_P=0.95 ;;
        # Llama 4 has no reasoning mode; sampling of its generation_config.json (checked by download_model.sh).
        llama4scout:nothink) export LOCAL_TEMPERATURE=0.6 LOCAL_TOP_P=0.9 ;;
        *) echo "[!] Unknown config '$1' for $MODEL"; return 1 ;;
    esac
}
