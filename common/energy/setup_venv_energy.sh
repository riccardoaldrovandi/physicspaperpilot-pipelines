#!/bin/bash
# One-time setup of the small environment of the energy monitor (common/energy/gpu_monitor.py):
# CodeCarbon + nvidia-ml-py only, kept apart from $WORK/venv_vllm so its pinned packages
# (numpy 1.26.4, ...) are never touched. On Leonardo run it on a LOGIN node (compute nodes have
# no internet), from the repository root:
#   bash common/energy/setup_venv_energy.sh
set -e
VENV="${ENERGY_VENV:-${WORK:-.}/venv_energy}"
# The Python 3.11 that $WORK/venv_vllm was built from (the system python3 is too old for CodeCarbon 3).
BASE="$("${WORK:-.}/venv_vllm/bin/python" -c 'import sys, os; print(os.path.join(sys.base_prefix, "bin", "python3"))' 2>/dev/null || echo python3)"
echo "[~] Base interpreter: $BASE ($("$BASE" --version))"
"$BASE" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "codecarbon==3.3.3" nvidia-ml-py
"$VENV/bin/python" -c "import codecarbon, pynvml; print('[+] codecarbon', codecarbon.__version__, '+ nvidia-ml-py OK in $VENV')"
