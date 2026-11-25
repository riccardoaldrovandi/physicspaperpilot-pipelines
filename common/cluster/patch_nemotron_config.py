"""Adds vLLM's name of the token embeddings to the quantization ignore list of
the Nemotron 3.5 FP8 checkpoint (RedHatAI/...-FP8), in its config.json.

Why: the checkpoint excludes the embeddings from FP8 under their HF name
("re:.*backbone\\.embeddings$"), and its weights really are BF16. vLLM 0.32 names
that layer "...embed_tokens", so the ignore rule does not match, vLLM asks
compressed-tensors for an FP8 method for the embedding and stops at startup:
"compressed-tensors embeddings only support weight-only INT group/channel
(WNA16) quantization" (job 63290012). The card tested vLLM 0.31, which did not
quantize embeddings yet. Nothing about the weights changes: only the name rule.

Idempotent; the original is kept once as config.json.orig. Every
"quantization_config" in the file is patched (top level and llm_config).
  python3 common/cluster/patch_nemotron_config.py $MODELS_DIR/nemotron-3.5-super-vl-120b-fp8 (done by download_model.sh)
"""
import json
import os
import shutil
import sys

RULE = "re:.*embed_tokens$"


def patch(node, changed):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "quantization_config" and isinstance(value, dict):
                ignore = value.setdefault("ignore", [])
                if RULE not in ignore:
                    ignore.append(RULE)
                    changed.append(key)
            patch(value, changed)
    elif isinstance(node, list):
        for value in node:
            patch(value, changed)


def main():
    model_dir = sys.argv[1]
    path = os.path.join(model_dir, "config.json")
    with open(path) as f:
        config = json.load(f)
    changed = []
    patch(config, changed)
    if not changed:
        print(f"[~] {path}: '{RULE}' already in every ignore list, nothing to do")
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"[~] Original saved as {backup}")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[+] '{RULE}' added to {len(changed)} quantization_config ignore list(s) in {path}")


if __name__ == "__main__":
    main()
