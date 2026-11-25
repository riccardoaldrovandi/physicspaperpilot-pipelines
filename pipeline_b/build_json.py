"""Pipeline B, step B1: one structured JSON summary per paper (prompt v2.1).

One call per paper: system prompt = prompt v2.1 + the instructions for attached images; user
message = the paper's figures as images, then its text (same inputs as Pipeline A). The answer is
decoded against the v2.1 JSON schema, cleaned (placeholder values normalized) and appended to
--output as one JSON line per paper (the messages hold exactly what the model saw and answered).
Answers that are not valid JSON go to <output>.rejected.jsonl. Rerunning skips the papers already
in --output and retries the rejected ones (EXPORT_SKIP_REJECTED=1 keeps them rejected).
  python pipeline_b/build_json.py --output runs/pdf/qwen38_nothink_s0_B1.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import llm_client as llm  # noqa: E402
from common.paper_inputs import IMAGE_TOKENS, find_sources, paper_inputs  # noqa: E402
from pipeline_b.prompt import ANSWER_SCHEMA, PAPER_FIELDS, SINGLE_CALL_INSTRUCTIONS, SYSTEM_PROMPT, is_empty_value  # noqa: E402

PROMPT_VERSION = "v2.1"


def _without_placeholders(value):
    """Placeholder items dropped from lists, placeholder strings replaced by 'Not specified'."""
    if isinstance(value, dict):
        return {k: _without_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [v for v in (_without_placeholders(x) for x in value) if not is_empty_value(v)]
    if isinstance(value, str) and value.strip() and is_empty_value(value):
        return "Not specified"
    return value


_FINDINGS_KEY_RE = re.compile(r',\s*"visual[ _]?analysis[ _]?findings"\s*:', re.IGNORECASE)


def _salvage(text: str):
    """An answer whose only invalid part is the last field (visual_analysis_findings, always emptied
    anyway): cut before it and close the object. None if the rest is not a complete answer."""
    match = _FINDINGS_KEY_RE.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(llm.repair_json_backslashes(text[:match.start()] + "\n}"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and all(k in parsed for k in PAPER_FIELDS) else None


def finalize(raw: str):
    """The cleaned JSON text, or None if the answer is unusable."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if not isinstance(parsed, dict):
        parsed = _salvage(raw)
        if parsed is None:
            return None
        print("[~] Salvaged an answer whose only invalid JSON was inside visual_analysis_findings.")
    elif not parsed:
        return None
    for key in PAPER_FIELDS:
        if key in parsed:
            parsed[key] = _without_placeholders(parsed[key])
    parsed["visual_analysis_findings"] = []   # reserved for external curve fits, none here
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def jsonl_line(obj) -> str:
    """One JSONL line; a lone UTF-16 surrogate a model may emit cannot be written in UTF-8: replaced by '?'."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8") + "\n"


def sources_in(path: str) -> set:
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["source_file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def code_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="pdf", choices=["pdf", "latex"])
    ap.add_argument("--pdf-dir", default="papers/pdf")
    ap.add_argument("--latex-dir", default="papers/latex")
    ap.add_argument("--figures-dir", default="papers/latex_figures", help="LaTeX: figures converted to PNG by common/fetch_latex_sources.py")
    ap.add_argument("--cache-dir", default="cache", help="Rendered PDF pages")
    ap.add_argument("--output", required=True, help="JSONL, appended to (resumable)")
    ap.add_argument("--papers", default=None, help="Only these arXiv ids (comma-separated)")
    args = ap.parse_args()

    sources = find_sources(args.source, args.pdf_dir, args.latex_dir)
    if args.papers:
        sources = {a: p for a, p in sources.items() if a in args.papers.split(",")}
    rejected_path = os.path.splitext(args.output)[0] + ".rejected.jsonl"
    done = sources_in(args.output)
    if os.environ.get("EXPORT_SKIP_REJECTED") == "1":
        done |= sources_in(rejected_path)
    pending = [p for _, p in sorted(sources.items()) if p not in done]
    print(f"[+] {len(sources)} paper(s), {len(sources) - len(pending)} already done, {len(pending)} pending.")

    system_prompt = SYSTEM_PROMPT + SINGLE_CALL_INSTRUCTIONS
    metadata = {"prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
                "code_version": code_version(), "mode": "single", "source": args.source,
                "figures_dir": args.figures_dir if args.source == "latex" else None}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    for i, path in enumerate(pending, 1):
        print(f"\n========== Paper {i}/{len(pending)}: {os.path.basename(path)} ==========")
        try:
            text, images = paper_inputs(path, args.source, args.figures_dir, args.cache_dir)
        except Exception as e:
            print(f"[!] Skipping {path}: {type(e).__name__}: {e}")
            continue
        llm.set_call_context(paper=os.path.basename(path), source=args.source)
        model_input = text
        try:
            model_input, truncated = llm.fit_to_context(text, system_prompt=system_prompt, reserve_tokens=len(images) * IMAGE_TOKENS)
            user = [llm.image_part(p) for p in images] + [{"type": "text", "text": f"{llm.USER_PREFIX}{model_input}"}]
            raw = llm.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                           schema=ANSWER_SCHEMA if llm.GUIDED_JSON else None)
            raw = llm.clean_json_answer(raw)
        except Exception as e:
            print(f"[!] Generation failed: {type(e).__name__}: {e}")
            with open(rejected_path, "a", encoding="utf-8") as f:
                f.write(jsonl_line({"source_file": path, "provider": llm.MODEL_NAME, "input_chars": len(model_input),
                                    "raw_output": None, "error": f"{type(e).__name__}: {e}"}))
            continue
        answer = finalize(raw)
        if answer is None:
            print(f"[!] Invalid JSON: raw answer saved to {rejected_path}")
            with open(rejected_path, "a", encoding="utf-8") as f:
                f.write(jsonl_line({"source_file": path, "provider": llm.MODEL_NAME, "input_chars": len(model_input),
                                    "raw_output": raw, "error": None}))
            continue
        record = {"source_file": path, "provider": llm.MODEL_NAME, "input_truncated": truncated,
                  "images": [os.path.basename(p) for p in images],
                  "generation": dict(metadata, created_at=datetime.now().isoformat(timespec="seconds"), **llm.GENERATION_SETTINGS),
                  "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": model_input},
                               {"role": "assistant", "content": answer}]}
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(jsonl_line(record))
        written += 1
        print(f"[+] JSON written ({len(model_input)} characters of input{', truncated' if truncated else ''}).")
    print(f"\n[+] Done: {written} JSON(s) appended to {args.output}")


if __name__ == "__main__":
    main()
