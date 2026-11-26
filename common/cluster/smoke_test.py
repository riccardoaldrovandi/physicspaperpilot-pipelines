"""Smoke test of a served model before any benchmark run: is it worth a full job?

With the server up and the environment of `model_env <config>` (common/cluster/models.sh), it checks:
  1. text        a one-word answer ("capital of France?"): broken FP8 kernels give garbage here
  2. image       a page of a benchmark paper is read (the answer names something on the page)
  3. json        guided decoding returns valid JSON for a small schema
  4. reasoning   with the config's reasoning switch on, a reasoning field comes back
  5. speed       a pipeline-like request (paper text, 8 images, the v2.1 schema, the config's sampling),
                 streamed: tokens per second per 250 tokens. Fails if decoding collapses as the answer
                 grows (last window below half the first, or below 5 tok/s), as Mistral think did.
Writes a JSON report and exits 1 if a check fails.
  python common/cluster/smoke_test.py --out logs/smoke_qwen38_nothink.json
"""
import argparse
import glob
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import llm_client as llm  # noqa: E402
from common.paper_inputs import paper_inputs  # noqa: E402
from pipeline_b.prompt import ANSWER_SCHEMA  # noqa: E402


def body(messages, max_tokens, schema=None, stream=False):
    b = {"model": llm.MODEL_PATH, "messages": messages, **llm._sampling_kwargs(), "max_tokens": max_tokens}
    if llm.ENABLE_THINKING is not None:
        b["chat_template_kwargs"] = {"enable_thinking": llm.ENABLE_THINKING}
    b.update(llm.EXTRA_BODY)
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    return b


def ask(messages, max_tokens=512, schema=None):
    r = requests.post(f"{llm.TEXT_SERVER_URL}/chat/completions", json=body(messages, max_tokens, schema), timeout=900)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return msg.get("content") or "", msg.get("reasoning") or msg.get("reasoning_content") or ""


def speed(messages, max_tokens, time_limit):
    """Streamed request: generated tokens, time to first token and tok/s per window of 250 tokens."""
    start, stamps = time.time(), []
    with requests.post(f"{llm.TEXT_SERVER_URL}/chat/completions", json=body(messages, max_tokens, ANSWER_SCHEMA, True),
                       stream=True, timeout=time_limit + 300) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[5:]).get("choices", []):
                d = choice.get("delta", {})
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    stamps.append(time.time())
            if time.time() - start > time_limit:
                break
    windows = [round(250 / (stamps[i - 1] - stamps[i - 250]), 1) for i in range(500, len(stamps) + 1, 250)
               if stamps[i - 1] > stamps[i - 250]]
    return {"generated": len(stamps), "ttft_s": round(stamps[0] - start, 1) if stamps else None,
            "total_s": round(time.time() - start, 1), "tok_per_s_per_250": windows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", default="papers/pdf/2402.11575*.pdf", help="PDF used for the image and speed checks")
    ap.add_argument("--speed-tokens", type=int, default=3000)
    ap.add_argument("--speed-time-limit", type=int, default=900, help="seconds")
    args = ap.parse_args()
    pdf = sorted(glob.glob(args.paper))[0]
    text, images = paper_inputs(pdf, "pdf", cache_dir="cache/smoke")
    template_kwargs = llm.EXTRA_BODY.get("chat_template_kwargs", {})
    reasoning_on = (llm.ENABLE_THINKING is True or llm.EXTRA_BODY.get("reasoning_effort") == "high"
                    or template_kwargs.get("thinking") is True or template_kwargs.get("enable_thinking") is True)
    report, ok = {"model": llm.MODEL_NAME, "settings": llm.GENERATION_SETTINGS}, True

    def check(name, fn):
        nonlocal ok
        t = time.time()
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"{type(e).__name__}: {e}"[:300]
        ok &= passed
        report[name] = {"passed": passed, "detail": detail, "seconds": round(time.time() - t, 1)}
        print(f"[{'+' if passed else '!'}] {name}: {'PASS' if passed else 'FAIL'} - {str(detail)[:200]}", flush=True)

    def text_check():
        a, _ = ask([{"role": "user", "content": "What is the capital of France? Answer with one word."}], 2048)
        return "paris" in a.lower(), a.strip()[:100]

    def image_check():
        a, _ = ask([{"role": "user", "content": [llm.image_part(images[0]), {"type": "text", "text":
                    "This is a page of a physics paper. Name one figure, table or axis label you can read on it."}]}], 2048)
        return len(a.strip()) > 10, a.strip()[:200]

    def json_check():
        schema = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                  "required": ["city", "country"], "additionalProperties": False}
        a, _ = ask([{"role": "user", "content": "Give the capital of Italy as JSON with keys city and country."}], 2048, schema)
        parsed = json.loads(llm.clean_json_answer(a))
        return "rom" in str(parsed.get("city", "")).lower(), parsed

    def reasoning_check():
        a, r = ask([{"role": "user", "content": "A train covers 120 km in 1.5 hours. What is its average speed in km/h?"}], 4096)
        return bool(r.strip()) and "80" in a, {"answer": a.strip()[:100], "reasoning_chars": len(r)}

    def speed_check():
        fitted, _ = llm.fit_to_context(text, system_prompt="", reserve_tokens=8 * 2500 + args.speed_tokens)
        content = [llm.image_part(p) for p in images[:8]] + [{"type": "text", "text": llm.USER_PREFIX + fitted +
                   "\n\nWrite the structured summary of this paper, as complete as possible."}]
        r = speed([{"role": "user", "content": content}], args.speed_tokens, args.speed_time_limit)
        w = r["tok_per_s_per_250"]
        passed = len(w) >= 2 and w[-1] >= 0.5 * w[0] and w[-1] >= 5
        return passed, r

    check("1_text", text_check)
    check("2_image", image_check)
    check("3_json", json_check)
    if reasoning_on:
        check("4_reasoning", reasoning_check)
    check("5_speed", speed_check)
    report["passed"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{'+' if ok else '!'}] Smoke test {'PASSED' if ok else 'FAILED'}: {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
