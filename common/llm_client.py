"""Client for the model served by vLLM (OpenAI-compatible API), shared by both pipelines.

Everything comes from environment variables set by common/cluster/models.sh:
  LOCAL_TEXT_SERVER_URL   e.g. http://localhost:20001/v1 (required)
  LOCAL_MODEL_PATH        the model directory, which is also the model name the server expects
  LOCAL_TEMPERATURE, LOCAL_TOP_P, LOCAL_TOP_K, LOCAL_PRESENCE_PENALTY, LOCAL_REPETITION_PENALTY,
  LOCAL_FREQUENCY_PENALTY, LOCAL_MAX_MODEL_LEN, LOCAL_MAX_NEW_TOKENS, LOCAL_SEED   sampling
  LOCAL_ENABLE_THINKING   1 / 0 for hybrid thinking models, unset = chat template default
  LOCAL_EXTRA_BODY        JSON merged into every request (e.g. Mistral's reasoning_effort)
  LOCAL_GUIDED_JSON       0 disables guided JSON decoding (default on)
  LOCAL_RUN_TAG           label appended to the model name saved in the outputs
  LLM_CALL_LOG_DIR        if set, one JSON line per call is appended there (energy accounting);
                          every LLM_CALL_<NAME> variable is saved in it as <name>
"""
import base64
import json
import os
import re
import socket
import time

import requests

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.environ.get("WORK", "."), "models"))
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", os.path.join(MODELS_DIR, "qwen2.5-14b"))
RUN_TAG = os.environ.get("LOCAL_RUN_TAG", "").strip()
MODEL_NAME = f"{os.path.basename(MODEL_PATH.rstrip('/'))}{f' [{RUN_TAG}]' if RUN_TAG else ''} (local/vLLM)"
TEXT_SERVER_URL = os.environ.get("LOCAL_TEXT_SERVER_URL", "").rstrip("/")

TEMPERATURE = float(os.environ.get("LOCAL_TEMPERATURE", "0.2"))
MAX_MODEL_LEN = int(os.environ.get("LOCAL_MAX_MODEL_LEN", "32768"))
MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", "4096"))
SEED = int(os.environ.get("LOCAL_SEED", "0"))
GUIDED_JSON = os.environ.get("LOCAL_GUIDED_JSON", "1") != "0"
_OPTIONAL_SAMPLING = {
    name: float(os.environ[var]) if name != "top_k" else int(os.environ[var])
    for name, var in [("top_p", "LOCAL_TOP_P"), ("top_k", "LOCAL_TOP_K"), ("presence_penalty", "LOCAL_PRESENCE_PENALTY")]
    if os.environ.get(var, "").strip()
}
ENABLE_THINKING = {"1": True, "0": False}.get(os.environ.get("LOCAL_ENABLE_THINKING", "").strip())
EXTRA_BODY = json.loads(os.environ.get("LOCAL_EXTRA_BODY", "") or "{}")

# Saved with every output, so each answer can be traced back to its settings.
GENERATION_SETTINGS = {
    "temperature": TEMPERATURE,
    "max_model_len": MAX_MODEL_LEN,
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "guided_json": GUIDED_JSON,
    "repetition_penalty": float(os.environ.get("LOCAL_REPETITION_PENALTY", "1.15")),
    "frequency_penalty": float(os.environ.get("LOCAL_FREQUENCY_PENALTY", "0.3")),
    **_OPTIONAL_SAMPLING,
    **({"text_server": True} if TEXT_SERVER_URL else {}),
    **({"enable_thinking": ENABLE_THINKING} if ENABLE_THINKING is not None else {}),
    **({"run_tag": RUN_TAG} if RUN_TAG else {}),
    **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
}

USER_PREFIX = "--- PAPER CONTENT ---\n"
_TEMPLATE_MARGIN = 256   # room for the chat template's special tokens and role headers
_HEAD_SHARE = 0.7        # share of a truncated paper kept from its start; the rest from its end

# ---------------------------------------------------------------- call log (energy accounting)
CALL_LOG_DIR = os.environ.get("LLM_CALL_LOG_DIR", "").strip()
CALL_LABELS = {k[len("LLM_CALL_"):].lower(): v for k, v in os.environ.items()
               if k.startswith("LLM_CALL_") and k != "LLM_CALL_LOG_DIR" and v.strip()}
_call_context = {}


def set_call_context(**context) -> None:
    """What the next calls are about (paper, question...), saved in the call log."""
    _call_context.clear()
    _call_context.update({k: v for k, v in context.items() if v not in (None, "")})


def _log_call(record: dict) -> None:
    if not CALL_LOG_DIR:
        return
    try:
        os.makedirs(CALL_LOG_DIR, exist_ok=True)
        path = os.path.join(CALL_LOG_DIR, f"calls_{socket.gethostname()}_{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[!] Call log not written ({e}).")   # accounting must never cost an answer


# ---------------------------------------------------------------- context window
class _ServerTokenizer:
    """The served model's own tokenizer, through the server's /tokenize and /detokenize."""

    def _post(self, endpoint: str, payload: dict) -> dict:
        base = TEXT_SERVER_URL[:-3] if TEXT_SERVER_URL.endswith("/v1") else TEXT_SERVER_URL
        response = requests.post(f"{base}/{endpoint}", json={"model": MODEL_PATH, **payload}, timeout=120)
        response.raise_for_status()
        return response.json()

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        return self._post("tokenize", {"prompt": text, "add_special_tokens": add_special_tokens})["tokens"]

    def decode(self, ids: list) -> str:
        return self._post("detokenize", {"tokens": list(ids)})["prompt"]


def fit_to_context(text: str, system_prompt: str, reserve_tokens: int = 0, tokenizer=None) -> tuple[str, bool]:
    """
    The paper text that fits the context window, and whether it was cut. A text that is too long
    keeps its start and its end (conclusions, limitations) and loses the middle, with a note.
    reserve_tokens: room for what is sent besides the text (images, instructions).
    """
    tokenizer = tokenizer or _ServerTokenizer()

    def count(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    budget = MAX_MODEL_LEN - MAX_NEW_TOKENS - count(system_prompt) - count(USER_PREFIX) - _TEMPLATE_MARGIN - reserve_tokens
    if count(text) <= budget:
        return text, False
    ids = tokenizer.encode(text, add_special_tokens=False)
    body_budget = budget - 64   # 64 tokens for the omission note
    for _ in range(3):
        head_n = int(body_budget * _HEAD_SHARE)
        tail_n = body_budget - head_n
        head = tokenizer.decode(ids[:head_n])
        tail = tokenizer.decode(ids[-tail_n:]) if tail_n > 0 else ""
        omitted = max(0, len(text) - len(head) - len(tail))
        note = f"\n\n[... {omitted} characters of the paper omitted here to fit the model's context window ...]\n\n"
        fitted = head + note + tail.rstrip()
        excess = count(fitted) - budget
        if excess <= 0:
            break
        body_budget -= excess + 32   # decoding and re-encoding can shift token boundaries
    print(f"[!] Paper text too long for the {MAX_MODEL_LEN}-token context: kept the start and end ({omitted} characters omitted).")
    return fitted, True


# ---------------------------------------------------------------- requests
def _sampling_kwargs() -> dict:
    return dict(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS,
                repetition_penalty=GENERATION_SETTINGS["repetition_penalty"],
                frequency_penalty=GENERATION_SETTINGS["frequency_penalty"], seed=SEED, **_OPTIONAL_SAMPLING)


def _chat(messages: list, schema: dict = None) -> str:
    """One chat completion; returns the answer text (a thinking model's reasoning comes back apart)."""
    payload = {"model": MODEL_PATH, "messages": messages, **_sampling_kwargs()}
    if ENABLE_THINKING is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
    payload.update(EXTRA_BODY)
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    n_images = sum(1 for m in messages if isinstance(m.get("content"), list)
                   for part in m["content"] if part.get("type") == "image_url")
    record = {"t_start": time.time(), "model": MODEL_NAME, "run_tag": RUN_TAG, "seed": SEED, **CALL_LABELS,
              **_call_context, "n_images": n_images, "guided_json": schema is not None, "status": "error"}
    try:
        # A 128k-token prompt plus a long thinking phase can take several minutes.
        response = requests.post(f"{TEXT_SERVER_URL}/chat/completions", json=payload, timeout=1800)
        record["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        msg = body["choices"][0]["message"]
        record.update({"status": "ok", "finish_reason": body["choices"][0].get("finish_reason"),
                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                       "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
                       "answer_chars": len(msg.get("content") or "")})
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"[:300]
        raise
    finally:
        record["t_end"] = time.time()
        record["latency_s"] = round(record["t_end"] - record["t_start"], 3)
        _log_call(record)
    choice = body["choices"][0]
    reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_content") or ""
    if reasoning:
        print(f"[~] Model reasoning: {len(reasoning)} characters (not part of the answer).")
    tokens = (body.get("usage") or {}).get("completion_tokens")
    print(f"[~] Generation ended: finish_reason={choice.get('finish_reason')}, {tokens} completion token(s) (max {MAX_NEW_TOKENS}).")
    if choice.get("finish_reason") == "length":
        print("[!] Answer cut by max_tokens: it is likely truncated.")
    return choice["message"].get("content") or ""


def chat(messages: list, schema: dict = None) -> str:
    """A chat call with the pipeline's sampling settings. With a schema, guided JSON decoding;
    if that request fails it is retried once without the constraint."""
    if not TEXT_SERVER_URL:
        raise RuntimeError("LOCAL_TEXT_SERVER_URL is not set: start the vLLM server first (common/cluster/models.sh).")
    if schema is not None:
        try:
            return _chat(messages, schema)
        except Exception as e:
            print(f"[!] Guided JSON generation failed ({type(e).__name__}: {e}); retrying without the schema constraint.")
    return _chat(messages)


def image_part(path: str) -> dict:
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(path, "rb") as f:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"}}


# ---------------------------------------------------------------- JSON answers
def strip_json_fences(text: str) -> str:
    """Removes ``` / ```json fences around an answer."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def extract_json_object(text: str) -> str:
    """If the text is not valid JSON as it is, the span from the first '{' to the last '}'."""
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    return text if start == -1 or end <= start else text[start:end + 1]


_JSON_BACKSLASH_RE = re.compile(r'\\(["\\/]|u[0-9a-fA-F]{4})|\\')


def repair_json_backslashes(text: str) -> str:
    """LaTeX written inside JSON strings without escaping (\\frac, \\tilde): doubles every lone
    backslash, only when the text does not parse as it is."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return _JSON_BACKSLASH_RE.sub(lambda m: m.group(0) if m.group(1) else r'\\', text)


def clean_json_answer(raw: str) -> str:
    return repair_json_backslashes(extract_json_object(strip_json_fences(raw)))
