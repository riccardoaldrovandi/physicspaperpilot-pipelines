"""Pipeline B, step B2: the model answers the benchmark questions reading only a JSON from step B1.

One call per question: the package's prompt 03_pilot_run/prompt_json_only.md with the paper's JSON
and the question object; the answer is decoded against the format that prompt asks for (answer,
mcq_choice, evidence_fields, cited_evidence, abstained). With the JSONs of another model it is the
fixed-reader experiment (condition C). Output: <output>.jsonl (raw, resumable), <output>.csv
(package format) and <output>.evidence_fields.csv. Papers without a JSON are skipped.
  python pipeline_b/answer_from_json.py --benchmark-dir benchmark/PhysicsPaperPilotEval_v0.3.1 \
      --json-input runs/pdf/qwen38_nothink_s0_B1.jsonl --output runs/pdf/qwen38_nothink_s0_B2.jsonl \
      --label "qwen38|nothink|s0|B_json_same_model"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import benchmark_io as bio  # noqa: E402
from common import llm_client as llm  # noqa: E402
from common.paper_inputs import arxiv_id_of  # noqa: E402

PROMPT_FILE = os.path.join("03_pilot_run", "prompt_json_only.md")
# Guided decoding: the structure the prompt asks for, with one answer.
ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["paper_id", "answers"],
    "properties": {
        "paper_id": {"type": "string"},
        "answers": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["question_id", "answer", "mcq_choice", "evidence_fields", "cited_evidence", "abstained"],
            "properties": {
                "question_id": {"type": "string"},
                "answer": {"type": "string"},
                "mcq_choice": {"type": "string", "enum": ["A", "B", "C", "D", ""]},
                "evidence_fields": {"type": "array", "items": {"type": "string"}},
                "cited_evidence": {"type": "array", "items": {"type": "string"}},
                "abstained": {"type": "boolean"},
            }}},
    },
}


def question_object(q: dict) -> str:
    """The public row of the question (options parsed), as the prompt asks."""
    obj = {k: v for k, v in q.items() if v not in ("", None)}
    if q.get("options"):
        obj["options"] = json.loads(q["options"])
    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_jsons(path: str) -> dict:
    """arXiv id -> the JSON of a B1 file (the assistant message, unchanged)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        aid = arxiv_id_of(rec.get("source_file", ""))
        answer = next((m["content"] for m in reversed(rec.get("messages", [])) if m.get("role") == "assistant"), None)
        if aid and answer:
            out[aid] = {"json": answer, "provider": rec.get("provider")}
    return out


def answer_paper(benchmark_dir, questions, paper_json, raw_path, label, json_file) -> int:
    prompt = bio.read_prompt(benchmark_dir, PROMPT_FILE)
    done = bio.done_questions(raw_path)
    n = 0
    for q in questions:
        if q["question_id"] in done:
            continue
        content = (prompt.replace("[INSERT PAPER ID]", q["arxiv_id"])
                         .replace("[INSERT THE COMPLETE STRUCTURED JSON]", paper_json["json"])
                         .replace("[INSERT QUESTION OBJECT FROM questions_public.csv]", question_object(q)))
        record = bio.base_record("json", label, q, llm.MODEL_NAME, llm.GENERATION_SETTINGS)
        record.update({"json_file": os.path.basename(json_file), "json_provider": paper_json["provider"],
                       "json_sha256": bio.sha256_of(paper_json["json"])})
        print(f"[~] {q['question_id']} ({q['question_type']})...")
        llm.set_call_context(paper=q["arxiv_id"], question_id=q["question_id"], question_type=q["question_type"],
                             json_file=os.path.basename(json_file))
        try:
            raw = llm.chat([{"role": "user", "content": content}], schema=ANSWER_SCHEMA)
            record["raw_output"] = raw
            item = (json.loads(llm.extract_json_object(llm.strip_json_fences(raw))).get("answers") or [{}])[0]
            record.update({
                "answer": str(item.get("answer", "")).strip(),
                "mcq_choice": str(item.get("mcq_choice", "")).strip().upper(),
                "evidence_fields": [str(x) for x in item.get("evidence_fields") or []],
                "cited_evidence": "; ".join(str(x).strip() for x in item.get("cited_evidence") or [] if str(x).strip()),
                "abstained": bool(item.get("abstained", False)),
            })
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            print(f"[!] {q['question_id']}: {record['error']}")
        bio.append_record(raw_path, record)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark-dir", required=True, help="PhysicsPaperPilot package")
    ap.add_argument("--json-input", required=True, help="B1 JSONL of the model that wrote the summaries")
    ap.add_argument("--output", required=True, help="Raw answers JSONL; the predictions CSV is written next to it")
    ap.add_argument("--label", required=True, help="Value of the 'model' column of the predictions CSV")
    ap.add_argument("--papers", default=None, help="Only these arXiv ids (comma-separated)")
    args = ap.parse_args()

    questions = bio.load_questions(args.benchmark_dir)
    papers = sorted({q["arxiv_id"] for q in questions})
    if args.papers:
        papers = [p for p in papers if p in args.papers.split(",")]
    jsons = load_jsons(args.json_input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for i, aid in enumerate(papers, 1):
        qs = [q for q in questions if q["arxiv_id"] == aid]
        print(f"\n========== Paper {i}/{len(papers)}: {aid} ({len(qs)} questions) ==========")
        if aid not in jsons:
            print(f"[!] No JSON for {aid} in {args.json_input}: skipped (its questions stay unanswered).")
            continue
        answer_paper(args.benchmark_dir, qs, jsons[aid], args.output, args.label, args.json_input)
    if os.path.exists(args.output):
        n = bio.write_predictions(args.benchmark_dir, args.output, os.path.splitext(args.output)[0] + ".csv")
        print(f"[+] {n} answer(s) in {os.path.splitext(args.output)[0]}.csv")


if __name__ == "__main__":
    main()
