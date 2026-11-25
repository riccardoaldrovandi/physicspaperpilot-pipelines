"""Pipeline A: the model answers the benchmark questions reading the paper itself.

One call per question: the paper (text + figure images, from the PDF or the LaTeX source) followed
by the package's prompt 03_pilot_run/prompt_for_tested_models.md with the question. The paper part
is the same for all the questions of a paper. Output: <output>.jsonl (raw, resumable) and
<output>.csv (the package's predictions format). Run from the repository root, with the vLLM
server up (see run_pipeline_a.sbatch):
  python pipeline_a/answer_from_paper.py --benchmark-dir benchmark/PhysicsPaperPilotEval_v0.3.1 \
      --output runs/pdf/qwen38_nothink_s0_A.jsonl --label "qwen38|nothink|s0|A_direct_pdf"
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import benchmark_io as bio  # noqa: E402
from common import llm_client as llm  # noqa: E402
from common.paper_inputs import IMAGE_TOKENS, find_sources, paper_inputs  # noqa: E402

PROMPT_FILE = os.path.join("03_pilot_run", "prompt_for_tested_models.md")


def question_text(q: dict) -> str:
    """[INSERT QUESTION] of the prompt: the question, plus its options for a multiple-choice one."""
    text = q["question"]
    options = json.loads(q["options"]) if q.get("options") else []
    if options:
        text += "\nOptions:\n" + "\n".join(f"{o['label']}) {o['text']}" for o in options)
    return text


def parse_answer(text: str) -> tuple:
    """The ANSWER and EVIDENCE fields the prompt asks for."""
    # Gemma 4 runs with skip_special_tokens=false: drop chat-template tokens such as <end_of_turn>.
    text = re.sub(r"<\|[^<>|]*\|?>|<[a-z_]+\|>|<(?:end_of_turn|start_of_turn|eos|bos)>", "", text or "").strip()
    m_ans = re.search(r"ANSWER:\s*(.*?)(?:\n\s*EVIDENCE:|\Z)", text, re.S | re.I)
    m_ev = re.search(r"EVIDENCE:\s*(.*)\Z", text, re.S | re.I)
    return (m_ans.group(1).strip() if m_ans else text), (" ".join(m_ev.group(1).split()) if m_ev else "")


def answer_paper(benchmark_dir, questions, text, images, raw_path, label, paper_path, source) -> int:
    prompt = bio.read_prompt(benchmark_dir, PROMPT_FILE)
    # Room for the prompt and the longest question besides the images.
    fitted, truncated = llm.fit_to_context(text, system_prompt=prompt, reserve_tokens=len(images) * IMAGE_TOKENS + 512)
    paper_part = f"{llm.USER_PREFIX}{fitted}"
    paper_hash = bio.sha256_of(open(paper_path, "rb").read())
    done = bio.done_questions(raw_path)
    n = 0
    for q in questions:
        if q["question_id"] in done:
            continue
        content = [llm.image_part(p) for p in images] + [
            {"type": "text", "text": paper_part + "\n\n" + prompt.replace("[INSERT QUESTION]", question_text(q))}]
        record = bio.base_record("pdf", label, q, llm.MODEL_NAME, llm.GENERATION_SETTINGS)
        record.update({"source": source, source: os.path.basename(paper_path), f"{source}_sha256": paper_hash,
                       "n_images": len(images), "input_truncated": truncated})
        print(f"[~] {q['question_id']} ({q['question_type']})...")
        llm.set_call_context(paper=q["arxiv_id"], question_id=q["question_id"], question_type=q["question_type"], source=source)
        try:
            raw = llm.chat([{"role": "user", "content": content}])
            record["raw_output"] = raw
            record["answer"], record["cited_evidence"] = parse_answer(raw)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            print(f"[!] {q['question_id']}: {record['error']}")
        bio.append_record(raw_path, record)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark-dir", required=True, help="PhysicsPaperPilot package")
    ap.add_argument("--source", default="pdf", choices=["pdf", "latex"])
    ap.add_argument("--pdf-dir", default="papers/pdf")
    ap.add_argument("--latex-dir", default="papers/latex")
    ap.add_argument("--figures-dir", default="papers/latex_figures", help="LaTeX: figures converted to PNG by common/fetch_latex_sources.py")
    ap.add_argument("--cache-dir", default="cache", help="Rendered PDF pages")
    ap.add_argument("--output", required=True, help="Raw answers JSONL; the predictions CSV is written next to it")
    ap.add_argument("--label", required=True, help="Value of the 'model' column of the predictions CSV")
    ap.add_argument("--papers", default=None, help="Only these arXiv ids (comma-separated)")
    args = ap.parse_args()

    questions = bio.load_questions(args.benchmark_dir)
    papers = sorted({q["arxiv_id"] for q in questions})
    if args.papers:
        papers = [p for p in papers if p in args.papers.split(",")]
    sources = find_sources(args.source, args.pdf_dir, args.latex_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for i, aid in enumerate(papers, 1):
        qs = [q for q in questions if q["arxiv_id"] == aid]
        print(f"\n========== Paper {i}/{len(papers)}: {aid} ({len(qs)} questions) ==========")
        if aid not in sources:
            print(f"[!] No {args.source} file for {aid}: skipped.")
            continue
        try:
            text, images = paper_inputs(sources[aid], args.source, args.figures_dir, args.cache_dir)
        except Exception as e:
            print(f"[!] Skipping {aid}: {type(e).__name__}: {e}")
            continue
        answer_paper(args.benchmark_dir, qs, text, images, args.output, args.label, sources[aid], args.source)
    if os.path.exists(args.output):
        n = bio.write_predictions(args.benchmark_dir, args.output, os.path.splitext(args.output)[0] + ".csv")
        print(f"[+] {n} answer(s) in {os.path.splitext(args.output)[0]}.csv")


if __name__ == "__main__":
    main()
