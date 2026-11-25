"""Blind manual review of PhysicsPaperPilot answers: A (PDF) and B (own JSON) of one reader, config and seed.

  make   writes an .xlsx with the 30 short/open questions: gold answer and key points, then the two
         answers as R1/R2 in random order per question (the condition and the automatic scores are in a
         hidden sheet, so they don't steer the judgement). Columns to fill: Giudizio (dropdown), KP coperti, Note.
  score  reads the filled .xlsx back: judgements per condition, human vs automatic key-point coverage
         (evaluate_v3's lexical matcher), written next to the .xlsx as <name>_risultati.csv.
The sheet contains the PRIVATE answer key: for people only, never as model input.
Usage, from the repository root:
  python evaluation/review_sheet.py make --reader qwen38 --config nothink --seed 0
  python evaluation/review_sheet.py score evaluation/human_review/revisione_qwen38_nothink_s0.xlsx
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_csv, num  # noqa: E402

JUDGEMENTS = ["corretta", "parziale", "sbagliata", "astensione"]
COND = {"A": "A_direct_pdf", "B2": "B_json_same_model"}
INSTRUCTIONS = [
    ("Revisione cieca", "Per ogni domanda ci sono due risposte, R1 e R2, in ordine casuale: una data leggendo il PDF (A), "
                        "l'altra leggendo solo il JSON prodotto dallo stesso modello (B). Quale sia quale è nel foglio nascosto 'Chiave'."),
    ("Giudizio", "corretta = contiene i punti chiave essenziali della risposta gold, senza errori fattuali (le parafrasi valgono). "
                 "parziale = alcuni punti chiave, oppure un'imprecisione minore. "
                 "sbagliata = errore fattuale, risposta a un'altra domanda, o nessun punto chiave. "
                 "astensione = il modello dice che l'informazione non è recuperabile ('NOT RECOVERABLE ...')."),
    ("KP coperti", "Numeri dei punti chiave presenti nella risposta, separati da virgola (es. 1,3). Vuoto = nessuno."),
    ("Note", "Facoltative: errori, informazioni in più non presenti nel gold, gold discutibile, ecc."),
    ("Riferimento", "Gold e punti chiave dal pacchetto PhysicsPaperPilot v0.3.1 (chiave privata, solo lettura umana)."),
]


def make(args):
    bench, bd = args.runs_dir, args.benchmark_dir
    stem = f"{args.reader}_{args.config}_s{args.seed}"
    questions = [q for q in read_csv(os.path.join(bd, "01_questions_for_models", "questions_public.csv"))
                 if q["question_type"] != "multiple_choice"]
    key = {r["question_id"]: r for r in read_csv(os.path.join(bd, "02_private_answer_key", "answer_key_private.csv"))}
    lexicon = defaultdict(list)
    for r in read_csv(os.path.join(bd, "02_private_answer_key", "keypoints_lexicon.csv")):
        lexicon[r["question_id"]].append(r["key_point"])
    scored = {}
    for cond in ("A", "B2"):
        path = os.path.join(bench, "scores", f"{stem}_{cond}", "scored_answers.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run evaluation/score_runs.py first")
        scored[cond] = {r["question_id"]: r for r in read_csv(path)}

    rng = random.Random(f"{stem}-review")
    wb = Workbook()
    ws = wb.active
    ws.title = "Revisione"
    header = ["Riga", "Domanda ID", "Paper", "Tipo", "Categoria", "Domanda", "Risposta gold", "Punti chiave",
              "Valori numerici attesi", "Risposta", "Testo della risposta", "Giudizio", "KP coperti", "Note"]
    ws.append(header)
    hidden = wb.create_sheet("Chiave")
    hidden.append(["Riga", "Domanda ID", "Condizione", "Copertura KP automatica", "Token F1 automatico", "KP totali"])
    fills = [PatternFill("solid", fgColor="FFFFFF"), PatternFill("solid", fgColor="EEF3F8")]
    row_id = 0
    for qi, q in enumerate(questions):
        k = key[q["question_id"]]
        kps = lexicon.get(q["question_id"], [])
        nums = json.loads(k.get("numeric_targets_json") or "[]")
        conds = ["A", "B2"]
        rng.shuffle(conds)
        for label, cond in zip(("R1", "R2"), conds):
            a = scored[cond].get(q["question_id"])
            row_id += 1
            ws.append([row_id, q["question_id"], q["arxiv_id"], q["question_type"], q["category"], q["question"],
                       k["gold_answer"], "\n".join(f"{i + 1}. {p}" for i, p in enumerate(kps)),
                       "; ".join(f"{n.get('label', '')} = {n.get('value', '')}" for n in nums),
                       label, (a or {}).get("model_answer", "") or "(nessuna risposta: JSON mancante)", "", "", ""])
            for cell in ws[ws.max_row]:
                cell.fill = fills[qi % 2]
            hidden.append([row_id, q["question_id"], cond, num((a or {}).get("keypoint_coverage")),
                           num((a or {}).get("token_f1")), len(kps)])
    widths = [6, 18, 11, 12, 18, 40, 45, 45, 22, 9, 70, 13, 11, 30]
    for col, w in zip("ABCDEFGHIJKLMN", widths):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "F2"
    dv = DataValidation(type="list", formula1=f'"{",".join(JUDGEMENTS)}"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"L2:L{ws.max_row}")
    hidden.sheet_state = "hidden"
    info = wb.create_sheet("Istruzioni", 0)
    info.append([f"Revisione manuale: {args.reader}, {args.config}, seed {args.seed}", ""])
    info["A1"].font = Font(bold=True)
    for title, text in INSTRUCTIONS:
        info.append([title, text])
    info.column_dimensions["A"].width = 18
    info.column_dimensions["B"].width = 110
    for row in info.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.active = 1
    out = os.path.join(args.out_dir, f"revisione_{stem}.xlsx")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists (it may contain your judgements): use --force to overwrite")
    wb.save(out)
    print(f"[+] {out}: {row_id} answers to review ({row_id // 2} questions)")


def score(args):
    wb = load_workbook(args.xlsx, data_only=True)
    review = {r[0]: r for r in wb["Revisione"].iter_rows(min_row=2, values_only=True) if r[0] is not None}
    keyrows = {r[0]: r for r in wb["Chiave"].iter_rows(min_row=2, values_only=True) if r[0] is not None}
    rows = []
    for rid, r in sorted(review.items()):
        k = keyrows[rid]
        judgement = (r[11] or "").strip().lower()
        covered = {s.strip() for s in str(r[12] or "").replace(";", ",").split(",") if s.strip()}
        total = k[5] or 0
        rows.append({"riga": rid, "question_id": k[1], "condizione": k[2], "giudizio": judgement,
                     "kp_coperti_umano": len(covered), "kp_totali": total,
                     "copertura_kp_umano": round(len(covered) / total, 3) if total else "",
                     "copertura_kp_automatica": k[3], "token_f1": k[4], "note": r[13] or ""})
    done = [x for x in rows if x["giudizio"] in JUDGEMENTS]
    print(f"[~] {len(done)}/{len(rows)} answers judged")
    for cond, name in COND.items():
        sub = [x for x in done if x["condizione"] == cond]
        if not sub:
            continue
        counts = {j: sum(1 for x in sub if x["giudizio"] == j) for j in JUDGEMENTS}
        hum = [x["copertura_kp_umano"] for x in sub if x["copertura_kp_umano"] != ""]
        auto = [x["copertura_kp_automatica"] for x in sub if x["copertura_kp_automatica"] is not None and x["copertura_kp_umano"] != ""]
        print(f"  {name}: {counts} | corrette {100 * counts['corretta'] / len(sub):.0f}% | "
              f"corrette+parziali {100 * (counts['corretta'] + counts['parziale']) / len(sub):.0f}% | "
              f"copertura KP umana {100 * statistics.mean(hum):.0f}% vs automatica {100 * statistics.mean(auto):.0f}%"
              if hum and auto else f"  {name}: {counts}")
    out = os.path.splitext(args.xlsx)[0] + "_risultati.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[+] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--reader", default="qwen38")
    m.add_argument("--config", default="nothink")
    m.add_argument("--seed", default="0")
    m.add_argument("--runs-dir", default="runs/pdf")
    m.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2")
    m.add_argument("--out-dir", default="evaluation/human_review")
    m.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("xlsx")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
