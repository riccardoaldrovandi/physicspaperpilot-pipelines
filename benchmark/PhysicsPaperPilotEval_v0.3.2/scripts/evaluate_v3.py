#!/usr/bin/env python3
"""Deterministic, no-generative-LLM evaluator for PhysicsPaperPilot (release v0.3.1).

Metrics:
- MCQ accuracy
- Normalized exact match, token P/R/F1, ROUGE-L
- Deterministic key-point coverage (lexical phrase/alias matching)
- Section-citation P/R/F1 (NOT paragraph-level evidence F1)
- Numeric target coverage with gold tolerances
- Optional local BERTScore encoder (no generation / no judge)

No LLM API or generative model is called by this script.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / '01_questions_for_models' / 'questions_public.csv'
KEY_PATH = ROOT / '02_private_answer_key' / 'answer_key_private.csv'
LEXICON_PATH = ROOT / '02_private_answer_key' / 'keypoints_lexicon.csv'
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being','of','for','to','and','or','in','on','at','by','with','from',
    'that','this','these','those','it','its','as','into','between','their','they','them','which','who','what','how','why',
    'paper','authors','author','use','uses','used','using','one','two','three','each','per'
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def safe_json(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (json.JSONDecodeError, TypeError):
        return default


SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
SUPERSCRIPT_POWER_RE = re.compile(r'(\d)\s*([⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)')


def superscripts_to_caret(text: str) -> str:
    """v0.3.1: write Unicode superscript exponents as ^n before NFKC.

    NFKC turns '10⁵' into '105' and '10⁻⁶' into '10-6', so a correct answer written
    with superscripts was scored as the number 105 (or as 10 and 6)."""
    return SUPERSCRIPT_POWER_RE.sub(lambda m: m.group(1) + '^' + m.group(2).translate(SUPERSCRIPTS), str(text or ''))


def normalize_text(text: str) -> str:
    """Normalize unicode/scientific glyphs to a stable lexical form."""
    s = unicodedata.normalize('NFKC', superscripts_to_caret(text)).lower()
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    replacements = {
        'γ': ' gamma ', 'π': ' pi ', 'η': ' eta ', 'φ': ' phi ', 'δ': ' delta ',
        '×': ' x ', '·': ' ', '⁺': ' plus ', '⁻': ' minus ', '−': '-',
        '–': '-', '—': '-', '≈': ' approximately ', '≥': ' greater equal ', '≤': ' less equal ',
        'μ': ' mu ', 'σ': ' sigma ', 'ε': ' epsilon ', 'χ': ' chi ',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Normalize powers such as 10^{5}, 10^5 and 10⁵. v0.3.1: keep the sign as a word
    # ('10^-6' -> '10 to minus 6', as the '10-6' rule below), it was dropped with the punctuation;
    # and end with a space: the pattern eats the following whitespace, so '10^5 on' became '10 to 5on'.
    s = re.sub(r'10\s*\^\s*\{?\s*([+-]?)(\d+)\s*\}?',
               lambda m: '10 to ' + ('minus ' if m.group(1) == '-' else '') + m.group(2) + ' ', s)
    s = re.sub(r'10\s*-\s*(\d+)', r'10 to minus \1', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def tokenize(text: str) -> list[str]:
    s = normalize_text(text)
    return TOKEN_RE.findall(s)


def exact_match(pred: str, refs: list[str]) -> float:
    pt = tokenize(pred)
    return float(any(pt == tokenize(ref) for ref in refs))


def token_prf(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = tokenize(pred), tokenize(ref)
    if not p and not r:
        return 1.0, 1.0, 1.0
    if not p or not r:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    precision = overlap / len(p)
    recall = overlap / len(r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def lcs_length(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for tok_a in a:
        current = [0]
        for j, tok_b in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if tok_a == tok_b else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_prf(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = tokenize(pred), tokenize(ref)
    if not p and not r:
        return 1.0, 1.0, 1.0
    if not p or not r:
        return 0.0, 0.0, 0.0
    common = lcs_length(p, r)
    precision, recall = common / len(p), common / len(r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def extract_numbers(text: str) -> list[float]:
    s = unicodedata.normalize('NFKC', superscripts_to_caret(text)).replace('×', 'x').replace('−', '-')
    found: list[tuple[int, int, float]] = []
    occupied: list[tuple[int, int]] = []
    patterns = [
        re.compile(r'(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*x\s*10\s*\^\s*\{?([+-]?\d+)\}?(?!\d)', re.I),
        re.compile(r'(?<![\w.])10\s*\^\s*\{?([+-]?\d+)\}?(?!\d)', re.I),
        re.compile(r'(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+)(?!\w)'),
    ]
    for idx, pat in enumerate(patterns):
        for m in pat.finditer(s):
            if any(not (m.end() <= a or m.start() >= b) for a, b in occupied):
                continue
            try:
                if idx == 0: value = float(m.group(1)) * (10 ** int(m.group(2)))
                elif idx == 1: value = 10 ** int(m.group(1))
                else: value = float(m.group(1))
                found.append((m.start(), m.end(), value)); occupied.append((m.start(), m.end()))
            except (ValueError, OverflowError):
                pass
    plain = re.compile(r'(?<![\w.])([+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+))(?!\d)')
    for m in plain.finditer(s):
        if any(not (m.end() <= a or m.start() >= b) for a, b in occupied):
            continue
        try:
            found.append((m.start(), m.end(), float(m.group(1).replace(',', '')))); occupied.append((m.start(), m.end()))
        except ValueError:
            pass
    return [x[2] for x in sorted(found, key=lambda x: x[0])]


def numeric_score(pred: str, targets: list[dict[str, Any]]) -> tuple[int, int, float | str, int | str, str]:
    if not targets:
        return 0, 0, '', '', ''
    values = extract_numbers(pred)
    used: set[int] = set(); matched = 0; details = []
    for target in targets:
        expected = float(target['value'])
        rel = float(target.get('rel_tolerance', 0.0)); abs_tol = float(target.get('abs_tolerance', 0.0))
        candidates = []
        for i, actual in enumerate(values):
            if i not in used and math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol):
                candidates.append((abs(actual - expected), i, actual))
        if candidates:
            _, i, actual = min(candidates); used.add(i); matched += 1
            details.append(f"MATCH {target.get('label','value')}: expected {expected:g}, found {actual:g}")
        else:
            details.append(f"MISS {target.get('label','value')}: expected {expected:g}")
    return matched, len(targets), matched / len(targets), int(matched == len(targets)), '; '.join(details)


def parse_mcq(pred: str, explicit: str) -> str:
    choice = str(explicit or '').strip().upper()
    if choice in {'A', 'B', 'C', 'D'}:
        return choice
    m = re.match(r'^\s*(?:\*\*)?\(?\s*([ABCD])\s*[\).:\-]?', str(pred or '').upper())
    return m.group(1) if m else ''


def phrase_present(answer_norm: str, phrase: str) -> bool:
    norm = normalize_text(phrase)
    if not norm:
        return False
    # Exact contiguous token phrase; surrounding spaces avoid substring false positives.
    return f' {norm} ' in f' {answer_norm} '


def keypoint_scores(answer: str, qid: str, lexicon: dict[str, list[dict[str, Any]]]) -> tuple[int, int, float | str, str, str]:
    points = lexicon.get(qid, [])
    if not points:
        return 0, 0, '', '', ''
    norm_answer = normalize_text(answer)
    matched_ids = []; missing_ids = []
    for kp in points:
        aliases = kp.get('accepted_phrases', [])
        hit = any(phrase_present(norm_answer, alias) for alias in aliases)
        # Conservative token-set fallback: only accept if all informative anchor tokens are present.
        # This accommodates minor word-order changes without silently using semantic/LLM judging.
        if not hit:
            anchors = [t for t in tokenize(kp.get('key_point', '')) if t not in STOPWORDS and len(t) > 1]
            anchors = list(dict.fromkeys(anchors))
            if len(anchors) >= 3:
                present = sum(1 for t in anchors if t in norm_answer.split())
                hit = present / len(anchors) >= 0.85
        if hit: matched_ids.append(kp['keypoint_id'])
        else: missing_ids.append(kp['keypoint_id'])
    total = len(points); matched = len(matched_ids)
    return matched, total, matched / total, ';'.join(matched_ids), ';'.join(missing_ids)


def split_citations(text: str) -> list[str]:
    raw = str(text or '').strip()
    if not raw:
        return []
    parsed = safe_json(raw, None)
    if isinstance(parsed, list):
        items = [str(x).strip() for x in parsed if str(x).strip()]
    else:
        items = [x.strip() for x in re.split(r'[;|\n]+', raw) if x.strip()]
    # Remove common labels and page suffixes while retaining the section heading.
    cleaned = []
    for item in items:
        x = re.sub(r'^\s*(?:evidence|source|sources|citation|citations)\s*:\s*', '', item, flags=re.I)
        x = re.sub(r'\s*\(?\s*(?:p\.?|page)\s*\d+(?:\s*[-–]\s*\d+)?\s*\)?\s*$', '', x, flags=re.I)
        x = re.sub(r'^\s*(?:section|sec\.?)\s+', '', x, flags=re.I)
        if x.strip(): cleaned.append(x.strip())
    return cleaned


def section_code(text: str) -> str:
    norm = normalize_text(text)
    m = re.match(r'^((?:[ivxlcdm]+|\d+)(?:\s+\d+)*)(?:\s|$)', norm)
    return m.group(1).replace(' ', '.') if m else ''


# v0.3.1: a leading section number in any of the styles found in the 15 papers and in
# the gold labels: '2.1', '1', 'II', 'II.1', 'II.A', 'A.' (REVTeX subsection), with an
# optional final dot. Roman numerals and the single letter must be upper case, so a
# title starting with an ordinary word ('Mixed ...', 'A model ...') is not stripped.
SECTION_NUMBER_RE = re.compile(r'^\s*(?:(?:[IVXLCDM]+|\d+)(?:\.(?:\d+|[A-Z]))*\.?|[A-Z]\.)\s+')


def section_title(text: str) -> str:
    """Normalized heading title without its section number."""
    return normalize_text(SECTION_NUMBER_RE.sub('', str(text or '').strip(), count=1))


def section_matches(pred: str, gold: str) -> bool:
    pn, gn = normalize_text(pred), normalize_text(gold)
    if not pn or not gn: return False
    if pn == gn: return True
    # v0.3.1: same title, whatever the numbering. The gold labels follow the arXiv HTML
    # numbering ('II.1 Graph creation') while the PDFs print 'A. Graph creation',
    # '1 Introduction' or no number at all, so a correct citation copied from the PDF
    # scored as a false positive.
    pt, gt = section_title(pred), section_title(gold)
    if pt and pt == gt: return True
    pc, gc = section_code(pred), section_code(gold)
    return bool(pc and gc and pc == gc)


def evidence_section_score(pred_text: str, gold_groups: list[dict[str, Any]] | list[str]) -> tuple[int, int, int, float, float, float, str, str]:
    """Score cited section headings against evidence groups.

    One gold evidence group may have multiple acceptable source headings. A
    group is covered when at least one cited heading matches any alternative.
    Extra valid alternatives for the same group are not double-counted; a
    citation that matches no gold group is a false positive.

    This is section-level source localization, not paragraph-level evidence
    verification.
    """
    preds = split_citations(pred_text)
    groups: list[dict[str, Any]] = []
    for i, g in enumerate(gold_groups):
        if isinstance(g, dict):
            groups.append({
                'evidence_id': str(g.get('evidence_id', f'E{i+1}')),
                'accepted_sections': [str(x) for x in g.get('accepted_sections', []) if str(x).strip()]
            })
        else:
            groups.append({'evidence_id': f'E{i+1}', 'accepted_sections': [str(g)]})

    covered_groups: set[int] = set()
    invalid_predictions: list[str] = []
    matched_details: list[str] = []
    for pred in preds:
        matched_ids = []
        for gi, group in enumerate(groups):
            for section in group['accepted_sections']:
                if section_matches(pred, section):
                    covered_groups.add(gi)
                    matched_ids.append(group['evidence_id'])
                    break
        if matched_ids:
            matched_details.append(f"{pred} => {','.join(sorted(set(matched_ids)))}")
        else:
            invalid_predictions.append(pred)

    tp = len(covered_groups)
    fp = len(invalid_predictions)
    fn = len(groups) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = ';'.join(groups[i]['evidence_id'] for i in range(len(groups)) if i not in covered_groups)
    matches = '; '.join(matched_details)
    if invalid_predictions:
        matches += ('; ' if matches else '') + 'UNMATCHED: ' + '; '.join(invalid_predictions)
    return tp, fp, fn, precision, recall, f1, matches, missing


def macro_over_papers(rows: list[dict[str, Any]], metric: str, type_filter: str | None = None) -> float | None:
    by_paper: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if type_filter and r['question_type'] != type_filter: continue
        v = r.get(metric)
        try:
            if v in ('', None): continue
            fv = float(v)
            if math.isnan(fv): continue
        except (ValueError, TypeError):
            continue
        by_paper[r['arxiv_id']].append(fv)
    paper_means = [statistics.mean(vs) for vs in by_paper.values() if vs]
    return statistics.mean(paper_means) if paper_means else None


def main() -> None:
    parser = argparse.ArgumentParser(description='Deterministic no-LLM evaluation for PhysicsPaperPilot v0.3.1.')
    parser.add_argument('--input', required=True, help='CSV: model, question_id, model_answer, optional mcq_choice, cited_evidence')
    parser.add_argument('--outdir', default='04_results')
    parser.add_argument('--with-bertscore', action='store_true', help='Optional local encoder metric; no generative judge/API')
    parser.add_argument('--bert-model', default='roberta-large')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    questions = {r['question_id']: r for r in read_csv(PUBLIC_PATH)}
    keys = {r['question_id']: r for r in read_csv(KEY_PATH)}
    lex_rows = read_csv(LEXICON_PATH)
    lexicon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lex_rows:
        lexicon[row['question_id']].append({
            'keypoint_id': row['keypoint_id'],
            'key_point': row['key_point'],
            'accepted_phrases': safe_json(row.get('accepted_phrases_json', ''), [])
        })
    predictions = read_csv(Path(args.input))
    if not predictions: raise SystemExit('Input CSV is empty.')

    scored: list[dict[str, Any]] = []; warnings: list[str] = []; seen: set[tuple[str, str]] = set()
    for line_no, row in enumerate(predictions, start=2):
        model = (row.get('model') or '').strip(); qid = (row.get('question_id') or '').strip()
        if not model and not qid and not (row.get('model_answer') or '').strip(): continue
        if not model or not qid:
            warnings.append(f'Line {line_no}: model and question_id are required; row skipped.'); continue
        if qid not in keys or qid not in questions:
            warnings.append(f'Line {line_no}: unknown question_id {qid}; row skipped.'); continue
        if (model, qid) in seen:
            warnings.append(f'Line {line_no}: duplicate model/question pair {model}/{qid}; duplicate row skipped.'); continue
        seen.add((model, qid))
        q, k = questions[qid], keys[qid]
        qtype = q['question_type']; answer = (row.get('model_answer') or '').strip()
        refs = safe_json(k.get('reference_answers_json', ''), [k.get('gold_answer', '')]) or [k.get('gold_answer', '')]
        refs = [str(x) for x in refs if str(x).strip()] or ['']
        gold_sections = safe_json(k.get('gold_evidence_groups_json', ''), [])
        out: dict[str, Any] = {
            'model': model, 'question_id': qid, 'arxiv_id': q['arxiv_id'], 'question_type': qtype, 'category': q['category'],
            'model_answer': answer, 'mcq_prediction': '', 'mcq_correct': '',
            'exact_match': '', 'token_precision': '', 'token_recall': '', 'token_f1': '',
            'rougeL_precision': '', 'rougeL_recall': '', 'rougeL_f1': '',
            'keypoints_matched': '', 'keypoints_total': '', 'keypoint_coverage': '', 'keypoint_recall': '',
            'keypoints_matched_ids': '', 'keypoints_missing_ids': '',
            'evidence_sections_tp': '', 'evidence_sections_fp': '', 'evidence_sections_fn': '',
            'evidence_section_precision': '', 'evidence_section_recall': '', 'evidence_section_f1': '',
            'evidence_section_matches': '', 'evidence_sections_missing': '',
            'numeric_targets_matched': '', 'numeric_targets_total': '', 'numeric_target_coverage': '',
            'numeric_all_targets_correct': '', 'numeric_details': '',
            'bertscore_precision': '', 'bertscore_recall': '', 'bertscore_f1': '',
        }
        if qtype == 'multiple_choice':
            choice = parse_mcq(answer, row.get('mcq_choice', ''))
            gold = str(k.get('gold_answer', '')).strip().upper()
            out['mcq_prediction'] = choice
            if choice: out['mcq_correct'] = int(choice == gold)
            else: warnings.append(f'Line {line_no}: no A/B/C/D choice parsed for {qid}.')
        else:
            metric_values = []
            for ref in refs:
                em = exact_match(answer, [ref]); tp, tr, tf = token_prf(answer, ref); rp, rr, rf = rouge_l_prf(answer, ref)
                metric_values.append((em, tp, tr, tf, rp, rr, rf))
            fields = ['exact_match','token_precision','token_recall','token_f1','rougeL_precision','rougeL_recall','rougeL_f1']
            for idx, field in enumerate(fields): out[field] = max(v[idx] for v in metric_values)
            km, kt, kc, km_ids, km_missing = keypoint_scores(answer, qid, lexicon)
            out['keypoints_matched'], out['keypoints_total'], out['keypoint_coverage'], out['keypoint_recall'] = km, kt, kc, kc
            out['keypoints_matched_ids'], out['keypoints_missing_ids'] = km_ids, km_missing
            targets = safe_json(k.get('numeric_targets_json', ''), [])
            if targets:
                nm, nt, ncov, nall, ndetails = numeric_score(answer, targets)
                out['numeric_targets_matched'], out['numeric_targets_total'] = nm, nt
                out['numeric_target_coverage'], out['numeric_all_targets_correct'] = ncov, nall
                out['numeric_details'] = ndetails
        # Section-level source citation scoring is run for every question type.
        evidence = row.get('cited_evidence', '') or ''
        tp, fp, fn, ep, er, ef, matches, missing = evidence_section_score(evidence, gold_sections)
        out['evidence_sections_tp'], out['evidence_sections_fp'], out['evidence_sections_fn'] = tp, fp, fn
        out['evidence_section_precision'], out['evidence_section_recall'], out['evidence_section_f1'] = ep, er, ef
        out['evidence_section_matches'], out['evidence_sections_missing'] = matches, missing
        if not evidence.strip(): warnings.append(f'Line {line_no}: no cited_evidence for {qid}; evidence-section score is 0.')
        scored.append(out)

    if args.with_bertscore:
        indices = [i for i, r in enumerate(scored) if r['question_type'] != 'multiple_choice']
        if indices:
            try: from bert_score import score as bert_score
            except ImportError as exc: raise SystemExit('BERTScore requested but missing. Install requirements_bertscore.txt') from exc
            cands, refs, owners = [], [], []
            for i in indices:
                k = keys[scored[i]['question_id']]
                references = safe_json(k.get('reference_answers_json', ''), [k.get('gold_answer', '')]) or [k.get('gold_answer', '')]
                for ref in references:
                    cands.append(scored[i]['model_answer']); refs.append(str(ref)); owners.append(i)
            print(f'Computing local BERTScore with {args.bert_model} on {len(cands)} pair(s)...', file=sys.stderr)
            P, R, F = bert_score(cands, refs, lang='en', model_type=args.bert_model,
                                 batch_size=args.batch_size, rescale_with_baseline=False, verbose=False)
            grouped: dict[int, list[tuple[float,float,float]]] = defaultdict(list)
            for j, owner in enumerate(owners): grouped[owner].append((float(P[j]),float(R[j]),float(F[j])))
            for owner, vals in grouped.items():
                best = max(vals, key=lambda x: x[2])
                scored[owner]['bertscore_precision'], scored[owner]['bertscore_recall'], scored[owner]['bertscore_f1'] = best

    models = sorted({r['model'] for r in scored}); expected = set(questions)
    for model in models:
        present = {r['question_id'] for r in scored if r['model'] == model}
        missing = sorted(expected - present)
        if missing: warnings.append(f'{model}: {len(missing)} question(s) missing: {", ".join(missing[:10])}')

    write_csv(outdir / 'scored_answers.csv', scored)
    metrics = {
        'mcq_accuracy': ('mcq_correct', 'multiple_choice'),
        'short_exact_match': ('exact_match', 'short_answer'),
        'short_token_f1': ('token_f1', 'short_answer'),
        'short_keypoint_coverage': ('keypoint_coverage', 'short_answer'),
        'short_evidence_section_f1': ('evidence_section_f1', 'short_answer'),
        'open_token_f1': ('token_f1', 'open_ended'),
        'open_rougeL_f1': ('rougeL_f1', 'open_ended'),
        'open_keypoint_coverage': ('keypoint_coverage', 'open_ended'),
        'open_evidence_section_f1': ('evidence_section_f1', 'open_ended'),
        'all_evidence_section_f1': ('evidence_section_f1', None),
        'numeric_target_coverage': ('numeric_target_coverage', None),
        'short_bertscore_f1': ('bertscore_f1', 'short_answer'),
        'open_bertscore_f1': ('bertscore_f1', 'open_ended'),
    }
    summary = []
    for model in models:
        rs = [r for r in scored if r['model'] == model]
        item = {'model': model, 'n_answers': len(rs), 'n_papers': len({r['arxiv_id'] for r in rs})}
        for label, (field, typ) in metrics.items():
            subset = [r for r in rs if typ is None or r['question_type'] == typ]
            val = macro_over_papers(subset, field)
            item[label + '_percent'] = round(val * 100, 3) if val is not None else ''
            item[label + '_n'] = sum(1 for r in subset if r.get(field) not in ('', None))
        summary.append(item)
    write_csv(outdir / 'summary_by_model.csv', summary)

    paper_rows = []
    groups: dict[tuple[str,str], list[dict[str, Any]]] = defaultdict(list)
    for r in scored: groups[(r['model'], r['arxiv_id'])].append(r)
    for (model, paper), rs in sorted(groups.items()):
        item = {'model': model, 'arxiv_id': paper, 'n_answers': len(rs)}
        for label, (field, typ) in metrics.items():
            vals = []
            for r in rs:
                if typ and r['question_type'] != typ: continue
                try:
                    v = r.get(field)
                    if v not in ('', None) and not math.isnan(float(v)): vals.append(float(v))
                except (ValueError, TypeError): pass
            item[label + '_percent'] = round(statistics.mean(vals) * 100, 3) if vals else ''
        paper_rows.append(item)
    write_csv(outdir / 'summary_by_paper.csv', paper_rows)

    type_rows = []
    for model in models:
        for qtype in ['multiple_choice', 'short_answer', 'open_ended']:
            rs = [r for r in scored if r['model'] == model and r['question_type'] == qtype]
            item = {'model': model, 'question_type': qtype, 'n_answers': len(rs)}
            fields = ['mcq_correct'] if qtype == 'multiple_choice' else [
                'exact_match','token_precision','token_recall','token_f1','rougeL_f1','keypoint_coverage',
                'evidence_section_precision','evidence_section_recall','evidence_section_f1','numeric_target_coverage','bertscore_f1']
            for field in fields:
                val = macro_over_papers(rs, field)
                item[field + '_percent'] = round(val * 100, 3) if val is not None else ''
            type_rows.append(item)
    write_csv(outdir / 'summary_by_question_type.csv', type_rows)
    (outdir / 'run_warnings.txt').write_text('\n'.join(warnings) if warnings else 'No warnings.\n', encoding='utf-8')
    print(f'Processed {len(scored)} answers from {len(models)} model(s).')
    print(f'Results saved to: {outdir.resolve()}')
    print('No generative LLM judge or API was used.')
    print('Evidence metric: section-citation P/R/F1 only; not passage-level evidence verification.')
    if args.with_bertscore: print(f'Optional local encoder used: {args.bert_model}')
    if warnings: print(f'Warnings: {len(warnings)} — inspect {outdir / "run_warnings.txt"}.')

if __name__ == '__main__':
    main()
