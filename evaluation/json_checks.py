"""Level 1 and Level 2 checks of the structured JSONs of Pipeline B (step B1, prompt v2.1).

Level 1 - is the JSON well formed? It fails if any of these fires: invalid JSON or not an object;
missing keys or wrong types against the v2.1 schema; more than half of the paper fields empty;
repetitive text (4-gram repetition > 0.30 or zlib compression < 0.25, both without the
"(approx., read from the figure)" markers and the figure curves, which repeat by design); text copied
from the schema's own examples (similarity >= 0.85); filler items with no letter or digit (unit
symbols such as "%" excepted).
Level 2 - is what the JSON says about the paper in the paper? Mean of the components that have
something to check: numbers (cited numbers found in the paper text, written in any equivalent
notation; single digits and values read off plots excluded), references (cited tables, figures,
equations, algorithms that exist), datasets (named datasets that appear in the paper).
Outputs in <runs-dir>/json_checks/: per_json.csv (one row per JSON) and summary.csv (per model and
config: Level 1 passed, means, Level 2 with a 95% bootstrap CI over papers, ungrounded counts).
  python evaluation/json_checks.py --runs-dir runs/pdf
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import sys
import zlib
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIGS, MODELS, model_order, write_csv  # noqa: E402
from pipeline_b.prompt import FIELDS, PAPER_FIELDS, SYSTEM_PROMPT, as_text, is_empty_value  # noqa: E402

EXPECTED_KEYS = list(FIELDS)

MAX_EMPTY_FIELD_RATIO = 0.5


# v2 / v2.1 answers are lists of short structured items (key_results, tables,
# figures) that legitimately share phrasing ("Dataset 2, batch size ...").
# With the figure-reading markers left out (_repetition_text), the 315 valid
# v2.1 JSONs of the PhysicsPaperPilot benchmark (2026-11-23) reach at most
# 0.235, the looping ones of the corrupted Nemotron run 0.80-0.99.
MAX_REPEATED_NGRAM_RATIO_V2 = 0.30


MIN_COMPRESSION_RATIO = 0.25


# Schema echo: a weak model (Velvet-14B, job 60313868) can return valid JSON
# whose fields are SYSTEM_PROMPT's own example texts ("Brief description of the
# main problem...") or pure filler (". . ."). Neither is caught by the checks
# above. 0.85 similarity catches near-verbatim copies; on the 82 real records
# of runs v3-v9 no field comes close.
ECHO_SIMILARITY = 0.85


def _keyed_leaves(value, key: str = None) -> list:
    """(innermost object key, leaf) pairs: _leaves with the name of the slot each value fills."""
    if isinstance(value, dict):
        return [pair for k, v in value.items() for pair in _keyed_leaves(v, k)]
    if isinstance(value, list):
        return [pair for v in value for pair in _keyed_leaves(v, key)]
    return [] if value is None else [(key, value)]


# Slots whose value is a symbol by nature: "%" or "×" in key_results[].unit is an answer, not filler.
_SYMBOL_SLOTS = {"unit"}


def _leaves(value) -> list:
    """Every string (or scalar) inside a value, however deeply nested (v2 has objects and lists of objects)."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, list):
        return [leaf for v in value for leaf in _leaves(v)]
    return [] if value is None else [value]


def _schema_examples(prompt: str) -> list:
    """The example texts of a prompt's JSON schema, normalized."""
    start = prompt.find("{", prompt.find("JSON Schema"))
    end = prompt.find("\n}", start) + 2
    return [" ".join(str(e).lower().split()) for e in _leaves(json.loads(prompt[start:end]))]


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    """Share of word n-grams that are repeats of an earlier one (0 = no repetition)."""
    words = text.lower().split()
    if len(words) < n * 5:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams)


# The marker every v2 prompt asks for on values read off a plot, e.g. "(approx., read from the figure)".
_FIGURE_MARKER_RE = re.compile(r"\(\s*approx[^)]*\)", re.IGNORECASE)


def compression_ratio(text: str) -> float:
    """zlib size / raw size: a looping answer compresses far better than prose."""
    raw = text.encode("utf-8")
    if len(raw) < 500:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def _check_v2(value, spec, path: str, result: dict) -> None:
    """Type and key checks of one v2 field against its FIELDS_V2 spec, recursively."""
    if spec == "string":
        if isinstance(value, str):
            return
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            result["list_for_string"].append(path)
        else:
            result["type_errors"].append(path)
    elif spec == "list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            result["type_errors"].append(path)
    elif isinstance(spec, list):
        if not isinstance(value, list):
            result["type_errors"].append(path)
            return
        for item in value:
            _check_v2(item, spec[0], f"{path}[]", result)
    else:
        if not isinstance(value, dict):
            result["type_errors"].append(path)
            return
        for sub, sub_spec in spec.items():
            if sub not in value:
                result["missing_keys"].append(f"{path}.{sub}")
            else:
                _check_v2(value[sub], sub_spec, f"{path}.{sub}", result)
        result["extra_keys"].extend(f"{path}.{k}" for k in value if k not in spec)


_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text: str) -> set:
    """Numbers normalized so '1,000' == '1000' and '0.50' == '0.5'."""
    out = set()
    for m in _NUMBER_RE.findall(text):
        n = m.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


# The same quantity written another way in the paper's text (checked on the
# PhysicsPaperPilot benchmark, 2026-11-23): thousands separated by a space or by
# ", 000" after PDF extraction ("148 157", "5 × 100, 000"), k / M suffixes ("40k"),
# powers of ten ("10^5", "10⁵", and "× 106" where the PDF flattened the
# superscript; a bare "105" is left alone, it is just as likely one hundred and
# five) and e-notation. Each adds its value to the numbers available in the paper.
# ", " only before "000": "[1, 100] GeV" is a range, not 1100.
_SPACE_THOUSANDS_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:(?:,[ \u00a0\u2009\u202f](?=000)|[ \u00a0\u2009\u202f])\d{3})+(?![\d])")


# No capital K: "90 K" is a temperature.
_SUFFIX_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(k|M|million|thousand)\b")


_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


_MULT = r"\s*(?:×|x|\*|·|\\times|\\cdot)\s*"


_POWER_RE = re.compile(r"(?:(?<![\w.])(\d+(?:\.\d+)?)" + _MULT + r"|(?<![\w.]))10\s*(?:\^\s*\{?\s*([-−–]?\d{1,2})(?![\d.])\s*\}?|([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+))")


_FLAT_POWER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)" + _MULT + r"10([-−–]?\d{1,2})(?![\d.])")


_E_NOTATION_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([-+]?\d{1,2})\b")


def _canonical(value) -> str:
    """A Decimal in the form _numbers gives ('100000', '0.5')."""
    from decimal import Decimal
    text = format(Decimal(value).normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _notation_numbers(text: str) -> set:
    from decimal import Decimal, InvalidOperation
    out = set()
    try:
        for m in _SPACE_THOUSANDS_RE.finditer(text):
            out.add(_canonical(re.sub(r"[^\d]", "", m.group(0))))
        for m in _SUFFIX_RE.finditer(text):
            factor = {"k": 3, "thousand": 3, "M": 6, "million": 6}[m.group(2)]
            out.add(_canonical(Decimal(m.group(1)).scaleb(factor)))
        for m in _POWER_RE.finditer(text):
            exp = (m.group(2) or m.group(3).translate(_SUPERSCRIPTS)).replace("−", "-").replace("–", "-")
            out.add(_canonical(Decimal(m.group(1) or "1").scaleb(int(exp))))
        for m in _FLAT_POWER_RE.finditer(text):
            out.add(_canonical(Decimal(m.group(1)).scaleb(int(m.group(2).replace("−", "-").replace("–", "-")))))
        for m in _E_NOTATION_RE.finditer(text):
            out.add(_canonical(Decimal(m.group(1)).scaleb(int(m.group(2)))))
    except (InvalidOperation, ValueError):
        pass
    return out


# "[300,350,400]" is read by _NUMBER_RE as the single number 300350400. A
# comma-grouped number whose groups all have three digits also counts as found
# when every group is found as a number of its own.
_GROUPED_RE = re.compile(r"(?<![\w.])\d{3}(?:,\d{3})+(?![\d.])")


def _list_alternatives(text: str) -> dict:
    return {m.group(0).replace(",", ""): [g.lstrip("0") or "0" for g in m.group(0).split(",")]
            for m in _GROUPED_RE.finditer(text) if "000" not in m.group(0).split(",")}


# Values read off a plot are not in the paper's text by construction: the v2.1
# figures[].curves slots and any clause marked "approx., read from the figure"
# (the marking every prompt asks for). They are left out of the number check
# and checked against the figures by gold/figure_checks.jsonl instead.
_READ_FROM_FIGURE_RE = re.compile(r"(?:[^.;()]|\.(?=\d))*\(\s*approx[^)]*\)", re.IGNORECASE)


def _without_figure_readings(value) -> str:
    if isinstance(value, list):
        value = [{k: v for k, v in item.items() if k != "curves"} if isinstance(item, dict) else item for item in value]
    return _READ_FROM_FIGURE_RE.sub(" ", as_text(value))


def number_grounding(parsed: dict, paper_text: str, paper_fields: list = PAPER_FIELDS) -> dict:
    """
    Every number the answer states about the paper should appear in the paper.
    Single digits are skipped: they show up everywhere (section numbers, list
    items), so they'd count as "grounded" without proving anything.
    """
    latex = _is_latex(paper_text)
    cited, lists = set(), {}
    for k in paper_fields:
        text = _without_figure_readings(parsed.get(k))
        text = _drop_latex_structural_refs(text, paper_text) if latex else text
        cited |= _numbers(text)
        lists.update(_list_alternatives(text))
    cited = {n for n in cited if len(n.replace(".", "")) > 1}
    if not cited:
        return {"score": None, "ungrounded": []}
    # In LaTeX a number is often glued to a macro or a citation key
    # ("\sim0.14", "\citep{Smerald2014}"): split them so it can be found.
    source = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", paper_text) if latex else paper_text
    available = _numbers(source) | _notation_numbers(paper_text)
    ungrounded = sorted(n for n in cited if n not in available
                        and not (n in lists and all(g in available for g in lists[n])))
    return {"score": round(1 - len(ungrounded) / len(cited), 3), "ungrounded": ungrounded}


_ROMAN = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi", 7: "vii", 8: "viii", 9: "ix", 10: "x",
          11: "xi", 12: "xii", 13: "xiii", 14: "xiv", 15: "xv"}


_REF_RE = re.compile(r"\b(table|tab\.|figure|fig\.|algorithm|alg\.|eqs?\.|equations?)\s*\(?(\d+)\)?((?:\s*(?:-|–|,|and|to)\s*\(?\d+\)?)*)", re.IGNORECASE)


def _ref_found(kind: str, num: int, paper_lower: str) -> bool:
    kind = kind.lower()
    if kind.startswith("eq"):
        return f"({num})" in paper_lower
    if kind.startswith("tab"):
        names = ["table", "tab."]
    elif kind.startswith("fig"):
        names = ["figure", "fig.", "fig"]
    else:
        names = ["algorithm", "alg."]
    variants = [str(num)] + ([_ROMAN[num]] if num in _ROMAN else [])
    return any(re.search(rf"{re.escape(n)}\s*{v}\b", paper_lower) for n in names for v in variants)


# A LaTeX source never prints "Figure 3": it has \\begin{figure} environments
# referenced by \\ref{label}. There a cited number counts as grounded if the
# source has at least that many environments of the kind (numbering follows
# their order), so PDF and LaTeX inputs are scored on equal terms.
_LATEX_ENVS = {
    "fig": r"\\begin\{figure\*?\}",
    "tab": r"\\begin\{table\*?\}",
    "alg": r"\\begin\{algorithm\*?\}",
    "eq": r"\\begin\{(?:equation|align|eqnarray|multline|gather)\}|\\\[",
}


def _is_latex(paper_text: str) -> bool:
    return "--- SOURCE FILE:" in paper_text or "\\begin{document}" in paper_text


def _latex_ref_found(kind: str, num: int, paper_text: str) -> bool:
    key = next((k for k in _LATEX_ENVS if kind.lower().startswith(k)), "alg")
    return len(re.findall(_LATEX_ENVS[key], paper_text)) >= num


# Section / equation / figure / table / reference numbers ("Sec. 3.2", "Eq. (11)")
# are printed by the PDF but only generated when LaTeX compiles: in a source
# they are \section and \label/\ref. In a LaTeX input a structural reference is
# dropped from the number check when the source has enough objects of its kind;
# one pointing past them stays and is checked like any other number.
_STRUCT_REF_RE = re.compile(
    r"(?<!\w)(sections?|secs?\.|§|appendix|app\.|eqs?\.|equations?|figs?\.|figures?|tables?|tabs?\.|refs?\.|references?)"
    r"\s*\(?(\d+)(?:\.(\d+))?\)?((?:\s*(?:-|–|,|and|to)\s*\(?\d+(?:\.\d+)?\)?)*)", re.IGNORECASE)


_LATEX_SECTION_RE = re.compile(r"\\section\*?\{")


_LATEX_SUBSECTION_RE = re.compile(r"\\subsection\*?\{")


_LATEX_BIB_RE = re.compile(r"\\bibitem\b")


def _latex_structure_count(kind: str, paper_text: str) -> int:
    kind = kind.lower()
    if kind.startswith(("sec", "§", "app")):
        return len(_LATEX_SECTION_RE.findall(paper_text))
    if kind.startswith("ref"):
        # Without a compiled .bbl the bibliography size is unknown: trust the citation.
        return len(_LATEX_BIB_RE.findall(paper_text)) or 10**6
    key = next((k for k in _LATEX_ENVS if kind.startswith(k)), None)
    return len(re.findall(_LATEX_ENVS[key], paper_text)) if key else 0


def _drop_latex_structural_refs(text: str, paper_text: str) -> str:
    """text without the structural references a LaTeX source can back (see above)."""
    def keep_or_drop(m):
        kind, first, sub, rest = m.groups()
        available = _latex_structure_count(kind, paper_text)
        nums = [int(first)] + [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", rest)]
        if sub and kind.lower().startswith(("sec", "§", "app")) and int(sub) > max(len(_LATEX_SUBSECTION_RE.findall(paper_text)), 1):
            return m.group(0)
        return " " if all(n <= available for n in nums) else m.group(0)
    return _STRUCT_REF_RE.sub(keep_or_drop, text)


def reference_grounding(parsed: dict, paper_text: str, paper_fields: list = PAPER_FIELDS) -> dict:
    """Tables / figures / algorithms / equations the answer cites must exist in the paper."""
    paper_lower = paper_text.lower()
    latex = _is_latex(paper_text)
    refs = set()
    for k in paper_fields:
        for kind, first, rest in _REF_RE.findall(as_text(parsed.get(k))):
            for num in [first] + re.findall(r"\d+", rest):
                refs.add((kind.lower().rstrip("s."), int(num)))
    if not refs:
        return {"score": None, "ungrounded": []}
    ungrounded = sorted(f"{kind} {num}" for kind, num in refs
                        if not (_ref_found(kind, num, paper_lower) or latex and _latex_ref_found(kind, num, paper_text)))
    return {"score": round(1 - len(ungrounded) / len(refs), 3), "ungrounded": ungrounded}


def dataset_grounding(parsed: dict, paper_text: str) -> dict:
    """
    Each dataset name should appear in the paper. Matching is loose on purpose:
    the full name, an acronym in parentheses, or most of its significant words.
    """
    items = parsed.get("datasets_used")
    if items is None and isinstance(parsed.get("data_and_setup"), dict):
        items = parsed["data_and_setup"].get("datasets")  # v2
    if not isinstance(items, list):
        return {"score": None, "ungrounded": []}
    items = [str(i) for i in items if not is_empty_value(i)]
    if not items:
        return {"score": None, "ungrounded": []}
    paper_lower = paper_text.lower()
    ungrounded = []
    for item in items:
        name = item.lower().strip()
        if name in paper_lower:
            continue
        acronyms = re.findall(r"\(([^)]+)\)", name)
        if any(a.strip() and a.strip() in paper_lower for a in acronyms):
            continue
        words = [w for w in re.findall(r"[a-z0-9\-]+", re.sub(r"\([^)]*\)", "", name)) if len(w) >= 4]
        if words and sum(w in paper_lower for w in words) / len(words) >= 0.5:
            continue
        ungrounded.append(item)
    return {"score": round(1 - len(ungrounded) / len(items), 3), "ungrounded": ungrounded}


SCHEMA_EXAMPLES = _schema_examples(SYSTEM_PROMPT)


def schema_echo(parsed: dict) -> tuple[list, list]:
    """Fields with an item copied from the schema examples, and fields with a filler item."""
    echoed, filler = set(), set()
    for k in [k for k in EXPECTED_KEYS if k != "visual_analysis_findings"]:
        for slot, item in _keyed_leaves(parsed.get(k), k):
            text = " ".join(str(item).lower().split())
            if not text:
                continue
            if not re.search(r"[a-z0-9]", text):
                if slot not in _SYMBOL_SLOTS:
                    filler.add(k)
            elif any(SequenceMatcher(None, text, ex).ratio() >= ECHO_SIMILARITY for ex in SCHEMA_EXAMPLES):
                echoed.add(k)
    return sorted(echoed), sorted(filler)


def _repetition_text(parsed: dict) -> str:
    parts = []
    for k in PAPER_FIELDS:
        value = parsed.get(k)
        if isinstance(value, list):
            value = [{kk: vv for kk, vv in item.items() if kk != "curves"} if isinstance(item, dict) else item for item in value]
        parts.append(_FIGURE_MARKER_RE.sub(" ", as_text(value)))
    return "\n".join(parts)


def level1(raw_answer: str) -> dict:
    result = {"json_ok": False, "missing_keys": [], "extra_keys": [], "type_errors": [], "list_for_string": [],
              "schema_echo": [], "filler_fields": [], "empty_field_ratio": None, "repeated_ngram_ratio": None,
              "compression_ratio": None, "passed": False, "reasons": []}
    try:
        parsed = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        result["reasons"].append("invalid JSON")
        return result
    if not isinstance(parsed, dict):
        result["reasons"].append("JSON is not an object")
        return result
    result["json_ok"] = True
    result["missing_keys"] = [k for k in EXPECTED_KEYS if k not in parsed]
    result["extra_keys"] = [k for k in parsed if k not in EXPECTED_KEYS]
    for k, spec in FIELDS.items():
        if k in parsed:
            _check_v2(parsed[k], spec, k, result)
    for key in ("missing_keys", "extra_keys", "type_errors", "list_for_string"):
        result[key] = list(dict.fromkeys(result[key]))
    present = [k for k in PAPER_FIELDS if k in parsed]
    result["empty_field_ratio"] = round(sum(is_empty_value(parsed[k]) for k in present) / len(present), 3) if present else 1.0
    text = _repetition_text(parsed)
    result["repeated_ngram_ratio"] = round(repeated_ngram_ratio(text), 3)
    result["compression_ratio"] = round(compression_ratio(text), 3)
    result["schema_echo"], result["filler_fields"] = schema_echo(parsed)
    if result["missing_keys"]:
        result["reasons"].append(f"missing keys: {result['missing_keys']}")
    if result["type_errors"]:
        result["reasons"].append(f"wrong types: {result['type_errors']}")
    if result["empty_field_ratio"] > MAX_EMPTY_FIELD_RATIO:
        result["reasons"].append(f"too many empty fields ({result['empty_field_ratio']})")
    if result["repeated_ngram_ratio"] > MAX_REPEATED_NGRAM_RATIO_V2:
        result["reasons"].append(f"repetitive text (4-gram ratio {result['repeated_ngram_ratio']})")
    if result["compression_ratio"] < MIN_COMPRESSION_RATIO:
        result["reasons"].append(f"repetitive text (compression {result['compression_ratio']})")
    if result["schema_echo"]:
        result["reasons"].append(f"schema text copied instead of an answer: {result['schema_echo']}")
    if result["filler_fields"]:
        result["reasons"].append(f"filler items: {result['filler_fields']}")
    result["passed"] = not result["reasons"]
    return result


def level2(raw_answer: str, paper_text: str) -> dict:
    try:
        parsed = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        return {"score": None}
    if not isinstance(parsed, dict):
        return {"score": None}
    parts = {"numbers": number_grounding(parsed, paper_text, PAPER_FIELDS),
             "references": reference_grounding(parsed, paper_text, PAPER_FIELDS),
             "datasets": dataset_grounding(parsed, paper_text)}
    scores = [p["score"] for p in parts.values() if p["score"] is not None]
    # visual_analysis_findings is reserved for external curve fits, which these runs never have:
    # a non-empty one is invented and scores 0.
    if not is_empty_value(parsed.get("visual_analysis_findings") or []):
        scores.append(0.0)
    parts["score"] = round(sum(scores) / len(scores), 3) if scores else None
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs/pdf")
    ap.add_argument("--out-dir", default=None, help="Default: <runs-dir>/json_checks")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.runs_dir, "json_checks")
    os.makedirs(out_dir, exist_ok=True)
    name_re = re.compile(r"^([a-z0-9]+)_(think|nothink)_s(\d+)_B1$")
    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*_B1.jsonl"))):
        m = name_re.match(os.path.basename(path)[:-6])
        if not m:
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            answer = [x["content"] for x in rec["messages"] if x["role"] == "assistant"][-1]
            paper = [x["content"] for x in rec["messages"] if x["role"] == "user"][-1]
            l1, l2 = level1(answer), level2(answer, paper)
            rows.append({"model": m.group(1), "config": m.group(2), "seed": int(m.group(3)),
                         "arxiv_id": re.search(r"(\d{4}\.\d{4,5})", rec["source_file"]).group(1),
                         "level1_passed": int(l1["passed"]), "level1_reasons": "; ".join(l1["reasons"]),
                         "empty_field_ratio": l1["empty_field_ratio"], "repeated_ngram_ratio": l1["repeated_ngram_ratio"],
                         "compression_ratio": l1["compression_ratio"], "level2": l2.get("score"),
                         **{f"{c}_grounded": (l2.get(c) or {}).get("score") for c in ("numbers", "references", "datasets")},
                         **{f"ungrounded_{c}": "; ".join((l2.get(c) or {}).get("ungrounded", [])) for c in ("numbers", "references", "datasets")},
                         **{f"n_ungrounded_{c}": len((l2.get(c) or {}).get("ungrounded", [])) for c in ("numbers", "references")}})
    write_csv(os.path.join(out_dir, "per_json.csv"), rows)

    summary = []
    for key in sorted({(r["model"], r["config"]) for r in rows}, key=lambda k: (model_order(k[0]), CONFIGS.index(k[1]))):
        rs = [r for r in rows if (r["model"], r["config"]) == key]
        by_p = defaultdict(list)
        for r in rs:
            if r["level2"] is not None:
                by_p[r["arxiv_id"]].append(r["level2"])
        pm = {p: statistics.mean(v) for p, v in by_p.items()}
        rng, papers = random.Random(0), sorted(pm)
        boots = sorted(statistics.mean(pm[rng.choice(papers)] for _ in papers) for _ in range(args.n_boot)) if pm else [None]

        def mean(c, digits=3, scale=1):
            vals = [r[c] for r in rs if r[c] is not None]   # e.g. no dataset named in any JSON: nothing to check
            return round(scale * statistics.mean(vals), digits) if vals else ""
        summary.append({"model": key[0], "model_name": MODELS.get(key[0], key[0]), "config": key[1], "jsons": len(rs),
                        "level1_passed": sum(r["level1_passed"] for r in rs),
                        "empty_fields_percent": mean("empty_field_ratio", 1, 100),
                        "repeated_ngram_ratio": mean("repeated_ngram_ratio"), "compression_ratio": mean("compression_ratio"),
                        "level2": round(statistics.mean(pm.values()), 3) if pm else "",
                        "level2_ci_low": round(boots[int(0.025 * len(boots))], 3) if pm else "",
                        "level2_ci_high": round(boots[max(int(0.975 * len(boots)) - 1, 0)], 3) if pm else "",
                        **{f"{c}_grounded": mean(f"{c}_grounded") for c in ("numbers", "references", "datasets")},
                        "ungrounded_numbers_per_json": round(statistics.mean(r["n_ungrounded_numbers"] for r in rs), 2),
                        "ungrounded_references_per_json": round(statistics.mean(r["n_ungrounded_references"] for r in rs), 2)})
    write_csv(os.path.join(out_dir, "summary.csv"), summary)


if __name__ == "__main__":
    main()
