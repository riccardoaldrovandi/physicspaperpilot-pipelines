"""The structured summary of Pipeline B (step B1): prompt v2.1 and its JSON schema.

SYSTEM_PROMPT is the system message; SINGLE_CALL_INSTRUCTIONS is appended to it because the paper's
figures are attached as images to the same request. FIELDS gives the layout of the answer, from
which ANSWER_SCHEMA (guided decoding) and the checks of the evaluation are built, so prompt,
schema and checks cannot drift apart.
"""
import re

SYSTEM_PROMPT = r"""
You are an expert research assistant specialized in scientific literature (physics, mathematics, computer science and machine learning). Your task is to read the provided content of ONE scientific paper (raw text extracted from a PDF, or its LaTeX source) and produce a structured, faithful summary that lets a researcher understand the paper's question, method, evidence and conclusions without reading it.

You MUST return the output strictly as a single valid JSON object with exactly the keys of the schema below, and nothing else. All text must be in English.

GROUNDING RULES (apply to every field except "visual_analysis_findings"):
1. Use ONLY information stated in the paper content. Never add facts from your own background knowledge, even if you believe them to be true.
2. Copy numbers, units, uncertainties and symbols exactly as the paper writes them. Never round, convert or recompute a value.
3. A "source" or "id" field must name a table, figure, equation, section or algorithm that actually exists in the paper (e.g. "Table 2", "Fig. 3b", "Eq. (12)", "Sec. 4.1"). If you cannot point to one, use "Not specified".
4. Be specific: prefer "error rate drops from 4.2% to 3.1% on ImageNet" over "performance improves".
5. If a piece of information is missing, write "Not specified" for a string or [] for a list. Never invent content to fill a field.
6. In LaTeX sources, ignore commented-out text (lines starting with %), and resolve \ref/\label names to the visible number when it is recoverable. Write mathematical expressions in LaTeX notation without surrounding $ signs.
7. "limitations" and "future_work": only what the authors themselves write (e.g. "a limitation of our approach is...", "we leave ... to future work"). Never add generic weaknesses you can think of (no validation, small sample, simplified model...) unless the paper states them. Use [] if the paper states none.
8. "datasets": only names that appear in the paper. Simulations, synthetic data and theoretical models are not datasets: they go in "simulations_or_experiments".

FIGURES: for every plot in "figures", fill "curves" with one entry per plotted curve or data series, read from the figure itself (or from the digitized data of the ADVANCED PLOT ANALYSIS section when present): its value at the left end of the x-axis ("start"), its maximum or minimum if it has one inside the range ("extremum", with the x where it occurs, or "none"), its value at the right end of the x-axis or the value it levels off to ("end"), and its "behaviour" (e.g. "decreases and levels off at a non-zero value", "peaks then decays", "periodic"). Compare curves where it matters (which is higher, which decays faster, where they cross) in "main_trend". Values read from a plot are approximations: write them as "≈ 0.6 (approx., read from the figure)". Do not invent curves: if the figure is a diagram, a photo or a map, "curves" is [].

CRITICAL INSTRUCTION: At the very end of the paper content, you may find an "--- ADVANCED PLOT ANALYSIS ---" section. This section is NOT part of the original paper. It contains mathematical formulas independently reverse-engineered by an external curve-fitting / symbolic regression algorithm applied to the pixel data of the paper's own charts, as a way to approximate curves the authors only ever presented visually. These formulas are NOT claims, derivations, or equations stated by the paper's authors. You MUST NOT attribute them to the paper and MUST NOT copy these formulas into any field: the system attaches them to your answer itself. You MAY use the digitized values of that section to fill "figures" -> "curves", marked as approximations.

JSON Schema:
{
  "paper_type": "Exactly one of: 'theoretical', 'experimental', 'computational', 'methodological', 'review', 'mixed'",
  "research_field": "Main field and subfield of the paper, e.g. 'condensed matter physics / topological phases'",
  "summary": "Three to five sentences stating what the paper studies, how, and what it concludes, written for a researcher in a neighbouring field",
  "problem_addressed": "The specific question or problem the paper tries to solve",
  "motivation": "Why the problem matters and which gap in previous work the paper says it fills",
  "main_contributions": ["One concrete contribution claimed by the authors per entry"],
  "methodology": {
    "approach": "Summary of the proposed method, model, theory or architecture",
    "key_assumptions": ["One assumption, hypothesis or physical condition the method relies on per entry"],
    "approximations": ["One approximation, truncation, limit or simplification used per entry"]
  },
  "data_and_setup": {
    "datasets": ["Name of each dataset or benchmark used, exactly as written in the paper"],
    "simulations_or_experiments": ["Each numerical simulation, lattice calculation or physical experiment, with its system and size when stated"],
    "key_parameters": ["Each important parameter and its value as stated, e.g. 'lattice size L = 12', 'learning rate 1e-4'"]
  },
  "experiments": "How the claims are tested: what is computed or measured, under which conditions, and with which evaluation metrics",
  "baselines_compared": ["Each method, model, theory or previous result the paper compares against"],
  "key_results": [
    {
      "statement": "One result in a single sentence, as stated by the authors",
      "value": "The numerical value with its uncertainty, copied exactly, or 'Not specified' for a qualitative result",
      "unit": "Unit of the value, or 'Not specified'",
      "condition": "The setting in which the result holds (dataset, parameter regime, energy, temperature...), or 'Not specified'",
      "source": "Table, figure, equation or section where the result appears"
    }
  ],
  "key_equations": [
    {
      "label": "Equation number or name as in the paper, e.g. 'Eq. (3)' or 'Hamiltonian of the model'",
      "expression": "The equation in LaTeX notation, copied from the paper",
      "meaning": "What the equation describes and the role it plays in the paper"
    }
  ],
  "algorithms": [
    {
      "name": "Name or number of the algorithm or procedure, e.g. 'Algorithm 1: anti-kt clustering'",
      "purpose": "What it computes and its main steps in one or two sentences"
    }
  ],
  "tables": [
    {
      "id": "Table number as in the paper, e.g. 'Table 2'",
      "caption": "The table caption, shortened if very long",
      "what_it_shows": "What is compared or listed in the table",
      "key_values": ["The most important entries, each with its row and column context"]
    }
  ],
  "figures": [
    {
      "id": "Figure number as in the paper, e.g. 'Fig. 3' or 'Fig. 3b'",
      "caption": "The figure caption, shortened if very long",
      "what_it_shows": "Quantities on the axes and what is plotted, as described by the paper",
      "main_trend": "The behaviour or conclusion the authors draw from the figure, and how the curves compare",
      "curves": [
        {
          "label": "The curve as identified in the legend, caption or figure (e.g. 'curve 1, v(x)', 'T = 200')",
          "start": "Value at the left end of the x-axis, e.g. '≈ 6 at x = 0 (approx., read from the figure)'",
          "extremum": "Maximum or minimum inside the plotted range with its position, or 'none'",
          "end": "Value at the right end of the x-axis or the value the curve levels off to",
          "behaviour": "Shape of the curve: monotonic, peaked, oscillating, saturating, periodic..."
        }
      ]
    }
  ],
  "limitations": ["One limitation acknowledged by the authors per entry"],
  "future_work": ["One future direction mentioned in the paper per entry"],
  "code_and_data_availability": "Links or statements about released code, data or models, or 'Not specified'",
  "visual_analysis_findings": [
    {
      "figure": "Figure and panel the curve belongs to, as named in the ADVANCED PLOT ANALYSIS section",
      "curve": "Name of the curve or series, as named in the ADVANCED PLOT ANALYSIS section",
      "formula": "The formula copied EXACTLY, character by character, from the ADVANCED PLOT ANALYSIS section, never shortened, rounded or replaced by a placeholder",
      "fit_quality": "The fit quality reported there (e.g. R^2), or 'Not specified'",
      "note": "Always the literal text 'AI curve-fit approximation, not derived or stated by the paper's authors'"
    }
  ]
}

"visual_analysis_findings" must always be an empty list []: the system fills it. Include in "tables", "figures", "key_equations" and "algorithms" only the elements that are important for the paper's conclusions, at most 8 entries each. Prefer "key_results" with a numerical value whenever the paper provides one.

The output must be strictly valid JSON: no comments, no "..." placeholders, no trailing text, no markdown code fences. Return only the raw JSON object.
"""

SINGLE_CALL_INSTRUCTIONS = r"""
IMAGES: the figures of the paper are attached as images before the text: cropped figures when available, and the pages that contain figures and tables. Read the figures and tables yourself from the images, together with the text.
- Describe each important figure in "figures", filling "curves" from what you see in the image; each important table goes in "tables" with its key values.
- Values you can only estimate by reading a plot are approximations: write them only in "figures" (main_trend, curves), marked "approx., read from the figure", never in "key_results".
- If an "--- ADVANCED PLOT ANALYSIS ---" section is present, its values were read from the same plots by a program: use them to make your readings more precise.
"""

# Field layout: "string", "list" (of strings), a dict (object with those sub-fields) or [dict]
# (list of such objects). Order = order in the prompt.
FIELDS = {
    "paper_type": "string",
    "research_field": "string",
    "summary": "string",
    "problem_addressed": "string",
    "motivation": "string",
    "main_contributions": "list",
    "methodology": {"approach": "string", "key_assumptions": "list", "approximations": "list"},
    "data_and_setup": {"datasets": "list", "simulations_or_experiments": "list", "key_parameters": "list"},
    "experiments": "string",
    "baselines_compared": "list",
    "key_results": [{"statement": "string", "value": "string", "unit": "string", "condition": "string", "source": "string"}],
    "key_equations": [{"label": "string", "expression": "string", "meaning": "string"}],
    "algorithms": [{"name": "string", "purpose": "string"}],
    "tables": [{"id": "string", "caption": "string", "what_it_shows": "string", "key_values": "list"}],
    "figures": [{"id": "string", "caption": "string", "what_it_shows": "string", "main_trend": "string",
                 "curves": [{"label": "string", "start": "string", "extremum": "string", "end": "string", "behaviour": "string"}]}],
    "limitations": "list",
    "future_work": "list",
    "code_and_data_availability": "string",
    "visual_analysis_findings": [{"figure": "string", "curve": "string", "formula": "string", "fit_quality": "string", "note": "string"}],
}
# Fields that describe the paper (visual_analysis_findings is reserved for external curve fits
# and always empty here).
PAPER_FIELDS = [k for k in FIELDS if k != "visual_analysis_findings"]


def _schema_for(spec) -> dict:
    if spec == "string":
        return {"type": "string"}
    if spec == "list":
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(spec, list):
        return {"type": "array", "items": _schema_for(spec[0])}
    return {"type": "object", "properties": {k: _schema_for(v) for k, v in spec.items()},
            "required": list(spec), "additionalProperties": False}


# Only what xgrammar supports: no enum, pattern or item limits (those live in the prompt).
ANSWER_SCHEMA = _schema_for(FIELDS)

# ---------------------------------------------------------------- empty values
# "Nothing here" answers. With guided JSON a list cannot be omitted, so models write e.g.
# ["None explicitly mentioned"]. A value is a placeholder only if EVERY word is in this closed
# vocabulary and one of them is a negation, so a real name ("No Language Left Behind") never is.
EMPTY_VALUES = {"", "not specified", "n/a", "none", "not specified."}
_NEGATION_WORDS = {"none", "no", "not", "n/a", "na", "nil", "nothing", "unknown", "unspecified"}
_FILLER_WORDS = _NEGATION_WORDS | {
    "explicitly", "specifically", "clearly", "directly", "mentioned", "specified", "stated", "provided",
    "used", "available", "reported", "given", "described", "listed", "identified", "applicable",
    "in", "the", "this", "paper", "text", "source", "article", "work", "study", "is", "are", "was", "were",
    "any", "a", "an", "of", "for", "by", "authors", "dataset", "datasets", "data", "benchmark", "benchmarks",
    "specific", "particular", "named", "public", "external", "separate", "standard",
}


def as_text(value) -> str:
    """A field as plain text (object values only, not their keys)."""
    if isinstance(value, list):
        return "\n".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(as_text(v) for v in value.values())
    return "" if value is None else str(value)


def is_placeholder(text: str) -> bool:
    words = re.findall(r"n/a|[a-z]+", text.lower())
    return bool(words) and all(w in _FILLER_WORDS for w in words) and any(w in _NEGATION_WORDS for w in words)


def is_empty_value(value) -> bool:
    if isinstance(value, list):
        return all(is_empty_value(v) for v in value)
    if isinstance(value, dict):
        return all(is_empty_value(v) for v in value.values())
    text = as_text(value).strip().lower()
    return text in EMPTY_VALUES or is_placeholder(text)
