"""What the model receives for a paper: its text and its figures as images, from the PDF or from
the arXiv LaTeX source. Pipeline A and the JSON step of Pipeline B use exactly the same inputs.

PDF:   the text of every page (PyMuPDF), and the pages holding figures or tables rendered whole.
LaTeX: the .tex files of the archive, and the figure files already converted to PNG by
       common/fetch_latex_sources.py, in the order the source includes them.
At most MAX_IMAGES images (env SINGLE_CALL_MAX_IMAGES, default 12); each counts as
IMAGE_TOKENS tokens (env SINGLE_CALL_IMAGE_TOKENS, default 2500) when fitting the context.
"""
import glob
import os
import re
import tarfile

import pymupdf

MAX_IMAGES = int(os.environ.get("SINGLE_CALL_MAX_IMAGES", "12"))
IMAGE_TOKENS = int(os.environ.get("SINGLE_CALL_IMAGE_TOKENS", "2500"))
MAX_SIDE_PX = 1540   # longest side of a rendered page (Pixtral's limit; other models resize anyway)
_CAPTION_RE = re.compile(r"\b(Fig\.|Figure|FIG\.|Table|TABLE)\s*([IVX]+|\d+)")
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def arxiv_id_of(path: str) -> str:
    m = _ARXIV_ID_RE.search(os.path.basename(path))
    return m.group(1) if m else ""


def paper_name(path: str) -> str:
    """File name without .pdf / .tar.gz."""
    name = os.path.basename(path)
    for ext in (".tar.gz", ".pdf"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def pdf_text(pdf_path: str) -> str:
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def latex_text(tar_path: str) -> str:
    """Every .tex file of the archive, each preceded by its name (archive order)."""
    parts = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".tex"):
                raw = tar.extractfile(member).read()
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw.decode("latin-1")
                parts.append(f"--- SOURCE FILE: {member.name} ---\n{content}\n")
    return "\n".join(parts)


def _visual_score(page) -> int:
    """How much of a figure / table page this is: captions, images and long drawn paths."""
    long_paths = sum(1 for d in page.get_drawings() if len(d["items"]) >= 5)
    return 3 * len(_CAPTION_RE.findall(page.get_text())) + 2 * len(page.get_images()) + min(long_paths, 20)


def pdf_figure_pages(pdf_path: str, out_dir: str, max_images: int = MAX_IMAGES) -> list:
    """PNG renders of the pages with figures or tables (in page order), cached in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        scored = [(i, _visual_score(p)) for i, p in enumerate(doc)]
        keep = sorted(i for i, s in sorted(scored, key=lambda t: -t[1])[:max_images] if s > 0)
        paths = []
        for i in keep:
            path = os.path.join(out_dir, f"page{i + 1}.png")
            if not os.path.exists(path):
                page = doc[i]
                zoom = MAX_SIDE_PX / max(page.rect.width, page.rect.height)
                page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(path)
            paths.append(path)
    return paths


def latex_figure_files(tex_text: str, figures_dir: str, max_images: int = MAX_IMAGES) -> list:
    """The PNG/JPG figures in figures_dir, ordered as the LaTeX source includes them."""
    files = sorted(p for p in glob.glob(os.path.join(figures_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))
    order = [os.path.splitext(os.path.basename(m))[0] for m in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", tex_text)]
    rank = {name: i for i, name in reversed(list(enumerate(order)))}
    files.sort(key=lambda p: rank.get(os.path.splitext(os.path.basename(p))[0], len(order)))
    return files[:max_images]


def paper_inputs(source: str, kind: str, figures_dir: str = None, cache_dir: str = "cache") -> tuple:
    """(text, image paths) of one paper. kind: "pdf" or "latex"; figures_dir: required for LaTeX
    (the PNGs of common/fetch_latex_sources.py, in <figures_dir>/<paper name>/)."""
    name = paper_name(source)
    if kind == "pdf":
        text = pdf_text(source)
        images = pdf_figure_pages(source, os.path.join(cache_dir, "pages", name))
    elif kind == "latex":
        fig_dir = os.path.join(figures_dir or "", name)
        if not figures_dir or not os.path.isdir(fig_dir):
            raise FileNotFoundError(f"No converted figures in '{fig_dir}': run common/fetch_latex_sources.py")
        text = latex_text(source)
        images = latex_figure_files(text, fig_dir)
    else:
        raise ValueError(f"kind must be 'pdf' or 'latex', not {kind!r}")
    print(f"[+] {name}: {len(text)} characters, {len(images)} image(s).")
    return text, images


def find_sources(kind: str, pdf_dir: str, latex_dir: str) -> dict:
    """arXiv id -> paper file (.pdf in pdf_dir, or _source.tar.gz in latex_dir)."""
    pattern = os.path.join(pdf_dir, "*.pdf") if kind == "pdf" else os.path.join(latex_dir, "*.tar.gz")
    return {arxiv_id_of(p): p for p in sorted(glob.glob(pattern))}
