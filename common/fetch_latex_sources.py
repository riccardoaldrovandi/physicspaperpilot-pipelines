"""LaTeX sources of the PhysicsPaperPilot papers, for the runs with SOURCE=latex.

For every PDF in --pdf-dir (file name starting with the arXiv id), downloads the arXiv source of the
version listed in the package's papers.csv (the same version as the PDF) and writes:
  <latex-dir>/<PDF name without .pdf>_source.tar.gz
      the arXiv archive; a source submitted as a single gzipped .tex is repacked as a tar.gz with one
      file (main.tex), because src/processing/latex_extractor.py reads tar archives only;
  <figures-dir>/<PDF name without .pdf>_source/
      the figure files of the archive as PNG, flat (what main.py --figures-dir expects for LaTeX):
      .png/.jpg/.jpeg copied, .pdf rendered with PyMuPDF (300 dpi, first page, as
      extract_images_from_latex does), .eps/.ps converted with Ghostscript (-dSAFER -dEPSCrop, 300 dpi),
      which the pipeline cannot do by itself;
  <latex-dir>/manifest.csv
      one row per paper: version, kind of source, sha256, .tex files, figures found / converted / failed.
Papers whose source is only a PDF are reported and get no archive. Downloads are cached: an archive
already present is not fetched again (--refetch to force). Run from the repository root:
  python common/fetch_latex_sources.py
"""
import argparse
import csv
import glob
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

import pymupdf

ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})")
IMAGE_EXT = (".png", ".jpg", ".jpeg")
VECTOR_EXT = (".pdf", ".eps", ".ps")
USER_AGENT = "paper_analyzer-benchmark/1.0 (thesis research; one request every few seconds)"


def versions(benchmark_dir):
    with open(os.path.join(benchmark_dir, "papers.csv"), encoding="utf-8-sig", newline="") as f:
        return {r["arxiv_id"].strip(): r["version"].strip() for r in csv.DictReader(f)}


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def to_tar_gz(data):
    """(tar.gz bytes, kind) from what /e-print/ returns; (None, kind) when there is no LaTeX source."""
    if data[:4] == b"%PDF":
        return None, "pdf_only"
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            tar.getmembers()
        if data[:2] == b"\x1f\x8b":
            return data, "tar.gz"
        return gzip.compress(raw), "tar"
    except tarfile.TarError:
        pass
    text = raw.decode("utf-8", errors="replace")
    if "\\documentclass" in text or "\\begin{document}" in text:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo("main.tex")
            info.size, info.mtime = len(raw), int(time.time())
            tar.addfile(info, io.BytesIO(raw))
        return buf.getvalue(), "single_tex"
    return None, "unknown"


def safe_members(tar):
    """Regular files only, no absolute paths or '..' (the archives are untrusted)."""
    for m in tar.getmembers():
        parts = m.name.replace("\\", "/").split("/")
        if m.isfile() and not m.name.startswith("/") and ".." not in parts:
            yield m


def prepare_figures(tar_path, out_dir):
    """PNG figures of the archive in out_dir (flat). Returns (found, converted, failed names)."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    found, converted, failed = 0, 0, []
    used = set()
    with tarfile.open(tar_path, "r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
        for m in safe_members(tar):
            low = m.name.lower()
            if not low.endswith(IMAGE_EXT + VECTOR_EXT):
                continue
            found += 1
            stem, ext = os.path.splitext(os.path.basename(m.name))
            # Flat directory: a name used twice (fig1.pdf in two folders) keeps its folder as prefix.
            target = stem if stem not in used else re.sub(r"[^\w.-]", "_", os.path.dirname(m.name)) + "_" + stem
            used.add(target)
            png = os.path.join(out_dir, target + (ext.lower() if low.endswith(IMAGE_EXT) else ".png"))
            src = os.path.join(tmp, f"in{found}{ext.lower()}")
            with open(src, "wb") as f:
                f.write(tar.extractfile(m).read())
            try:
                if low.endswith(IMAGE_EXT):
                    shutil.copyfile(src, png)
                elif low.endswith(".pdf"):
                    doc = pymupdf.open(src)
                    doc[0].get_pixmap(dpi=300).save(png)
                    doc.close()
                else:
                    subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop", "-sDEVICE=png16m",
                                    "-r300", "-dFirstPage=1", "-dLastPage=1", f"-sOutputFile={png}", src],
                                   check=True, capture_output=True, timeout=120)
                if not os.path.exists(png) or os.path.getsize(png) == 0:
                    raise RuntimeError("no output")
                converted += 1
            except Exception as e:
                failed.append(f"{m.name} ({type(e).__name__})")
                if os.path.exists(png):
                    os.remove(png)
    return found, converted, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark-dir", default="benchmark/PhysicsPaperPilotEval_v0.3.2", help="PhysicsPaperPilot package (papers.csv gives the versions)")
    ap.add_argument("--pdf-dir", default="papers/pdf")
    ap.add_argument("--latex-dir", default="papers/latex")
    ap.add_argument("--figures-dir", default="papers/latex_figures")
    ap.add_argument("--refetch", action="store_true", help="Download again even if the archive exists")
    ap.add_argument("--delay", type=float, default=5.0, help="Seconds between arXiv requests")
    args = ap.parse_args()
    if not shutil.which("gs"):
        raise SystemExit("[!] Ghostscript (gs) not found: needed for .eps figures")
    vers = versions(args.benchmark_dir)
    os.makedirs(args.latex_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    manifest, first = [], True
    for pdf in sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf"))):
        base = os.path.basename(pdf)[:-4]
        m = ARXIV_ID_RE.match(base)
        if not m or m.group(1) not in vers:
            print(f"[!] {base}: arXiv id not in papers.csv, skipped")
            continue
        aid, ver = m.group(1), vers[m.group(1)]
        out = os.path.join(args.latex_dir, base + "_source.tar.gz")
        row = {"arxiv_id": aid, "version": ver, "archive": os.path.basename(out)}
        if args.refetch or not os.path.exists(out):
            if not first:
                time.sleep(args.delay)
            first = False
            url = f"https://arxiv.org/e-print/{aid}{ver}"
            print(f"[~] {aid}{ver}: {url}")
            data, kind = to_tar_gz(download(url))
            if data is None:
                print(f"[!] {aid}{ver}: no LaTeX source ({kind})")
                manifest.append({**row, "archive": "", "kind": kind})
                continue
            with open(out, "wb") as f:
                f.write(data)
        else:
            kind = "cached"
        with open(out, "rb") as f:
            row["sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
        with tarfile.open(out, "r:gz") as tar:
            row["tex_files"] = sum(1 for x in safe_members(tar) if x.name.endswith(".tex"))
        found, converted, failed = prepare_figures(out, os.path.join(args.figures_dir, base + "_source"))
        row.update({"kind": kind, "figures_found": found, "figures_png": converted, "figures_failed": "; ".join(failed)})
        manifest.append(row)
        print(f"[+] {aid}{ver} ({kind}): {row['tex_files']} .tex, figures {converted}/{found}"
              + (f", failed: {row['figures_failed']}" if failed else ""))
    path = os.path.join(args.latex_dir, "manifest.csv")
    fields = list(dict.fromkeys(k for r in manifest for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    print(f"[+] {path}: {len(manifest)} paper(s), {sum(1 for r in manifest if r.get('archive'))} with a LaTeX source")


if __name__ == "__main__":
    sys.exit(main())
