"""Assemble notebooks/CA2_Predictive_Analysis.ipynb from the cell files in nb_src/ (jupytext-style '# %%' markers)."""
import re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
files = sorted((ROOT / "nb_src").glob("0*.py"))
nb = nbf.v4.new_notebook(); cells = []
for f in files:
    chunks = re.split(r"^# %%", f.read_text(), flags=re.M)[1:]
    for ch in chunks:
        first, _, body = ch.partition("\n")
        if first.strip() == "[markdown]":
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.rstrip().split("\n"))
            cells.append(nbf.v4.new_markdown_cell(md))
        else:
            cells.append(nbf.v4.new_code_cell(body.strip("\n")))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "CA2_Predictive_Analysis.ipynb"
nbf.write(nb, out); print("wrote", out, "with", len(cells), "cells")
