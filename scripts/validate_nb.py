"""Print cell counts and section headers of the analysis notebook."""

import json
import sys

NB = sys.argv[1] if len(sys.argv) > 1 else "notebooks/accent_gop_notebook.ipynb"


def source(cell):
    # nbformat allows a string or a list of lines (Jupyter saves lists)
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


with open(NB, encoding="utf-8") as f:
    nb = json.load(f)
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]

print("Total cells :", len(nb["cells"]))
print("Code cells  :", len(code_cells))
print("Markdown    :", len(md_cells))
print()

for c in md_cells:
    text = source(c)
    if text.startswith("#"):
        print(" ", text[:90].replace("\n", " "))
