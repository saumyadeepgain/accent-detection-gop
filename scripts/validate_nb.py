import json

nb = json.load(open("accent_gop_notebook.ipynb"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
md_cells   = [c for c in nb["cells"] if c["cell_type"] == "markdown"]

print("Total cells :", len(nb["cells"]))
print("Code cells  :", len(code_cells))
print("Markdown    :", len(md_cells))
print()

headers = [
    c["source"][:90].replace("\n", " ")
    for c in nb["cells"]
    if c["cell_type"] == "markdown" and c["source"].startswith("#")
]
for h in headers:
    print(" ", h)
