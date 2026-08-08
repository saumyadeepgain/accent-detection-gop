"""Build Sections C + D — Model Architecture (13-15) + Training (16-19)."""
import json, uuid

NB = "accent_gop_notebook.ipynb"
def cell(src, cell_type="code"):
    return {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": src, **({"outputs": [], "execution_count": None} if cell_type == "code" else {})}
def md(src): return cell(src, "markdown")

cells = []

# ─── SECTION C: MODEL ARCHITECTURE ──────────────────────────────────────────
cells.append(md("---\n## 4 · Model Architecture <a id='4'></a>"))

# Graph 13 – Full system architecture
cells.append(md("### Graph 13 — Full System Architecture Diagram"))
cells.append(cell("""\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(15, 7))
ax.set_xlim(0, 15); ax.set_ylim(0, 7); ax.axis("off")

def box(ax, x, y, w, h, label, color, fontsize=9):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.12", fc=color, ec="white", lw=1.8, alpha=0.9))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", multialignment="center")

def arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    if label:
        ax.text((x0+x1)/2, (y0+y1)/2+0.15, label, ha="center", fontsize=7.5, color="dimgray")

# Main pipeline (top row)
box(ax, 0.2, 4.8, 2.0, 1.1, "Raw Audio\\n(16 kHz)", "#4C72B0")
box(ax, 2.8, 4.8, 2.2, 1.1, "WavLM-Large\\n(Frozen)", "#8172B3")
box(ax, 5.6, 4.8, 2.0, 1.1, "Layer 6+9\\nFeatures\\n2048-d", "#55A868")
box(ax, 8.2, 4.8, 2.0, 1.1, "Classifier\\nHead\\n(MLP)", "#DD8452")
box(ax, 11.0, 4.8, 2.2, 1.1, "Accent\\nPrediction\\n(7 classes)", "#C44E52")

# Language-ID branch (bottom row)
box(ax, 2.8, 2.6, 2.2, 1.1, "Voxlingua107\\nLang-ID\\n(ECAPA-TDNN)", "#937860")
box(ax, 5.6, 2.6, 2.0, 1.1, "107-d\\nLang Logits", "#55A868", fontsize=8.5)

# GoP branch (bottom)
box(ax, 2.8, 0.4, 2.2, 1.1, "wav2vec2-xlsr\\nPhone Post.", "#DA8BC3")
box(ax, 5.6, 0.4, 2.0, 1.1, "MFA\\nAlignment", "#4C72B0")
box(ax, 8.2, 0.4, 2.0, 1.1, "GoP Score\\nper Phoneme", "#C44E52")

# Feature concat
box(ax, 8.2, 2.6, 2.0, 1.1, "Concat\\n(2048+256+107)", "#937860", fontsize=8)

# Arrows — main
arrow(ax, 2.2, 5.35, 2.8, 5.35, "waveform")
arrow(ax, 5.0, 5.35, 5.6, 5.35, "embed")
arrow(ax, 7.6, 5.35, 8.2, 5.35)
arrow(ax, 10.2, 5.35, 11.0, 5.35)
# lang-id branch
arrow(ax, 2.2, 5.0, 2.9, 3.7)
arrow(ax, 5.0, 3.15, 5.6, 3.15)
arrow(ax, 7.6, 3.15, 8.2, 3.15)
arrow(ax, 9.5, 3.1, 9.5, 4.8)
# GoP branch
arrow(ax, 2.2, 5.0, 2.9, 0.95)
arrow(ax, 5.0, 0.95, 5.6, 0.95)
arrow(ax, 7.6, 0.95, 8.2, 0.95)

ax.set_title("Full System Architecture — Accent Detection + GoP Pipeline",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(); plt.show()
"""))

# Graph 14 – Classifier head
cells.append(md("### Graph 14 — Classifier Head Architecture (Layer Dimensions)"))
cells.append(cell("""\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 3.5); ax.axis("off")

layers_info = [
    ("Input\\n2048+256+107\\n= 2411-d", 0.5, "#4C72B0"),
    ("Linear\\n2411→512\\n+ LayerNorm", 2.4, "#8172B3"),
    ("GELU\\nActivation", 4.3, "#55A868"),
    ("Dropout\\np=0.3", 6.2, "#937860"),
    ("Linear\\n512→128", 8.1, "#DD8452"),
    ("GELU", 10.0, "#55A868"),
    ("Linear\\n128→7\\n(Softmax)", 11.9, "#C44E52"),
]
for label, x, color in layers_info:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.75, 0.7), 1.5, 2.1,
                 boxstyle="round,pad=0.12", fc=color, ec="white", lw=2, alpha=0.88))
    ax.text(x, 1.75, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", multialignment="center")

for i in range(len(layers_info)-1):
    x0 = layers_info[i][1]+0.76
    x1 = layers_info[i+1][1]-0.76
    ax.annotate("", xy=(x1, 1.75), xytext=(x0, 1.75),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

ax.set_title("Classifier Head Architecture — Exact Layer Dimensions",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# Graph 15 – GoP pipeline diagram
cells.append(md("### Graph 15 — Goodness-of-Pronunciation (GoP) Pipeline Diagram"))
cells.append(cell("""\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(15, 3.8))
ax.set_xlim(0, 15); ax.set_ylim(0, 3.8); ax.axis("off")

stages = [
    ("Audio +\\nRef. Text", 0.7,  "#4C72B0"),
    ("Montreal\\nForced\\nAligner", 2.7,  "#8172B3"),
    ("Phoneme\\nBoundaries\\n(start/end ms)", 4.9, "#937860"),
    ("wav2vec2-xlsr\\nPhone\\nPosteriors", 7.1, "#55A868"),
    ("Frame-level\\nP(phoneme|frame)\\nper segment", 9.5, "#DD8452"),
    ("GoP =\\nlog P(p*|frames)\\nper phoneme", 11.9, "#C44E52"),
    ("Per-phoneme\\nScore +\\nFeedback", 14.0, "#DA8BC3"),
]
for label, x, color in stages:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.85, 0.7), 1.7, 2.4,
                 boxstyle="round,pad=0.1", fc=color, ec="white", lw=2, alpha=0.88))
    ax.text(x, 1.9, label, ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", multialignment="center")

for i in range(len(stages)-1):
    ax.annotate("", xy=(stages[i+1][1]-0.86, 1.9),
                xytext=(stages[i][1]+0.86, 1.9),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

ax.set_title("Goodness-of-Pronunciation (GoP) Pipeline", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# ─── SECTION D: TRAINING ─────────────────────────────────────────────────────
cells.append(md("---\n## 5 · Training <a id='5'></a>"))
cells.append(cell("""\
import numpy as np
import matplotlib.pyplot as plt

# Synthetic training history — replace with your real history dict/CSV
np.random.seed(42)
EPOCHS = 40
t = np.arange(1, EPOCHS+1)

def smooth(arr, w=3):
    return np.convolve(arr, np.ones(w)/w, mode="same")

tr_loss = smooth(2.0 * np.exp(-t/10) + 0.05 * np.random.randn(EPOCHS) + 0.15)
val_loss = smooth(2.2 * np.exp(-t/10) + 0.07 * np.random.randn(EPOCHS) + 0.22)
tr_acc   = smooth(1 - np.exp(-t/8) - 0.04*np.random.randn(EPOCHS) - 0.05)
val_acc  = smooth(1 - np.exp(-t/9) - 0.05*np.random.randn(EPOCHS) - 0.08)
tr_acc   = np.clip(tr_acc, 0.4, 0.99); val_acc = np.clip(val_acc, 0.35, 0.97)

best_epoch = int(np.argmin(val_loss)) + 1
"""))

# Graph 16 – Loss curves
cells.append(md("### Graph 16 — Training vs Validation Loss Curve"))
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, tr_loss, label="Train Loss", color="#4C72B0", linewidth=2)
ax.plot(t, val_loss, label="Val Loss",  color="#DD8452", linewidth=2, linestyle="--")
ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
           label=f"Best epoch = {best_epoch}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout(); plt.show()
"""))

# Graph 17 – Accuracy curves
cells.append(md("### Graph 17 — Training vs Validation Accuracy Curve"))
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, tr_acc,  label="Train Acc", color="#4C72B0", linewidth=2)
ax.plot(t, val_acc, label="Val Acc",   color="#DD8452", linewidth=2, linestyle="--")
ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, label=f"Best epoch={best_epoch}")
ax.set_ylim(0, 1); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout(); plt.show()
"""))

# Graph 18 – LR schedule
cells.append(md("### Graph 18 — Learning Rate Schedule"))
cells.append(cell("""\
import numpy as np, matplotlib.pyplot as plt
# Cosine annealing with warmup (5 epochs)
lr_max = 3e-4; warmup = 5
lr = []
for e in range(1, EPOCHS+1):
    if e <= warmup:
        lr.append(lr_max * e / warmup)
    else:
        lr.append(lr_max * 0.5 * (1 + np.cos(np.pi * (e-warmup)/(EPOCHS-warmup))))
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, lr, color="#8172B3", linewidth=2)
ax.fill_between(t, lr, alpha=0.15, color="#8172B3")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule — Cosine Annealing with Linear Warmup",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# Graph 19 – Per-class accuracy over training
cells.append(md("### Graph 19 — Per-Class Accuracy over Training Epochs"))
cells.append(cell("""\
import numpy as np, matplotlib.pyplot as plt
ACCENTS = ["Tamil","Telugu","Hindi","Kannada","Malayalam","Marathi","Bengali"]
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3"]
np.random.seed(5)
fig, ax = plt.subplots(figsize=(11, 5))
for i, (acc, color) in enumerate(zip(ACCENTS, PALETTE)):
    base = 0.5 + i*0.01
    speed = 7 + i*0.8
    acc_curve = np.clip(1 - (1-base)*np.exp(-t/speed) + 0.02*np.random.randn(EPOCHS), 0, 1)
    ax.plot(t, acc_curve, label=acc, color=color, linewidth=1.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
ax.set_title("Per-Class Accuracy over Training", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8); ax.set_ylim(0.3, 1.0)
plt.tight_layout(); plt.show()
"""))

nb = json.load(open(NB))
nb["cells"].extend(cells)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=2)
print(f"Sections C+D written — {len(cells)} cells. Total: {len(nb['cells'])}")
