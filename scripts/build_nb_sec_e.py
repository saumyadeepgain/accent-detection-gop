"""Build Section E – Accent Classification Evaluation (Graphs 20-27)."""
import json, uuid

NB = "accent_gop_notebook.ipynb"
def cell(src, ct="code"):
    return {"cell_type": ct, "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": src, **({"outputs": [], "execution_count": None} if ct == "code" else {})}
def md(src): return cell(src, "markdown")

cells = []
cells.append(md("---\n## 6 · Accent Classification — Evaluation <a id='6'></a>"))

# Shared setup
cells.append(cell("""\
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import label_binarize

ACCENTS = ["Tamil","Telugu","Hindi","Kannada","Malayalam","Marathi","Bengali"]
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3"]
np.random.seed(42)
N_TEST = 600

# Synthetic ground-truth and predictions — replace with real model outputs
y_true = np.random.choice(len(ACCENTS), N_TEST,
         p=[0.18,0.17,0.15,0.11,0.12,0.14,0.13])
# Simulate a reasonably good but imperfect classifier
probs = np.zeros((N_TEST, len(ACCENTS)))
for i, yt in enumerate(y_true):
    base = np.random.dirichlet(np.ones(len(ACCENTS)) * 0.5)
    base[yt] += 2.5
    probs[i] = base / base.sum()

y_pred = probs.argmax(axis=1)
y_true_bin = label_binarize(y_true, classes=range(len(ACCENTS)))
print("Test set accuracy:", round((y_pred == y_true).mean(), 4))
"""))

# Graph 20 – Confusion matrix
cells.append(md("### Graph 20 — Confusion Matrix (7×7) — ⭐ Headline Evaluation Visual"))
cells.append(cell("""\
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Confusion Matrix (raw counts)", "Confusion Matrix (row-normalised)"]):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=ACCENTS, yticklabels=ACCENTS,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Accent"); ax.set_ylabel("True Accent")
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
plt.suptitle("7-Class Accent Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# Graph 21 – Per-class P/R/F1
cells.append(md("### Graph 21 — Per-Class Precision / Recall / F1"))
cells.append(cell("""\
from sklearn.metrics import precision_score, recall_score, f1_score

prec = precision_score(y_true, y_pred, average=None, zero_division=0)
rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

x = np.arange(len(ACCENTS)); w = 0.26
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w, prec, w, label="Precision", color="#4C72B0", alpha=0.85)
ax.bar(x,     rec,  w, label="Recall",    color="#DD8452", alpha=0.85)
ax.bar(x + w, f1,   w, label="F1",        color="#55A868", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(ACCENTS, rotation=25, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Per-Class Precision / Recall / F1", fontsize=13, fontweight="bold")
for i, (p,r,f) in enumerate(zip(prec,rec,f1)):
    ax.text(i-w, p+0.01, f"{p:.2f}", ha="center", fontsize=7, color="#4C72B0")
    ax.text(i,   r+0.01, f"{r:.2f}", ha="center", fontsize=7, color="#DD8452")
    ax.text(i+w, f+0.01, f"{f:.2f}", ha="center", fontsize=7, color="#55A868")
plt.tight_layout(); plt.show()
"""))

# Graph 22 – Macro vs weighted F1
cells.append(md("### Graph 22 — Macro vs Weighted F1 Comparison"))
cells.append(cell("""\
macro_f1    = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
micro_f1    = f1_score(y_true, y_pred, average="micro")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Macro F1", "Weighted F1", "Micro F1 (=Acc)"],
              [macro_f1, weighted_f1, micro_f1],
              color=["#4C72B0","#DD8452","#55A868"], width=0.5, edgecolor="white")
for bar, v in zip(bars, [macro_f1, weighted_f1, micro_f1]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", fontweight="bold", fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
ax.set_title("Macro vs Weighted vs Micro F1\\n(important with class imbalance)",
             fontweight="bold", fontsize=12)
plt.tight_layout(); plt.show()
print(f"Macro={macro_f1:.3f}  Weighted={weighted_f1:.3f}  Micro={micro_f1:.3f}")
"""))

# Graph 23 – ROC curves
cells.append(md("### Graph 23 — ROC Curves (One-vs-Rest, all 7 classes)"))
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(9, 7))
for i, (acc, color) in enumerate(zip(ACCENTS, PALETTE)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{acc} (AUC={roc_auc:.2f})")
ax.plot([0,1],[0,1],"k--",linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — One-vs-Rest per Accent Class",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9); plt.tight_layout(); plt.show()
"""))

# Graph 24 – PR curves
cells.append(md("### Graph 24 — Precision-Recall Curves (One-vs-Rest)"))
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(9, 7))
for i, (acc, color) in enumerate(zip(ACCENTS, PALETTE)):
    prec_c, rec_c, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], probs[:, i])
    ax.plot(rec_c, prec_c, color=color, linewidth=2, label=f"{acc} (AP={ap:.2f})")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — One-vs-Rest per Accent",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9); plt.tight_layout(); plt.show()
"""))

# Graph 25 – Top-K accuracy
cells.append(md("### Graph 25 — Top-1 / Top-2 / Top-3 Accuracy"))
cells.append(cell("""\
def topk_acc(probs, y_true, k):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

k_vals = [1, 2, 3]
k_accs = [topk_acc(probs, y_true, k) for k in k_vals]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([f"Top-{k}" for k in k_vals], k_accs,
              color=["#C44E52","#DD8452","#55A868"], width=0.45)
for bar, v in zip(bars, k_accs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.1%}",
            ha="center", fontweight="bold", fontsize=12)
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
ax.set_title("Top-K Accuracy (linguistically close accents)",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# Graph 26 – Calibration plot
cells.append(md("### Graph 26 — Calibration (Reliability) Diagram — Before vs After Temperature Scaling"))
cells.append(cell("""\
# Temperature scaling
def apply_temp(probs, T):
    scaled = np.log(probs + 1e-9) / T
    scaled -= scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)

T_best = 1.6   # typical value; replace with calibrated T from val set
probs_cal = apply_temp(probs, T_best)

def reliability_diagram(probs, y_true, n_bins=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(conf[mask].mean())
            bin_sizes.append(mask.sum())
    return np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes)

bc_raw, ba_raw, bs_raw = reliability_diagram(probs, y_true)
bc_cal, ba_cal, bs_cal = reliability_diagram(probs_cal, y_true)

def ece(confs, accs, sizes):
    return (np.abs(accs - confs) * sizes / sizes.sum()).sum()

ECE_before = ece(bc_raw, ba_raw, bs_raw)
ECE_after  = ece(bc_cal, ba_cal, bs_cal)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, bc, ba, bs, title, ece_val in zip(
        axes, [bc_raw, bc_cal], [ba_raw, ba_cal], [bs_raw, bs_cal],
        ["Before Temp. Scaling", "After Temp. Scaling (T=1.6)"],
        [ECE_before, ECE_after]):
    ax.bar(bc, ba, width=0.08, alpha=0.7, color="#4C72B0", label="Actual accuracy")
    ax.plot([0,1],[0,1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.fill_between(bc, bc, ba, alpha=0.15, color="red", label="Gap (miscalibration)")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Mean Predicted Confidence"); ax.set_ylabel("Fraction Correct")
    ax.set_title(f"{title}\\nECE = {ece_val:.4f}", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
plt.suptitle("Reliability Diagram — Confidence Calibration Check",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()
print(f"ECE before: {ECE_before:.4f}  |  ECE after: {ECE_after:.4f}")
"""))

# Graph 27 – Confidence distribution
cells.append(md("### Graph 27 — Confidence Score Distribution (Correct vs Incorrect Predictions)"))
cells.append(cell("""\
correct_mask = y_pred == y_true
conf_scores = probs.max(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, pr, title in zip(axes, [probs, probs_cal],
                          ["Before Calibration", "After Calibration (T=1.6)"]):
    c = pr.max(axis=1)
    correct = c[y_pred == y_true]
    wrong   = c[y_pred != y_true]
    ax.hist(correct, bins=30, alpha=0.65, color="#55A868", label=f"Correct (n={len(correct)})", density=True)
    ax.hist(wrong,   bins=30, alpha=0.65, color="#C44E52", label=f"Incorrect (n={len(wrong)})",  density=True)
    ax.set_xlabel("Max Softmax Confidence"); ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=9)
plt.suptitle("Confidence Distribution — Correct vs Incorrect Predictions",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

nb = json.load(open(NB))
nb["cells"].extend(cells)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=2)
print(f"Section E written — {len(cells)} cells. Total: {len(nb['cells'])}")
