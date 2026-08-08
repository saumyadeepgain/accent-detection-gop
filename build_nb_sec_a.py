"""Build Section A – Data Exploration / EDA (Graphs 1-8) into the notebook."""
import json, uuid, os

NB = "accent_gop_notebook.ipynb"

def cell(src, cell_type="code"):
    return {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }

def md(src):
    return cell(src, "markdown")

cells = []

# ── NOTEBOOK HEADER ──────────────────────────────────────────────────────────
cells.append(md("""# Accent Detection + Pronunciation Scoring (GoP) — Full Analysis Notebook
> **7 Indian-English accent classes · WavLM backbone · Goodness-of-Pronunciation pipeline**  
> Dataset: NISP + AccentDB + Svarah | Phoneme recogniser: wav2vec2-xlsr-53 | Aligner: MFA

---
**Table of Contents**
1. [Intro & Problem Framing](#1)
2. [Data Loading & EDA](#2)
3. [Feature Extraction — WavLM Embeddings](#3)
4. [Model Architecture](#4)
5. [Training](#5)
6. [Accent Classification — Evaluation](#6)
7. [Embedding Interpretability](#7)
8. [Language-ID Branch Evaluation](#8)
9. [GoP / Pronunciation Scoring](#9)
10. [Robustness & Testing](#10)
11. [Final Results & Summary](#11)
"""))

# ── SECTION 1: INTRO ─────────────────────────────────────────────────────────
cells.append(md("---\n## 1 · Intro & Problem Framing <a id='1'></a>"))
cells.append(md("""**Task:** Given a short audio clip of Indian-English speech, simultaneously predict:
- The speaker's **native-language accent** (Tamil / Telugu / Hindi / Kannada / Malayalam / Marathi / Bengali)
- A per-phoneme **Goodness-of-Pronunciation (GoP)** score indicating how close each sound is to a native-English reference

**Why this matters:** Accent-aware tutoring, call-centre quality scoring, L2 language learning apps.

**References**
- WavLM: [arxiv 2110.13900](https://arxiv.org/abs/2110.13900)  
- NISP: [arxiv 2007.06021](https://arxiv.org/abs/2007.06021)  
- Svarah: [arxiv 2305.15760](https://arxiv.org/abs/2305.15760)  
- AccentDB: https://github.com/AccentDB/data  
- MFA: https://montreal-forced-aligner.readthedocs.io  
- wav2vec2-xlsr-53-espeak: https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft  
- Voxlingua107: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
"""))

# ── SECTION 2: EDA ───────────────────────────────────────────────────────────
cells.append(md("---\n## 2 · Data Loading & EDA <a id='2'></a>"))

# Imports
cells.append(cell("""\
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3"]
ACCENTS = ["Tamil","Telugu","Hindi","Kannada","Malayalam","Marathi","Bengali"]
print("Imports OK")
"""))

# Synthetic data scaffold
cells.append(cell("""\
# ── SYNTHETIC DATA SCAFFOLD ──────────────────────────────────────────────────
# Replace the arrays below with real metadata loaded from your dataset CSVs.
# Expected columns: accent, speaker_id, gender, duration_sec, source, sample_rate

random.seed(42); np.random.seed(42)

COUNTS = dict(zip(ACCENTS, [312, 298, 255, 187, 203, 264, 253]))   # ~1772 speakers
rows = []
for acc, n in COUNTS.items():
    for i in range(n):
        rows.append({
            "accent":      acc,
            "speaker_id":  f"{acc[:3].upper()}{i:04d}",
            "gender":      random.choice(["M","F"]),
            "duration_sec":np.random.gamma(3, 3),   # utterance dur in seconds
            "sample_rate": random.choice([16000, 16000, 16000, 22050]),
            "source":      np.random.choice(["NISP","AccentDB","Svarah"],
                                         p=[0.42, 0.31, 0.27]),
        })
df = pd.DataFrame(rows)
print(df.shape, df.dtypes.to_dict())
df.head()
"""))

# Graph 1 – Speaker count per accent
cells.append(md("### Graph 1 — Speaker Count per Accent (Class Balance Check)"))
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(9, 4))
counts = df.groupby("accent")["speaker_id"].nunique().reindex(ACCENTS)
bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.7)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 4, str(v),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(counts.mean(), color="grey", linestyle="--", linewidth=1, label=f"Mean = {counts.mean():.0f}")
ax.set_title("Speaker Count per Accent Class", fontsize=14, fontweight="bold")
ax.set_xlabel("Accent"); ax.set_ylabel("Number of Speakers")
ax.legend(); plt.tight_layout(); plt.show()
print("Class imbalance ratio:", round(counts.max()/counts.min(), 2))
"""))

# Graph 2 – Duration distribution
cells.append(md("### Graph 2 — Audio Duration Distribution (per Utterance)"))
cells.append(cell("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# Left: overall histogram
axes[0].hist(df["duration_sec"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(df["duration_sec"].median(), color="red", linestyle="--",
                label=f'Median = {df["duration_sec"].median():.1f}s')
axes[0].set_title("Utterance Duration Distribution", fontweight="bold")
axes[0].set_xlabel("Duration (s)"); axes[0].set_ylabel("Count"); axes[0].legend()
# Right: per-accent box plot
data_by_acc = [df[df.accent == a]["duration_sec"].values for a in ACCENTS]
bp = axes[1].boxplot(data_by_acc, patch_artist=True, notch=True,
                     medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_xticklabels(ACCENTS, rotation=25, ha="right")
axes[1].set_title("Duration per Accent Group", fontweight="bold")
axes[1].set_ylabel("Duration (s)")
plt.tight_layout(); plt.show()
print(df["duration_sec"].describe().round(2))
"""))

# Graph 3 – Gender distribution
cells.append(md("### Graph 3 — Gender Distribution per Accent Group"))
cells.append(cell("""\
gender_df = df.groupby(["accent","gender"])["speaker_id"].nunique().unstack(fill_value=0)
gender_df = gender_df.reindex(ACCENTS)
fig, ax = plt.subplots(figsize=(9, 4))
gender_df.plot(kind="bar", stacked=True, color=["#4C72B0","#DD8452"],
               edgecolor="white", ax=ax, width=0.6)
ax.set_title("Gender Distribution per Accent Group", fontweight="bold")
ax.set_xlabel("Accent"); ax.set_ylabel("Number of Speakers")
ax.set_xticklabels(ACCENTS, rotation=25, ha="right")
ax.legend(title="Gender")
# annotate imbalance ratio
for i, acc in enumerate(ACCENTS):
    m = gender_df.loc[acc, "M"] if "M" in gender_df.columns else 0
    f = gender_df.loc[acc, "F"] if "F" in gender_df.columns else 0
    ratio = m/f if f > 0 else float("inf")
    ax.text(i, m+f+2, f"M:F={ratio:.1f}", ha="center", fontsize=7.5, color="dimgray")
plt.tight_layout(); plt.show()
"""))

# Graph 4 – Sample-rate consistency table
cells.append(md("### Graph 4 — Sampling Rate / Format Consistency Check"))
cells.append(cell("""\
sr_table = (df.groupby(["source","sample_rate"])
              .size().reset_index(name="utterances"))
print(sr_table.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis("off")
tbl = ax.table(
    cellText=sr_table.values,
    colLabels=sr_table.columns,
    cellLoc="center", loc="center",
    colColours=["#4C72B0","#DD8452","#55A868"],
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1.3, 1.8)
for (r, c), cell_obj in tbl.get_celld().items():
    if r == 0:
        cell_obj.set_text_props(color="white", fontweight="bold")
ax.set_title("Standardised to 16 kHz mono WAV before WavLM extraction",
             fontsize=9, color="gray", pad=8)
plt.tight_layout(); plt.show()
"""))

# Graph 5 – Waveform + spectrogram per accent
cells.append(md("### Graph 5 — Waveform + Mel-Spectrogram Samples per Accent"))
cells.append(cell("""\
# Generates synthetic signals; swap sr/y_signal with librosa.load(<real_path>)
fig, axes = plt.subplots(7, 2, figsize=(14, 20))
SR = 16000; DUR = 2.0
for row, (acc, color) in enumerate(zip(ACCENTS, PALETTE)):
    # synthetic signal — replace with: y, sr = librosa.load(path, sr=SR, duration=DUR)
    np.random.seed(row)
    base_freq = 120 + row * 18          # rough F0 proxy
    t = np.linspace(0, DUR, int(SR*DUR))
    y = 0.5 * np.sin(2*np.pi*base_freq*t) + 0.05*np.random.randn(len(t))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Waveform
    axes[row,0].plot(t[:SR//4], y[:SR//4], color=color, linewidth=0.6)
    axes[row,0].set_title(f"{acc} — Waveform", fontweight="bold", fontsize=9)
    axes[row,0].set_yticks([])
    # Spectrogram
    librosa.display.specshow(mel_db, sr=SR, x_axis="time", y_axis="mel",
                             fmax=8000, ax=axes[row,1], cmap="magma")
    axes[row,1].set_title(f"{acc} — Mel Spectrogram", fontweight="bold", fontsize=9)
plt.suptitle("Waveform + Mel-Spectrogram Examples per Accent", fontsize=13,
             fontweight="bold", y=1.005)
plt.tight_layout(); plt.show()
"""))

# Graph 6 – Pitch (F0) distribution
cells.append(md("### Graph 6 — Pitch (F0) Distribution per Accent Group"))
cells.append(cell("""\
# Synthetic F0 — replace with real pyin/praat estimates per speaker
np.random.seed(0)
f0_data = {a: np.random.normal(loc=100+i*12, scale=18, size=COUNTS[a])
            for i, a in enumerate(ACCENTS)}
fig, ax = plt.subplots(figsize=(11, 5))
parts = ax.violinplot([f0_data[a] for a in ACCENTS],
                      positions=range(len(ACCENTS)),
                      showmedians=True, showextrema=False)
for pc, color in zip(parts["bodies"], PALETTE):
    pc.set_facecolor(color); pc.set_alpha(0.65)
parts["cmedians"].set_color("red"); parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(ACCENTS))); ax.set_xticklabels(ACCENTS, rotation=20, ha="right")
ax.set_title("Pitch (F0) Distribution per Accent Group", fontweight="bold", fontsize=13)
ax.set_ylabel("Fundamental Frequency — F0 (Hz)")
ax.set_xlabel("Accent (mother-tongue background)")
plt.tight_layout(); plt.show()
"""))

# Graph 7 – Speaking rate
cells.append(md("### Graph 7 — Speaking Rate (Syllables/sec) per Accent"))
cells.append(cell("""\
np.random.seed(1)
rate_data = {a: np.random.normal(loc=3.5+i*0.15, scale=0.5, size=COUNTS[a])
             for i, a in enumerate(ACCENTS)}
fig, ax = plt.subplots(figsize=(9, 4))
bp = ax.boxplot([rate_data[a] for a in ACCENTS],
                patch_artist=True, notch=True,
                medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(ACCENTS, rotation=20, ha="right")
ax.set_title("Speaking Rate Distribution per Accent", fontweight="bold", fontsize=13)
ax.set_ylabel("Syllables per Second")
plt.tight_layout(); plt.show()
"""))

# Graph 8 – Dataset source pie
cells.append(md("### Graph 8 — Dataset Source Composition"))
cells.append(cell("""\
src_counts = df["source"].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    src_counts.values, labels=src_counts.index,
    autopct="%1.1f%%", startangle=140,
    colors=["#4C72B0","#DD8452","#55A868"],
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11),
)
for at in autotexts: at.set_fontweight("bold"); at.set_color("white")
ax.set_title("Dataset Source Composition\\n(NISP + AccentDB + Svarah)",
             fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
print(src_counts)
"""))

# ── WRITE TO NOTEBOOK ─────────────────────────────────────────────────────────
nb = json.load(open(NB))
nb["cells"].extend(cells)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=2)
print(f"Section A written — {len(cells)} cells added. Total cells: {len(nb['cells'])}")
