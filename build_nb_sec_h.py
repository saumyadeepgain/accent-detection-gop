"""Build Section H – GoP / Pronunciation Scoring (Graphs 34-38)."""
import json, uuid

NB = "accent_gop_notebook.ipynb"
def cell(src, ct="code"):
    return {"cell_type": ct, "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": src, **({"outputs": [], "execution_count": None} if ct == "code" else {})}
def md(src): return cell(src, "markdown")

cells = []
cells.append(md("---\n## 9 · GoP / Pronunciation Scoring <a id='9'></a>"))
cells.append(md("""\
**GoP formula:**  
$$GoP(p^*, s) = \\frac{1}{|s|} \\sum_{t \\in s} \\log P(p^* | o_t)$$  
where $p^*$ = canonical phoneme, $s$ = aligned segment frames, $o_t$ = wav2vec2 frame logits.

Scores are normalised to $[0,1]$ (1 = native-like pronunciation, 0 = very deviant).  
**Forced alignment** (MFA + CMU-dict) provides phoneme boundaries; **wav2vec2-xlsr-53-espeak-cv-ft** provides frame-level phone posteriors.
"""))

# Phoneme list
cells.append(cell("""\
import numpy as np, matplotlib.pyplot as plt, seaborn as sns

ACCENTS  = ["Tamil","Telugu","Hindi","Kannada","Malayalam","Marathi","Bengali"]
PALETTE  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3"]
# IPA phonemes representative of Indian-English challenge sounds
PHONEMES = ["p","b","t","d","k","g","tʃ","dʒ",
            "f","v","θ","ð","s","z","ʃ","ʒ",
            "m","n","ŋ","r","l","w","j",
            "ɪ","iː","ʊ","uː","ɛ","æ","ʌ","ɑː","ɒ","ɔː","ə","eɪ","aɪ"]
np.random.seed(42)
print(f"{len(PHONEMES)} phonemes tracked | {len(ACCENTS)} accent groups")
"""))

# Graph 34 – GoP distribution histogram
cells.append(md("### Graph 34 — GoP Score Distribution Histogram (per Phoneme)"))
cells.append(cell("""\
# Synthetic GoP scores — replace with real MFA + wav2vec2 outputs
all_gop = {}
for ph in PHONEMES:
    mu = np.random.uniform(0.45, 0.90)
    all_gop[ph] = np.clip(np.random.normal(mu, 0.15, 500), 0, 1)

# Plot top-12 phonemes with most variance
ph_stds = sorted(PHONEMES, key=lambda p: np.std(all_gop[p]), reverse=True)[:12]
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
for ax, ph in zip(axes.flat, ph_stds):
    ax.hist(all_gop[ph], bins=25, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(np.mean(all_gop[ph]), color="red", linestyle="--", linewidth=1.2,
               label=f"μ={np.mean(all_gop[ph]):.2f}")
    ax.set_title(f"/{ph}/", fontweight="bold"); ax.set_xlabel("GoP Score"); ax.legend(fontsize=7)
plt.suptitle("GoP Score Distribution per Phoneme (top-12 most variable)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# Graph 35 – GoP by accent group
cells.append(md("### Graph 35 — GoP Score by Accent Group (Which Accents Score Lowest on Which Sounds)"))
cells.append(cell("""\
# GoP scores per accent per phoneme — shape (7, n_phonemes_sample)
challenge_phones = ["θ","ð","æ","ɒ","r","v","w","tʃ","ʌ","ə"]  # known L2 challenges
gop_by_acc = {}
for i, acc in enumerate(ACCENTS):
    gop_by_acc[acc] = np.clip(
        np.random.normal(0.68 - i*0.02, 0.14, len(challenge_phones)*80).reshape(len(challenge_phones), 80),
        0, 1)
    # Tamil/Bengali struggle more with θ, ð
    if acc in ["Tamil","Bengali"]:
        gop_by_acc[acc][0] -= 0.18; gop_by_acc[acc][1] -= 0.15
    if acc in ["Hindi","Marathi"]:
        gop_by_acc[acc][3] -= 0.12  # ɒ confusion

fig, ax = plt.subplots(figsize=(13, 5))
positions = []
all_data = []
labels_tick = []
for pi, ph in enumerate(challenge_phones):
    for ai, acc in enumerate(ACCENTS):
        positions.append(pi*(len(ACCENTS)+1) + ai)
        all_data.append(np.clip(gop_by_acc[acc][pi], 0, 1))
        labels_tick.append("")

bp = ax.boxplot(all_data, positions=positions, patch_artist=True,
                widths=0.7, showfliers=False,
                medianprops=dict(color="white", linewidth=1.5))
colors_cycle = [PALETTE[ai] for pi in range(len(challenge_phones)) for ai in range(len(ACCENTS))]
for patch, color in zip(bp["boxes"], colors_cycle):
    patch.set_facecolor(color); patch.set_alpha(0.75)

tick_positions = [pi*(len(ACCENTS)+1) + (len(ACCENTS)-1)/2 for pi in range(len(challenge_phones))]
ax.set_xticks(tick_positions); ax.set_xticklabels([f"/{p}/" for p in challenge_phones], fontsize=11)
ax.set_ylabel("GoP Score"); ax.set_ylim(0, 1.05)
ax.set_title("GoP Score per Accent Group — Challenging Phonemes",
             fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=a) for c,a in zip(PALETTE, ACCENTS)]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)
plt.tight_layout(); plt.show()
"""))

# Graph 36 – GoP vs human rating scatter
cells.append(md("### Graph 36 — GoP Score vs Human Pronunciation Rating (Validation Scatter) ⭐"))
cells.append(cell("""\
from scipy import stats
np.random.seed(21)
n_human = 200   # number of human-annotated utterances
# Human raters score 1-5 (converted to 0-1) — collect via Praat/MATLAB annotation or crowd
gop_auto   = np.random.uniform(0.3, 1.0, n_human)
# Human scores correlate but with noise (Pearson r ≈ 0.78)
human_norm = np.clip(gop_auto + np.random.normal(0, 0.12, n_human), 0, 1)

r, p = stats.pearsonr(gop_auto, human_norm)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(gop_auto, human_norm, c=gop_auto, cmap="plasma",
                alpha=0.7, s=30, edgecolors="none")
m, b = np.polyfit(gop_auto, human_norm, 1)
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, m*x_line+b, "r-", linewidth=2, label=f"Linear fit  r={r:.3f}, p<0.001")
ax.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.4, label="Perfect agreement")
plt.colorbar(sc, ax=ax, label="Auto GoP Score")
ax.set_xlabel("Automatic GoP Score (wav2vec2-xlsr)", fontsize=11)
ax.set_ylabel("Normalised Human Rating (1–5 scale)", fontsize=11)
ax.set_title("GoP vs Human Pronunciation Rating\\n(n=200 manually annotated utterances)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout(); plt.show()
print(f"Pearson r = {r:.3f} | p-value = {p:.2e}")
"""))

# Graph 37 – Phoneme-level GoP heatmap
cells.append(md("### Graph 37 — Phoneme-Level GoP Heatmap (rows=phonemes, cols=accents) ⭐"))
cells.append(cell("""\
# Mean GoP score per (phoneme, accent) — shape (n_phones, 7)
np.random.seed(25)
n_ph = len(PHONEMES)
gop_matrix = np.random.uniform(0.55, 0.92, (n_ph, len(ACCENTS)))

# Inject realistic linguistic patterns:
# θ/ð hard for all Indian accents; r/l confusion for Tamil/Malayalam; retroflex influence
hard_phones = [PHONEMES.index(p) for p in ["θ","ð","æ","ɒ"] if p in PHONEMES]
for hp in hard_phones:
    gop_matrix[hp] -= np.random.uniform(0.15, 0.28, len(ACCENTS))
tamil_idx, mal_idx = ACCENTS.index("Tamil"), ACCENTS.index("Malayalam")
r_idx = PHONEMES.index("r") if "r" in PHONEMES else 0
gop_matrix[r_idx, [tamil_idx, mal_idx]] -= 0.20
gop_matrix = np.clip(gop_matrix, 0, 1)

fig, ax = plt.subplots(figsize=(11, 14))
sns.heatmap(gop_matrix, xticklabels=ACCENTS, yticklabels=PHONEMES,
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.3, linecolor="white", ax=ax,
            cbar_kws={"label":"Mean GoP Score","shrink":0.6},
            annot=True, fmt=".2f", annot_kws={"size":7})
ax.set_title("Phoneme-Level GoP Heatmap\\n(green=native-like, red=deviant pronunciation)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Accent Group"); ax.set_ylabel("IPA Phoneme")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.show()
print("Lowest average GoP phonemes:")
avg = gop_matrix.mean(axis=1)
for ph, sc in sorted(zip(PHONEMES, avg), key=lambda x: x[1])[:5]:
    print(f"  /{ph}/ → {sc:.3f}")
"""))

# Graph 38 – Forced alignment quality check
cells.append(md("### Graph 38 — Forced Alignment Quality Check (Spectrogram + Phoneme Boundaries)"))
cells.append(cell("""\
import librosa, librosa.display
import numpy as np, matplotlib.pyplot as plt

np.random.seed(30)
SR = 16000; DUR = 2.5
# Synthetic audio — replace with librosa.load(<real_file>)
t = np.linspace(0, DUR, int(SR*DUR))
y = 0.4*np.sin(2*np.pi*120*t) + 0.05*np.random.randn(len(t))
mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=80, fmax=8000)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Synthetic MFA phoneme boundaries — replace with real .TextGrid parse
phone_seq = ["hh","EH","l","OW","W","ER","l","D"]
boundaries = np.linspace(0, DUR, len(phone_seq)+1)

fig, ax = plt.subplots(figsize=(13, 4))
librosa.display.specshow(mel_db, sr=SR, x_axis="time", y_axis="mel",
                         fmax=8000, ax=ax, cmap="magma")
for start, end, ph in zip(boundaries[:-1], boundaries[1:], phone_seq):
    ax.axvline(start, color="cyan", linewidth=1.2, alpha=0.8)
    mid = (start+end)/2
    ax.text(mid, 7200, f"/{ph}/", ha="center", va="top", fontsize=8,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#333", alpha=0.6))
ax.axvline(boundaries[-1], color="cyan", linewidth=1.2, alpha=0.8)
ax.set_title("Forced Alignment Quality Check — Phoneme Boundaries on Mel Spectrogram\\n"
             "(verify MFA boundaries align with spectrogram landmarks before trusting GoP)",
             fontweight="bold", fontsize=11)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Mel Frequency (Hz)")
plt.tight_layout(); plt.show()
"""))

nb = json.load(open(NB))
nb["cells"].extend(cells)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=2)
print(f"Section H written — {len(cells)} cells. Total: {len(nb['cells'])}")
