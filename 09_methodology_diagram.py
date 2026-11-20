import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config

fig, ax = plt.subplots(figsize=(13, 8.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8.2)
ax.axis("off")

COLORS = {"data": "#e3f2fd", "prep": "#fff8e1", "model": "#e8f5e9", "fuse": "#f3e5f5", "eval": "#ffebee", "app": "#e0f2f1"}


def box(x, y, w, h, text, kind, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=COLORS[kind], ec="#455a64", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.2))


box(0.2, 6.6, 2.7, 1.2, "Dataset A\nPhishing emails (text)\nraw Kaggle CSV", "data")
box(3.2, 6.6, 2.7, 1.2, "Dataset B\nPhishing email detection (text)\nraw Kaggle CSV", "data")
box(7.4, 6.6, 2.7, 1.2, "Dataset C\nMalicious URLs\nraw Kaggle CSV (650k rows)", "data")

box(0.2, 5.0, 5.7, 1.1, "Cleaning: drop nulls, strip HTML, mask URLs, remove duplicates and\nconflicting labels, remove A/B overlap, stratified 70/15/15 split", "prep")
box(7.4, 5.0, 2.7, 1.1, "URL cleaning: drop defacement,\nstrip scheme, deduplicate,\nstratified split", "prep")

box(0.2, 3.3, 1.8, 1.2, "TF-IDF +\nLR, NB, SVM,\nXGBoost (tuned)", "model")
box(2.2, 3.3, 1.8, 1.2, "DistilBERT\nfine-tuned\n(Hugging Face)", "model")
box(7.4, 3.3, 2.7, 1.2, "URL model: lexical features\n+ HistGradientBoosting and\nchar n-gram SGD (blend)", "model")
box(4.2, 3.3, 1.7, 1.2, "URLs extracted\nfrom each message", "prep")

box(2.2, 1.6, 5.0, 1.1, "Stacked meta-learner (logistic regression on base-model probabilities\n+ URL risk scores), cost-sensitive threshold, calibration", "fuse")
box(8.0, 1.6, 2.5, 1.1, "SHAP token\nexplanations", "fuse")
box(2.2, 0.15, 5.0, 1.0, "Streamlit app: verdict, scam probability, highlighted words,\nlink scores, safety advice", "app")

box(10.8, 5.0, 2.0, 2.8, "Evaluation\n\nBaseline vs tuned\nvs ensemble\n\nAblation:\ntext / URL / hybrid\n\nCross-dataset test\n\nObfuscation\nrobustness\n\nFaithfulness of\nexplanations", "eval", fs=8)
box(10.8, 1.6, 2.0, 1.1, "Metrics: accuracy,\nprecision, recall,\nF1, ROC-AUC, ECE", "eval", fs=8)

arrow(1.55, 6.6, 1.55, 6.1)
arrow(4.55, 6.6, 4.55, 6.1)
arrow(8.75, 6.6, 8.75, 6.1)
arrow(1.1, 5.0, 1.1, 4.5)
arrow(3.1, 5.0, 3.1, 4.5)
arrow(5.05, 5.0, 5.05, 4.5)
arrow(8.75, 5.0, 8.75, 4.5)
arrow(7.4, 3.9, 5.9, 3.9)
arrow(5.05, 3.3, 5.05, 2.7)
arrow(1.1, 3.3, 3.0, 2.7)
arrow(3.1, 3.3, 3.6, 2.7)
arrow(7.2, 2.15, 8.0, 2.15)
arrow(4.7, 1.6, 4.7, 1.15)
arrow(10.8, 2.15, 10.5, 2.15)
arrow(11.8, 5.0, 11.8, 2.7)

fig.suptitle("ChaunkUp methodology: hybrid text + URL scam detection", fontsize=13, y=0.97)
fig.savefig(config.FIG_DIR / "methodology_diagram.png", dpi=200, bbox_inches="tight")
print("saved", config.FIG_DIR / "methodology_diagram.png")
