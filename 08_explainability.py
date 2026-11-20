import html
import re

import joblib
import numpy as np
import pandas as pd
import shap

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_DOCS = 1000
FAITHFULNESS_DOCS = 300
TOP_K = 5
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def row_shap(explainer, X):
    out = []
    for start in range(0, X.shape[0], 40):
        chunk = X[start : start + 40]
        vals = explainer.shap_values(chunk)
        if isinstance(vals, list):
            vals = vals[-1]
        vals = np.asarray(vals)
        for i in range(chunk.shape[0]):
            idx = chunk.indices[chunk.indptr[i] : chunk.indptr[i + 1]]
            out.append({int(j): float(vals[i, j]) for j in idx})
    return out


def remove_words(text, words):
    for w in words:
        text = re.sub(r"\b" + re.escape(w) + r"\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main():
    vec = joblib.load(config.MODEL_DIR / "tfidf.joblib")
    lr = joblib.load(config.MODEL_DIR / "lr_tuned.joblib")
    tr = common.load_split("a_train")
    te = common.load_split("a_test")
    feats = np.array(vec.get_feature_names_out())

    bg_idx = np.random.default_rng(config.SEED).choice(len(tr), min(2000, len(tr)), replace=False)
    Xbg = vec.transform(tr["text"].iloc[bg_idx])
    explainer = shap.LinearExplainer(lr, shap.maskers.Independent(Xbg, max_samples=500))

    sample = te.sample(min(SAMPLE_DOCS, len(te)), random_state=config.SEED).reset_index(drop=True)
    Xs = vec.transform(sample["text"]).tocsr()
    shap_rows = row_shap(explainer, Xs)
    probs = lr.predict_proba(Xs)[:, 1]

    totals, counts, abs_totals = {}, {}, {}
    for row in shap_rows:
        for j, v in row.items():
            totals[j] = totals.get(j, 0.0) + v
            abs_totals[j] = abs_totals.get(j, 0.0) + abs(v)
            counts[j] = counts.get(j, 0) + 1
    min_docs = max(5, int(0.005 * len(sample)))
    table = pd.DataFrame(
        {
            "token": [feats[j] for j in totals],
            "mean_shap_when_present": [totals[j] / counts[j] for j in totals],
            "mean_abs_shap_when_present": [abs_totals[j] / counts[j] for j in totals],
            "docs_present": [counts[j] for j in totals],
        }
    )
    table = table[table["docs_present"] >= min_docs]
    scam_tokens = table.sort_values("mean_shap_when_present", ascending=False).head(20)
    safe_tokens = table.sort_values("mean_shap_when_present").head(20)
    pd.concat([scam_tokens.assign(direction="scam"), safe_tokens.assign(direction="safe")]).to_csv(
        config.REPORT_DIR / "shap_top_tokens.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].barh(scam_tokens["token"][::-1], scam_tokens["mean_shap_when_present"][::-1], color="#c62828")
    axes[0].set_title("Tokens pushing towards SCAM (mean SHAP)")
    axes[1].barh(safe_tokens["token"][::-1], safe_tokens["mean_shap_when_present"][::-1], color="#2e7d32")
    axes[1].set_title("Tokens pushing towards SAFE (mean SHAP)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "shap_top_tokens.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(config.SEED + 5)
    scam_idx = [i for i in range(len(sample)) if sample["label"].iloc[i] == 1 and probs[i] >= 0.5]
    chosen = rng.choice(scam_idx, min(FAITHFULNESS_DOCS, len(scam_idx)), replace=False) if scam_idx else []
    drops_shap, drops_rand, flips_shap, flips_rand = [], [], [], []
    for i in chosen:
        text = sample["text"].iloc[i]
        row = shap_rows[i]
        unigrams = [(j, v) for j, v in row.items() if " " not in feats[j]]
        positive = sorted([x for x in unigrams if x[1] > 0], key=lambda x: -x[1])[:TOP_K]
        if not positive:
            continue
        k = len(positive)
        random_pick = [unigrams[int(r)] for r in rng.choice(len(unigrams), min(k, len(unigrams)), replace=False)]
        t_shap = remove_words(text, [feats[j] for j, _ in positive])
        t_rand = remove_words(text, [feats[j] for j, _ in random_pick])
        p0 = probs[i]
        p_shap = lr.predict_proba(vec.transform([t_shap]))[0, 1]
        p_rand = lr.predict_proba(vec.transform([t_rand]))[0, 1]
        drops_shap.append(p0 - p_shap)
        drops_rand.append(p0 - p_rand)
        flips_shap.append(p_shap < 0.5)
        flips_rand.append(p_rand < 0.5)

    faith = {
        "messages_tested": len(drops_shap),
        "tokens_removed_per_message": TOP_K,
        "mean_scam_probability_drop_top_shap_tokens": float(np.mean(drops_shap)) if drops_shap else None,
        "mean_scam_probability_drop_random_tokens": float(np.mean(drops_rand)) if drops_rand else None,
        "verdict_flip_rate_top_shap_tokens": float(np.mean(flips_shap)) if flips_shap else None,
        "verdict_flip_rate_random_tokens": float(np.mean(flips_rand)) if flips_rand else None,
    }
    common.save_json(faith, config.REPORT_DIR / "explanation_faithfulness.json")

    cards = []
    picks = []
    for lab in (1, 1, 1, 0, 0, 0):
        pool = [i for i in range(len(sample)) if sample["label"].iloc[i] == lab and i not in picks]
        if pool:
            picks.append(pool[0])
    for i in picks:
        text = sample["text"].iloc[i]
        word_scores = {}
        for j, v in shap_rows[i].items():
            if " " not in feats[j]:
                word_scores[feats[j]] = v
        mx = max([abs(v) for v in word_scores.values()] + [1e-9])
        parts = []
        for tok in re.findall(r"\S+|\s+", text[:600]):
            m = WORD_RE.search(tok)
            key = m.group(0).lower() if m else None
            v = word_scores.get(key)
            if v is None or abs(v) < 0.15 * mx:
                parts.append(html.escape(tok))
            else:
                a = min(abs(v) / mx, 1.0) * 0.75
                color = f"rgba(198,40,40,{a:.2f})" if v > 0 else f"rgba(46,125,50,{a:.2f})"
                parts.append(f'<span style="background:{color}">{html.escape(tok)}</span>')
        verdict = "SCAM" if probs[i] >= 0.5 else "SAFE"
        truth = "scam" if sample["label"].iloc[i] == 1 else "safe"
        cards.append(
            f'<div style="border:1px solid #ccc;padding:10px;margin:10px 0;font-family:sans-serif">'
            f"<b>Model: {verdict}</b> (scam probability {probs[i]:.2f}) | true label: {truth}<br><br>{''.join(parts)}</div>"
        )
    (config.REPORT_DIR / "shap_examples.html").write_text(
        "<html><body><h3>Token contributions (red = towards scam, green = towards safe)</h3>" + "".join(cards) + "</body></html>",
        encoding="utf-8",
    )

    print(scam_tokens.head(10).round(3).to_string(index=False))
    print(safe_tokens.head(10).round(3).to_string(index=False))
    print("Faithfulness:", faith)


if __name__ == "__main__":
    main()
