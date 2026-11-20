import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    tr = common.load_split("c_train")
    va = common.load_split("c_val")
    te = common.load_split("c_test")
    print("URL rows:", len(tr), len(va), len(te))

    t0 = time.time()
    Ftr = common.url_feature_frame(tr["url"])
    Fva = common.url_feature_frame(va["url"])
    Fte = common.url_feature_frame(te["url"])
    print("lexical features built in", round(time.time() - t0, 1), "s")

    best_hgb, best_f1 = None, -1
    grid = [
        {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 200},
        {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 300},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400},
    ]
    hgb_trials = []
    for params in grid:
        m = HistGradientBoostingClassifier(random_state=config.SEED, early_stopping=True, **params)
        m.fit(Ftr, tr["label"])
        f1 = f1_score(va["label"], m.predict(Fva))
        hgb_trials.append({**params, "val_f1": f1})
        if f1 > best_f1:
            best_hgb, best_f1, best_params = m, f1, params
    print("HGB best", best_params, round(best_f1, 4))

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=config.URL_TFIDF_FEATURES,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    Xtr = vec.fit_transform(tr["url"].apply(common.strip_scheme))
    Xva = vec.transform(va["url"].apply(common.strip_scheme))
    Xte = vec.transform(te["url"].apply(common.strip_scheme))

    best_sgd, best_f1s = None, -1
    sgd_trials = []
    for alpha in (1e-6, 1e-5, 1e-4):
        m = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=1e-4, random_state=config.SEED)
        m.fit(Xtr, tr["label"])
        f1 = f1_score(va["label"], m.predict(Xva))
        sgd_trials.append({"alpha": alpha, "val_f1": f1})
        if f1 > best_f1s:
            best_sgd, best_f1s, best_alpha = m, f1, alpha
    print("SGD best alpha", best_alpha, round(best_f1s, 4))

    p_hgb = best_hgb.predict_proba(Fte)[:, 1]
    p_sgd = best_sgd.predict_proba(Xte)[:, 1]
    p_blend = 0.5 * (p_hgb + p_sgd)

    rows = []
    for name, p in (
        ("URL lexical features + HistGradientBoosting", p_hgb),
        ("URL char n-gram TF-IDF + SGD logistic", p_sgd),
        ("URL blend (mean of both)", p_blend),
    ):
        rows.append({"model": name, **common.metrics_from_prob(te["label"], p)})
    results = pd.DataFrame(rows)
    results.to_csv(config.REPORT_DIR / "url_model_results.csv", index=False)
    common.save_json({"hgb_trials": hgb_trials, "sgd_trials": sgd_trials, "best_hgb": best_params, "best_alpha": best_alpha}, config.REPORT_DIR / "url_model_tuning.json")
    print(results.round(4).to_string(index=False))

    joblib.dump({"hgb": best_hgb, "vec": vec, "sgd": best_sgd}, config.MODEL_DIR / "url_model.joblib")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(te["label"], (p_blend >= 0.5).astype(int), display_labels=["benign", "malicious"], ax=ax, colorbar=False)
    ax.set_title("URL blend model - test set")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "url_confusion_matrix.png", dpi=150)
    plt.close(fig)

    n = min(5000, len(Fva))
    idx = np.random.default_rng(config.SEED).choice(len(Fva), n, replace=False)
    imp = permutation_importance(best_hgb, Fva.iloc[idx], va["label"].iloc[idx], n_repeats=3, random_state=config.SEED, scoring="f1")
    order = np.argsort(imp.importances_mean)[::-1][:12]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([common.URL_FEATURE_NAMES[i] for i in order][::-1], imp.importances_mean[order][::-1], color="#1565c0")
    ax.set_xlabel("drop in F1 when shuffled")
    ax.set_title("URL lexical feature importance")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "url_feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved URL model to", config.MODEL_DIR / "url_model.joblib")


if __name__ == "__main__":
    main()
