import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUG_MAX_ROWS = 30000
AUG_LEVELS = [0.05, 0.10, 0.20, 0.30]


def predict_fns():
    vec = joblib.load(config.MODEL_DIR / "tfidf.joblib")
    lr_base = joblib.load(config.MODEL_DIR / "lr_baseline.joblib")
    lr = joblib.load(config.MODEL_DIR / "lr_tuned.joblib")
    svm = joblib.load(config.MODEL_DIR / "svm_cal.joblib")
    nb = joblib.load(config.MODEL_DIR / "nb.joblib")
    xgb = joblib.load(config.MODEL_DIR / "xgb_bundle.joblib")

    def wrap(model, selector=None):
        def f(texts):
            X = vec.transform(texts)
            if selector is not None:
                X = selector.transform(X)
            return model.predict_proba(X)[:, 1]

        return f

    return {
        "Baseline: TF-IDF + Logistic Regression": wrap(lr_base),
        "TF-IDF + Naive Bayes (tuned)": wrap(nb),
        "TF-IDF + Logistic Regression (tuned)": wrap(lr),
        "TF-IDF + Linear SVM (tuned, calibrated)": wrap(svm),
        "TF-IDF + XGBoost (tuned)": wrap(xgb["model"], xgb["selector"]),
    }


def evaluate_under_attack(name, fn, scam_df, safe_df, rows):
    for level in config.ROBUST_LEVELS:
        seed = config.SEED + int(level * 1000)
        obf_scam = common.obfuscate_series(scam_df["text"], level, seed)
        obf_safe = common.obfuscate_series(safe_df["text"], level, seed + 1)
        p_scam = fn(obf_scam)
        p_safe_clean = fn(list(safe_df["text"]))
        p_safe_obf = fn(obf_safe)
        y = np.concatenate([np.ones(len(p_scam)), np.zeros(len(p_safe_clean))])
        p = np.concatenate([p_scam, p_safe_clean])
        m = common.metrics_from_prob(y, p)
        rows.append(
            {
                "model": name,
                "level": level,
                "scam_recall_under_attack": float((p_scam >= 0.5).mean()),
                "safe_false_positive_rate_when_obfuscated": float((p_safe_obf >= 0.5).mean()),
                "accuracy": m["accuracy"],
                "f1": m["f1"],
            }
        )


def main():
    te = common.load_split("a_test")
    tr = common.load_split("a_train")
    scam_te = te[te["label"] == 1].reset_index(drop=True)
    safe_te = te[te["label"] == 0].reset_index(drop=True)
    print("Attacking", len(scam_te), "scam and", len(safe_te), "safe test messages")

    rows = []
    for name, fn in predict_fns().items():
        evaluate_under_attack(name, fn, scam_te, safe_te, rows)
        print("evaluated", name)

    params = common.load_json(config.REPORT_DIR / "text_best_params.json")["lr"]
    aug_src = tr.sample(min(AUG_MAX_ROWS, len(tr)), random_state=config.SEED)
    rng = np.random.default_rng(config.SEED + 99)
    aug_texts = []
    for t in aug_src["text"]:
        level = AUG_LEVELS[int(rng.integers(0, len(AUG_LEVELS)))]
        aug_texts.append(common.obfuscate(t, level, rng))
    aug_df = pd.DataFrame({"text": aug_texts, "label": aug_src["label"].values})
    full = pd.concat([tr[["text", "label"]], aug_df], ignore_index=True)
    print("Adversarial augmentation: added", len(aug_df), "obfuscated copies (both classes)")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )
    X = vec.fit_transform(full["text"])
    clf = LogisticRegression(solver="liblinear", max_iter=1000, **params).fit(X, full["label"])

    def aug_fn(texts):
        return clf.predict_proba(vec.transform(texts))[:, 1]

    evaluate_under_attack("LR (tuned) + adversarial augmentation", aug_fn, scam_te, safe_te, rows)
    joblib.dump({"vec": vec, "clf": clf}, config.MODEL_DIR / "lr_adversarial.joblib")

    results = pd.DataFrame(rows)
    bert_path = config.REPORT_DIR / "bert_robustness.csv"
    if bert_path.exists():
        b = pd.read_csv(bert_path)
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "model": "DistilBERT fine-tuned",
                        "level": b["level"],
                        "scam_recall_under_attack": b["scam_recall"],
                        "safe_false_positive_rate_when_obfuscated": np.nan,
                        "accuracy": b["accuracy"],
                        "f1": b["f1"],
                    }
                ),
            ],
            ignore_index=True,
        )
    results.to_csv(config.REPORT_DIR / "robustness_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in results.groupby("model", sort=False):
        ax.plot(g["level"], g["scam_recall_under_attack"], marker="o", label=name.replace("TF-IDF + ", ""))
    ax.set_xlabel("obfuscation level (probability a letter is altered)")
    ax.set_ylabel("scam recall")
    ax.set_title("Robustness to obfuscated scam text")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "robustness_recall.png", dpi=150)
    plt.close(fig)

    pivot = results.pivot_table(index="model", columns="level", values="scam_recall_under_attack")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
