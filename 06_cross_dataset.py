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


def make_vec():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )


def train_lr(texts, labels, params):
    vec = make_vec()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(solver="liblinear", max_iter=1000, **params)
    clf.fit(X, labels)
    return vec, clf


def main():
    params = common.load_json(config.REPORT_DIR / "text_best_params.json")["lr"]
    params = {k: v for k, v in params.items()}

    a_tr, a_te = common.load_split("a_train"), common.load_split("a_test")
    b_tr, b_te = common.load_split("b_train"), common.load_split("b_test")
    print("A train/test:", len(a_tr), len(a_te), "| B train/test:", len(b_tr), len(b_te))

    rows = []

    def evaluate(train_name, vec, clf, test_name, test_df):
        p = clf.predict_proba(vec.transform(test_df["text"]))[:, 1]
        m = common.metrics_from_prob(test_df["label"].values, p)
        rows.append({"model": "Tuned TF-IDF + LR", "train_on": train_name, "test_on": test_name, **m})
        print(train_name, "->", test_name, {k: round(v, 4) for k, v in m.items() if k in ("accuracy", "f1", "recall")})

    vec_a, clf_a = train_lr(a_tr["text"], a_tr["label"], params)
    evaluate("A only", vec_a, clf_a, "A test (in-dataset)", a_te)
    evaluate("A only", vec_a, clf_a, "B test (unseen dataset)", b_te)

    vec_b, clf_b = train_lr(b_tr["text"], b_tr["label"], params)
    evaluate("B only", vec_b, clf_b, "B test (in-dataset)", b_te)
    evaluate("B only", vec_b, clf_b, "A test (unseen dataset)", a_te)

    ab = pd.concat([a_tr, b_tr], ignore_index=True)
    vec_ab, clf_ab = train_lr(ab["text"], ab["label"], params)
    evaluate("A + B combined", vec_ab, clf_ab, "A test", a_te)
    evaluate("A + B combined", vec_ab, clf_ab, "B test", b_te)

    d_path = config.PROC_DIR / "d_test.csv"
    d_te = common.load_split("d_test") if d_path.exists() else None
    if d_te is not None:
        evaluate("A only", vec_a, clf_a, "D test (SMS, unseen)", d_te)
        evaluate("B only", vec_b, clf_b, "D test (SMS, unseen)", d_te)
        evaluate("A + B combined", vec_ab, clf_ab, "D test (SMS, unseen)", d_te)

    bert_path = config.PROC_DIR / "bert_probs.joblib"
    if bert_path.exists():
        bert = joblib.load(bert_path)
        rows.append({"model": "DistilBERT", "train_on": "A only", "test_on": "A test (in-dataset)", **common.metrics_from_prob(a_te["label"].values, bert["test"])})
        rows.append({"model": "DistilBERT", "train_on": "A only", "test_on": "B test (unseen dataset)", **common.metrics_from_prob(b_te["label"].values, bert["b_test"])})
        if d_te is not None and "d_test" in bert:
            rows.append({"model": "DistilBERT", "train_on": "A only", "test_on": "D test (SMS, unseen)", **common.metrics_from_prob(d_te["label"].values, bert["d_test"])})

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORT_DIR / "cross_dataset_results.csv", index=False)

    lr_rows = results[results["model"] == "Tuned TF-IDF + LR"]
    labels = [f"{r.train_on}\n-> {r.test_on}" for r in lr_rows.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#2e7d32" if "in-dataset" in t or t in ("A test", "B test") else "#c62828" for t in lr_rows["test_on"]]
    ax.bar(range(len(lr_rows)), lr_rows["f1"], color=colors)
    ax.set_xticks(range(len(lr_rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-dataset generalization (green = seen distribution, red = unseen dataset)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "cross_dataset_f1.png", dpi=150)
    plt.close(fig)

    gap_a = lr_rows[(lr_rows["train_on"] == "A only")]
    if len(gap_a) == 2:
        drop = float(gap_a.iloc[0]["f1"] - gap_a.iloc[1]["f1"])
        print("F1 drop when moving A -> B:", round(drop, 4))
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
