import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )


def search(estimator, params, X, y, n_iter, n_jobs=-1):
    s = RandomizedSearchCV(
        estimator,
        params,
        n_iter=n_iter,
        cv=config.TUNE_CV,
        scoring="f1",
        n_jobs=n_jobs,
        random_state=config.SEED,
        refit=True,
    )
    s.fit(X, y)
    return s


def main():
    tr = common.load_split("a_train")
    va = common.load_split("a_val")
    te = common.load_split("a_test")
    ytr, yva, yte = tr["label"].values, va["label"].values, te["label"].values
    print("Dataset A rows:", len(tr), len(va), len(te))

    vec = tfidf_vectorizer()
    Xtr = vec.fit_transform(tr["text"])
    Xva = vec.transform(va["text"])
    Xte = vec.transform(te["text"])
    print("TF-IDF shape:", Xtr.shape)

    probs = {"val": {}, "test": {}}
    rows = []
    best_params = {}

    def record(name, model, Xv, Xt, fit_seconds, tuned):
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xt)[:, 1]
        probs["val"][name] = pv
        probs["test"][name] = pt
        rows.append({"model": name, "tuned": tuned, "fit_seconds": round(fit_seconds, 1), **common.metrics_from_prob(yte, pt)})
        print(name, {k: round(v, 4) for k, v in rows[-1].items() if k in ("accuracy", "f1", "roc_auc")})

    t0 = time.time()
    lr_base = LogisticRegression(solver="liblinear", max_iter=1000)
    lr_base.fit(Xtr, ytr)
    record("Baseline: TF-IDF + Logistic Regression", lr_base, Xva, Xte, time.time() - t0, False)

    t0 = time.time()
    nb = search(MultinomialNB(), {"alpha": loguniform(1e-3, 3)}, Xtr, ytr, config.TUNE_ITER)
    record("TF-IDF + Naive Bayes (tuned)", nb.best_estimator_, Xva, Xte, time.time() - t0, True)
    best_params["nb"] = nb.best_params_

    t0 = time.time()
    lr = search(
        LogisticRegression(solver="liblinear", max_iter=1000),
        {"C": loguniform(0.1, 100), "class_weight": [None, "balanced"]},
        Xtr, ytr, config.TUNE_ITER,
    )
    record("TF-IDF + Logistic Regression (tuned)", lr.best_estimator_, Xva, Xte, time.time() - t0, True)
    best_params["lr"] = lr.best_params_

    t0 = time.time()
    svm_search = search(
        LinearSVC(max_iter=5000),
        {"C": loguniform(0.05, 20), "class_weight": [None, "balanced"]},
        Xtr, ytr, config.TUNE_ITER,
    )
    svm_cal = CalibratedClassifierCV(LinearSVC(max_iter=5000, **svm_search.best_params_), cv=3)
    svm_cal.fit(Xtr, ytr)
    record("TF-IDF + Linear SVM (tuned, calibrated)", svm_cal, Xva, Xte, time.time() - t0, True)
    best_params["svm"] = svm_search.best_params_

    t0 = time.time()
    k = min(config.XGB_FEATURES, Xtr.shape[1])
    selector = SelectKBest(chi2, k=k).fit(Xtr, ytr)
    Str, Sva, Ste = selector.transform(Xtr), selector.transform(Xva), selector.transform(Xte)
    xgb = search(
        XGBClassifier(tree_method="hist", n_jobs=2, eval_metric="logloss", random_state=config.SEED),
        {
            "max_depth": [4, 6, 8],
            "learning_rate": loguniform(0.03, 0.3),
            "n_estimators": [200, 400],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.5, 1.0],
        },
        Str, ytr, max(config.TUNE_ITER // 2, 3), n_jobs=1,
    )
    record("TF-IDF + XGBoost (tuned)", xgb.best_estimator_, Sva, Ste, time.time() - t0, True)
    best_params["xgb"] = {k_: (float(v) if isinstance(v, (np.floating,)) else v) for k_, v in xgb.best_params_.items()}

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORT_DIR / "text_models_results.csv", index=False)
    common.save_json(best_params, config.REPORT_DIR / "text_best_params.json")
    joblib.dump(probs, config.PROC_DIR / "text_probs.joblib")
    joblib.dump(vec, config.MODEL_DIR / "tfidf.joblib")
    joblib.dump(lr_base, config.MODEL_DIR / "lr_baseline.joblib")
    joblib.dump(nb.best_estimator_, config.MODEL_DIR / "nb.joblib")
    joblib.dump(lr.best_estimator_, config.MODEL_DIR / "lr_tuned.joblib")
    joblib.dump(svm_cal, config.MODEL_DIR / "svm_cal.joblib")
    joblib.dump({"selector": selector, "model": xgb.best_estimator_}, config.MODEL_DIR / "xgb_bundle.joblib")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs["test"].items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, label=name.split(":")[-1].strip())
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves - text models (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "text_roc_curves.png", dpi=150)
    plt.close(fig)

    names = list(probs["test"].keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for ax, name in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay.from_predictions(yte, (probs["test"][name] >= 0.5).astype(int), display_labels=["safe", "scam"], ax=ax, colorbar=False)
        ax.set_title(name.split(":")[-1].strip()[:28], fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "text_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([n.split(":")[-1].strip() for n in results["model"]][::-1], results["f1"][::-1], color="#1565c0")
    ax.set_xlabel("test F1")
    ax.set_xlim(max(results["f1"].min() - 0.05, 0), 1.0)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "text_model_f1_comparison.png", dpi=150)
    plt.close(fig)

    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
