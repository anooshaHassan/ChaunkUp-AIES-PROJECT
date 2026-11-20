import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def build_matrix(blocks, names):
    cols = []
    for n in names:
        cols.append(blocks[n])
    return np.column_stack(cols)


def new_meta():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main():
    va = common.load_split("a_val")
    te = common.load_split("a_test")
    yva, yte = va["label"].values, te["label"].values

    text_probs = joblib.load(config.PROC_DIR / "text_probs.joblib")
    bert_path = config.PROC_DIR / "bert_probs.joblib"
    bert = joblib.load(bert_path) if bert_path.exists() else None
    print("DistilBERT probabilities available:", bert is not None)

    scorer = common.UrlScorer()
    uv = common.url_features_from_lists([common.split_url_field(x) for x in va["urls"]], scorer)
    ut = common.url_features_from_lists([common.split_url_field(x) for x in te["urls"]], scorer)

    classical_keys = [k for k in text_probs["val"] if not k.startswith("Baseline")]
    blocks = {"val": {}, "test": {}}
    for split, urlf in (("val", uv), ("test", ut)):
        for i, key in enumerate(classical_keys):
            blocks[split][f"classical::{key}"] = common.logit(text_probs[split][key])
        for j, name in enumerate(common.URL_META_NAMES):
            blocks[split][f"url::{name}"] = urlf[:, j]
        if bert is not None:
            blocks[split]["bert"] = common.logit(bert[split])

    classical_names = [f"classical::{k}" for k in classical_keys]
    url_names = [f"url::{n}" for n in common.URL_META_NAMES]
    combos = {
        "Classical text ensemble": classical_names,
        "URL model only": url_names,
        "Classical text + URL": classical_names + url_names,
    }
    if bert is not None:
        combos.update(
            {
                "DistilBERT only (recalibrated)": ["bert"],
                "DistilBERT + classical text": ["bert"] + classical_names,
                "DistilBERT + URL": ["bert"] + url_names,
                "Full hybrid: DistilBERT + classical + URL": ["bert"] + classical_names + url_names,
            }
        )

    rows = []
    fitted = {}
    for combo, names in combos.items():
        Zva = build_matrix(blocks["val"], names)
        Zte = build_matrix(blocks["test"], names)
        meta = new_meta().fit(Zva, yva)
        pt = meta.predict_proba(Zte)[:, 1]
        fitted[combo] = (meta, names, pt)
        rows.append({"model": combo, "n_features": len(names), **common.metrics_from_prob(yte, pt)})
        print(combo, {k: round(v, 4) for k, v in rows[-1].items() if k in ("accuracy", "f1", "roc_auc")})

    singles = []
    for key in classical_keys + [k for k in text_probs["val"] if k.startswith("Baseline")]:
        singles.append({"model": key, "n_features": 1, **common.metrics_from_prob(yte, text_probs["test"][key])})
    if bert is not None:
        singles.append({"model": "DistilBERT fine-tuned (raw)", "n_features": 1, **common.metrics_from_prob(yte, bert["test"])})

    ablation = pd.DataFrame(singles + rows)
    ablation.to_csv(config.REPORT_DIR / "ablation_results.csv", index=False)

    final_name = "Full hybrid: DistilBERT + classical + URL" if bert is not None else "Classical text + URL"
    meta, names, pt = fitted[final_name]
    Zva = build_matrix(blocks["val"], names)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    pv_cv = cross_val_predict(new_meta(), Zva, yva, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        pred = (pv_cv >= t).astype(int)
        fn = int(((pred == 0) & (yva == 1)).sum())
        fp = int(((pred == 1) & (yva == 0)).sum())
        costs.append(config.COST_FN * fn + config.COST_FP * fp)
    best_t = float(thresholds[int(np.argmin(costs))])

    default_m = common.metrics_from_prob(yte, pt, 0.5)
    tuned_m = common.metrics_from_prob(yte, pt, best_t)
    cm_default = confusion_matrix(yte, (pt >= 0.5).astype(int)).tolist()
    cm_tuned = confusion_matrix(yte, (pt >= best_t).astype(int)).tolist()
    threshold_report = {
        "final_model": final_name,
        "cost_fn": config.COST_FN,
        "cost_fp": config.COST_FP,
        "chosen_threshold": best_t,
        "test_metrics_at_0.5": default_m,
        "test_metrics_at_chosen_threshold": tuned_m,
        "confusion_matrix_at_0.5": cm_default,
        "confusion_matrix_at_chosen_threshold": cm_tuned,
        "brier_score": float(brier_score_loss(yte, pt)),
        "ece": expected_calibration_error(yte, pt),
    }
    common.save_json(threshold_report, config.REPORT_DIR / "threshold_and_calibration.json")
    common.save_json({"threshold": best_t}, config.MODEL_DIR / "threshold.json")

    bundle = {"final_name": final_name}
    bundle["no_bert"] = None
    nb_name = "Classical text + URL"
    m_nb, names_nb, _ = fitted[nb_name]
    bundle["no_bert"] = {"model": m_nb, "names": names_nb, "classical_keys": classical_keys}
    if bert is not None:
        bundle["with_bert"] = {"model": meta, "names": names, "classical_keys": classical_keys}
    joblib.dump(bundle, config.MODEL_DIR / "stack_meta.joblib")

    has_url_te = ut[:, 2] == 1
    subset_rows = []
    for label, mask in (("messages with URLs", has_url_te), ("messages without URLs", ~has_url_te)):
        if mask.sum() > 0 and len(np.unique(yte[mask])) > 1:
            ref_name = "Classical text ensemble"
            if bert is not None:
                ref_name = "DistilBERT + classical text"
            p_ref = fitted[ref_name][2]
            subset_rows.append(
                {"subset": label, "n": int(mask.sum()), "model": ref_name, **common.metrics_from_prob(yte[mask], p_ref[mask])}
            )
            subset_rows.append(
                {"subset": label, "n": int(mask.sum()), "model": final_name, **common.metrics_from_prob(yte[mask], pt[mask])}
            )
    pd.DataFrame(subset_rows).to_csv(config.REPORT_DIR / "url_subset_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ref_curves = [("Final hybrid", pt)]
    lr_key = "TF-IDF + Logistic Regression (tuned)"
    if lr_key in text_probs["test"]:
        ref_curves.append(("Tuned LR", text_probs["test"][lr_key]))
    for label, p in ref_curves:
        frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    ax.set_xlabel("predicted scam probability")
    ax.set_ylabel("observed scam fraction")
    ax.set_title("Reliability curve (test)")
    ax.legend()
    ax = axes[1]
    ax.plot(thresholds, costs, color="#c62828")
    ax.axvline(best_t, linestyle="--", color="k")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel(f"validation cost (FN x{config.COST_FN:g}, FP x{config.COST_FP:g})")
    ax.set_title(f"Cost-sensitive threshold = {best_t:.2f}")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "calibration_and_threshold.png", dpi=150)
    plt.close(fig)

    plot_df = ablation.copy()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(plot_df) + 1.5))
    ax.barh(plot_df["model"][::-1], plot_df["f1"][::-1], color="#1565c0")
    ax.set_xlabel("test F1")
    ax.set_xlim(max(plot_df["f1"].min() - 0.03, 0), 1.0)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "ablation_f1.png", dpi=150)
    plt.close(fig)

    print(ablation.round(4).to_string(index=False))
    print("Final model:", final_name, "| chosen threshold:", best_t)
    print("ECE:", round(threshold_report["ece"], 4), "Brier:", round(threshold_report["brier_score"], 4))


if __name__ == "__main__":
    main()
