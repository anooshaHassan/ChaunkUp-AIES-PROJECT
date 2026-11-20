import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def profile_text(df, name):
    raw_text = df["text_raw"].fillna("").astype(str)
    return {
        "dataset": name,
        "rows_raw": int(len(df)),
        "null_text": int(df["text_raw"].isna().sum()),
        "null_label": int(df["label_raw"].isna().sum()),
        "exact_duplicate_rows": int(df.duplicated(subset=["text_raw"]).sum()),
        "rows_with_html_tags": int(raw_text.str.contains(r"<[^>]+>", regex=True).sum()),
        "rows_with_urls": int(raw_text.str.contains(r"https?://|www\.", regex=True, case=False).sum()),
        "rows_non_ascii": int(raw_text.apply(lambda s: any(ord(c) > 127 for c in s)).sum()),
        "label_values": {str(k): int(v) for k, v in df["label_raw"].astype(str).value_counts().items()},
        "mean_chars": float(raw_text.str.len().mean()),
        "median_chars": float(raw_text.str.len().median()),
    }


def clean_text_dataset(df, name, log):
    log[name] = {"profile": profile_text(df, name)}
    steps = {}
    df = df.dropna(subset=["text_raw", "label_raw"]).copy()
    steps["after_drop_null"] = len(df)
    df["label"] = df["label_raw"].apply(common.map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    steps["after_drop_unmapped_label"] = len(df)
    prepared = df["text_raw"].apply(common.prepare_message)
    df["text"] = [p[0] for p in prepared]
    df["urls"] = [" ".join(p[1]) for p in prepared]
    df = df[df["text"].str.len() >= config.MIN_TEXT_CHARS]
    steps["after_min_length"] = len(df)
    df["key"] = df["text"].str.lower()
    nunique_labels = df.groupby("key")["label"].nunique()
    conflicting = set(nunique_labels[nunique_labels > 1].index)
    steps["conflicting_label_texts_removed"] = int(len(conflicting))
    df = df[~df["key"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates(subset=["key"])
    steps["duplicates_removed"] = int(before - len(df))
    steps["final_rows"] = len(df)
    steps["final_scam_share"] = float(df["label"].mean())
    log[name]["cleaning_steps"] = steps
    return df[["text", "urls", "label", "key"]].reset_index(drop=True)


def three_way_split(df, seed, stratify_col="label"):
    train, rest = train_test_split(
        df, test_size=config.TEST_SIZE + config.VAL_SIZE, stratify=df[stratify_col], random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        stratify=rest[stratify_col],
        random_state=seed,
    )
    return train, val, test


def clean_url_dataset(df, log):
    log["dataset_c"] = {
        "profile": {
            "rows_raw": int(len(df)),
            "null_url": int(df["url"].isna().sum()),
            "exact_duplicate_rows": int(df.duplicated(subset=["url"]).sum()),
            "type_values": {str(k): int(v) for k, v in df["type_raw"].astype(str).value_counts().items()},
            "mean_chars": float(df["url"].fillna("").astype(str).str.len().mean()),
        }
    }
    steps = {}
    df = df.dropna(subset=["url", "type_raw"]).copy()
    df["type_raw"] = df["type_raw"].astype(str).str.strip().str.lower()
    keep = config.URL_MALICIOUS_TYPES | config.URL_BENIGN_TYPES
    steps["dropped_other_types_(e.g._defacement)"] = int((~df["type_raw"].isin(keep)).sum())
    df = df[df["type_raw"].isin(keep)]
    df["label"] = df["type_raw"].isin(config.URL_MALICIOUS_TYPES).astype(int)
    df["key"] = df["url"].astype(str).apply(common.strip_scheme).str.lower().str.rstrip("/")
    nunique_labels = df.groupby("key")["label"].nunique()
    conflicting = set(nunique_labels[nunique_labels > 1].index)
    steps["conflicting_label_urls_removed"] = int(len(conflicting))
    df = df[~df["key"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates(subset=["key"])
    steps["duplicates_removed"] = int(before - len(df))
    if config.URL_MAX_ROWS and len(df) > config.URL_MAX_ROWS:
        df = df.sample(config.URL_MAX_ROWS, random_state=config.SEED)
        steps["subsampled_to"] = int(config.URL_MAX_ROWS)
    steps["final_rows"] = len(df)
    steps["final_malicious_share"] = float(df["label"].mean())
    log["dataset_c"]["cleaning_steps"] = steps
    return df[["url", "label", "key"]].reset_index(drop=True)


def main():
    for p in (config.DATASET_A_FILE, config.DATASET_B_FILE, config.DATASET_C_FILE):
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    log = {}

    a = clean_text_dataset(common.load_text_dataset(config.DATASET_A_FILE), "dataset_a", log)
    b = clean_text_dataset(common.load_text_dataset(config.DATASET_B_FILE), "dataset_b", log)

    overlap = b["key"].isin(set(a["key"]))
    log["dataset_b"]["cleaning_steps"]["overlap_with_dataset_a_removed"] = int(overlap.sum())
    b = b[~overlap].reset_index(drop=True)
    log["dataset_b"]["cleaning_steps"]["final_rows_after_overlap_removal"] = len(b)

    c = clean_url_dataset(common.load_url_dataset(config.DATASET_C_FILE), log)

    d = None
    if config.DATASET_D_FILE.exists():
        d = clean_text_dataset(common.load_text_dataset(config.DATASET_D_FILE), "dataset_d", log)
        seen = set(a["key"]) | set(b["key"])
        ov = d["key"].isin(seen)
        log["dataset_d"]["cleaning_steps"]["overlap_with_a_or_b_removed"] = int(ov.sum())
        d = d[~ov].reset_index(drop=True)
        d.drop(columns=["key"]).to_csv(config.PROC_DIR / "d_test.csv", index=False)
        log["split_d"] = {"test_only": len(d)}
        print("dataset_d (SMS, test only):", log["dataset_d"]["profile"]["rows_raw"], "->", len(d))

    for name, df, col in (("a", a, "label"), ("b", b, "label"), ("c", c, "label")):
        train, val, test = three_way_split(df, config.SEED, col)
        for split_name, part in (("train", train), ("val", val), ("test", test)):
            part.drop(columns=["key"]).to_csv(config.PROC_DIR / f"{name}_{split_name}.csv", index=False)
        log[f"split_{name}"] = {"train": len(train), "val": len(val), "test": len(test)}

    raw_total = (
        log["dataset_a"]["profile"]["rows_raw"]
        + log["dataset_b"]["profile"]["rows_raw"]
        + log["dataset_c"]["profile"]["rows_raw"]
    )
    log["total_raw_rows_all_datasets"] = raw_total
    log["text_rows_raw_a_plus_b"] = (
        log["dataset_a"]["profile"]["rows_raw"] + log["dataset_b"]["profile"]["rows_raw"]
    )
    common.save_json(log, config.REPORT_DIR / "cleaning_log.json")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, df) in zip(axes, (("Dataset A (text)", a), ("Dataset B (text)", b), ("Dataset C (URLs)", c))):
        counts = df["label"].value_counts().sort_index()
        ax.bar(["safe/benign", "scam/malicious"][: len(counts)], counts.values, color=["#2e7d32", "#c62828"][: len(counts)])
        ax.set_title(name)
        ax.set_ylabel("rows after cleaning")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(a["text"].str.len().clip(upper=config.MAX_TEXT_CHARS), bins=50, alpha=0.7, label="Dataset A")
    ax.hist(b["text"].str.len().clip(upper=config.MAX_TEXT_CHARS), bins=50, alpha=0.7, label="Dataset B")
    ax.set_xlabel("characters (after cleaning)")
    ax.set_ylabel("messages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "text_length_distribution.png", dpi=150)
    plt.close(fig)

    print("Cleaning summary")
    for name in ("dataset_a", "dataset_b", "dataset_c"):
        print(name, "raw:", log[name]["profile"]["rows_raw"], "->", log[name]["cleaning_steps"]["final_rows"])
    print("Total raw rows across datasets:", raw_total)
    print("Saved to", config.PROC_DIR)


if __name__ == "__main__":
    main()
