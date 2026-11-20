import inspect
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import config
import common


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=config.BERT_MAX_LEN)
        self.labels = None if labels is None else list(map(int, labels))

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def make_args(out_dir):
    kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH,
        per_device_eval_batch_size=config.BERT_BATCH * 2,
        learning_rate=config.BERT_LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[key] = "epoch"
    return TrainingArguments(**kwargs)


def predict_scam_prob(trainer, tokenizer, texts):
    ds = TextDataset(texts, None, tokenizer)
    logits = trainer.predict(ds).predictions
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e[:, 1] / e.sum(axis=1)).astype(float)


def main():
    train_rows = config.BERT_TRAIN_ROWS
    if len(sys.argv) > 1:
        train_rows = int(sys.argv[1])
    model_name = config.BERT_MODEL_NAME
    if len(sys.argv) > 2:
        model_name = sys.argv[2]

    tr = common.load_split("a_train")
    va = common.load_split("a_val")
    te = common.load_split("a_test")
    bte = common.load_split("b_test")

    if train_rows and len(tr) > train_rows:
        tr = tr.sample(train_rows, random_state=config.SEED)
    print("DistilBERT training rows:", len(tr), "| device:", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = TextDataset(tr["text"], tr["label"], tokenizer)
    val_ds = TextDataset(va["text"], va["label"], tokenizer)

    out_dir = config.MODEL_DIR / "distilbert"
    trainer = Trainer(
        model=model,
        args=make_args(out_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    t0 = time.time()
    trainer.train()
    fit_seconds = time.time() - t0

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    out = {"val": predict_scam_prob(trainer, tokenizer, va["text"])}
    out["test"] = predict_scam_prob(trainer, tokenizer, te["text"])
    out["b_test"] = predict_scam_prob(trainer, tokenizer, bte["text"])
    d_path = config.PROC_DIR / "d_test.csv"
    dte = common.load_split("d_test") if d_path.exists() else None
    if dte is not None:
        out["d_test"] = predict_scam_prob(trainer, tokenizer, dte["text"])
        print("DistilBERT D-test (SMS):", {k: round(v, 4) for k, v in common.metrics_from_prob(dte["label"].values, out["d_test"]).items()})

    m_a = common.metrics_from_prob(te["label"].values, out["test"])
    m_b = common.metrics_from_prob(bte["label"].values, out["b_test"])
    print("DistilBERT A-test:", {k: round(v, 4) for k, v in m_a.items()})
    print("DistilBERT B-test (cross-dataset):", {k: round(v, 4) for k, v in m_b.items()})

    scam_te = te[te["label"] == 1]
    robust = []
    for level in config.ROBUST_LEVELS:
        obf = common.obfuscate_series(scam_te["text"], level, config.SEED + int(level * 1000))
        p_scam = predict_scam_prob(trainer, tokenizer, obf)
        safe_te = te[te["label"] == 0]
        p_safe = out["test"][(te["label"] == 0).values]
        y = np.concatenate([np.ones(len(p_scam)), np.zeros(len(p_safe))])
        p = np.concatenate([p_scam, p_safe])
        row = common.metrics_from_prob(y, p)
        row["level"] = level
        row["scam_recall"] = float((p_scam >= 0.5).mean())
        robust.append(row)
        print("obfuscation", level, "scam recall", round(row["scam_recall"], 4))
    out["robustness"] = robust

    joblib.dump(out, config.PROC_DIR / "bert_probs.joblib")
    pd.DataFrame(
        [
            {"model": "DistilBERT fine-tuned", "test_set": "A (in-dataset)", "fit_seconds": round(fit_seconds, 1), **m_a},
            {"model": "DistilBERT fine-tuned", "test_set": "B (cross-dataset)", "fit_seconds": round(fit_seconds, 1), **m_b},
        ]
    ).to_csv(config.REPORT_DIR / "bert_results.csv", index=False)
    pd.DataFrame(robust).to_csv(config.REPORT_DIR / "bert_robustness.csv", index=False)
    print("Saved model to", out_dir)


if __name__ == "__main__":
    main()
