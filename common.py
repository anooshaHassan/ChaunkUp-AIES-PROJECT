import json
import math
import re
from collections import Counter
from urllib.parse import urlparse
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]]+", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SUSPICIOUS_TOKENS = [
    "login", "signin", "verify", "secure", "account", "update", "bank", "paypal",
    "confirm", "free", "bonus", "wallet", "password", "support", "billing", "unlock",
]

LEET = {"a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["$", "5"], "l": ["1"], "t": ["7"]}
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445"}
SEPS = [".", " ", "-", "*", "_"]


def strip_html(text):
    return HTML_TAG_RE.sub(" ", text)


def normalize_ws(text):
    return WS_RE.sub(" ", text).strip()


def extract_urls(text):
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]


def mask_urls(text):
    return URL_RE.sub(" urltoken ", text)


def clean_text(text):
    text = strip_html(str(text))
    text = normalize_ws(text)
    return text[: config.MAX_TEXT_CHARS]


def prepare_message(raw_text):
    raw = normalize_ws(strip_html(str(raw_text)))
    urls = extract_urls(raw)
    masked = normalize_ws(mask_urls(raw))[: config.MAX_TEXT_CHARS]
    return masked, urls


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def map_label(value):
    v = str(value).strip().lower()
    if v in config.SCAM_LABELS:
        return 1
    if v in config.SAFE_LABELS:
        return 0
    return np.nan


def read_csv_any_encoding(path):
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def load_text_dataset(path):
    df = read_csv_any_encoding(path)
    tcol = find_column(df, config.TEXT_COLUMN_CANDIDATES)
    lcol = find_column(df, config.LABEL_COLUMN_CANDIDATES)
    out = pd.DataFrame({"text_raw": df[tcol], "label_raw": df[lcol]})
    return out


def load_url_dataset(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    ucol = find_column(df, config.URL_COLUMN_CANDIDATES)
    lcol = find_column(df, config.URL_LABEL_CANDIDATES)
    return pd.DataFrame({"url": df[ucol], "type_raw": df[lcol]})


def strip_scheme(url):
    u = str(url).strip()
    u = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", u)
    u = re.sub(r"^www\d?\.", "", u, flags=re.I)
    return u


def shannon_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


URL_FEATURE_NAMES = [
    "len", "host_len", "path_len", "query_len", "n_dots", "n_hyphen", "n_underscore",
    "n_slash", "n_qm", "n_eq", "n_amp", "n_at", "n_pct", "n_digits", "n_letters",
    "digit_ratio", "n_subdomains", "has_ip", "tld_len", "entropy", "n_susp", "max_label_len",
    "has_port", "n_params",
]


def url_lexical(url):
    u = strip_scheme(url)
    try:
        p = urlparse("http://" + u)
    except ValueError:
        netloc, _, tail = u.partition("/")
        path, _, query = tail.partition("?")
        p = SimpleNamespace(netloc=netloc, path=("/" + path) if tail else "", query=query)
    host = p.netloc.lower().split("@")[-1]
    has_port = int(":" in host)
    host = host.split(":")[0]
    labels = [x for x in host.split(".") if x]
    n_digits = sum(ch.isdigit() for ch in u)
    n_letters = sum(ch.isalpha() for ch in u)
    low = u.lower()
    return [
        len(u),
        len(host),
        len(p.path),
        len(p.query),
        u.count("."),
        u.count("-"),
        u.count("_"),
        u.count("/"),
        u.count("?"),
        u.count("="),
        u.count("&"),
        u.count("@"),
        u.count("%"),
        n_digits,
        n_letters,
        n_digits / max(len(u), 1),
        max(len(labels) - 2, 0),
        int(bool(IP_RE.match(host))),
        len(labels[-1]) if labels else 0,
        shannon_entropy(u),
        sum(low.count(t) for t in SUSPICIOUS_TOKENS),
        max((len(x) for x in labels), default=0),
        has_port,
        len(p.query.split("&")) if p.query else 0,
    ]


def url_feature_frame(urls):
    return pd.DataFrame([url_lexical(u) for u in urls], columns=URL_FEATURE_NAMES)


class UrlScorer:
    def __init__(self, path=None):
        path = path or (config.MODEL_DIR / "url_model.joblib")
        bundle = joblib.load(path)
        self.hgb = bundle["hgb"]
        self.vec = bundle["vec"]
        self.sgd = bundle["sgd"]

    def score(self, urls):
        if len(urls) == 0:
            return np.zeros(0)
        stripped = [strip_scheme(u) for u in urls]
        feats = url_feature_frame(urls)
        p1 = self.hgb.predict_proba(feats)[:, 1]
        p2 = self.sgd.predict_proba(self.vec.transform(stripped))[:, 1]
        return 0.5 * (p1 + p2)


def url_features_from_lists(url_lists, scorer):
    flat = [u for lst in url_lists for u in lst]
    scores = scorer.score(flat) if flat else np.zeros(0)
    rows = []
    pos = 0
    for lst in url_lists:
        k = len(lst)
        s = scores[pos : pos + k]
        pos += k
        if k == 0:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            rows.append([float(s.max()), float(s.mean()), 1.0, math.log1p(k)])
    return np.array(rows)


def split_url_field(value):
    value = str(value).strip()
    return value.split(" ") if value else []


URL_META_NAMES = ["url_max", "url_mean", "has_url", "n_urls_log"]


def metrics_from_prob(y_true, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def obfuscate(text, level, rng):
    if level <= 0:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        lc = c.lower()
        if c.isalpha() and rng.random() < level:
            op = int(rng.integers(0, 5))
            if op == 0 and lc in LEET:
                options = LEET[lc]
                out.append(options[int(rng.integers(0, len(options)))])
            elif op == 1 and lc in HOMO:
                out.append(HOMO[lc])
            elif op == 2:
                out.append(c)
                out.append(SEPS[int(rng.integers(0, len(SEPS)))])
            elif op == 3 and i + 1 < n and text[i + 1].isalpha():
                out.append(text[i + 1])
                out.append(c)
                i += 1
            else:
                out.append(c.swapcase())
        else:
            out.append(c)
        i += 1
    return "".join(out)


def obfuscate_series(texts, level, seed):
    rng = np.random.default_rng(seed)
    return [obfuscate(t, level, rng) for t in texts]


def load_split(name):
    return pd.read_csv(config.PROC_DIR / f"{name}.csv", keep_default_na=False)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
