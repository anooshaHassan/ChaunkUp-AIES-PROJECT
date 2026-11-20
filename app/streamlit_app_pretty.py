import re
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import common  # noqa: E402

st.set_page_config(page_title="ChaunkUp - Scam Shield", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,600;0,700;1,500&family=Nunito:wght@400;600;700;800&display=swap');

:root{
  --plum:#3b2a50; --plum-soft:#6b5b7e; --purple:#6d3fd0; --purple-soft:#efe6ff;
  --cream:#fff3f8; --pink:#f9c6da; --lav:#e6dbfa; --red:#e11d2e; --green:#16a34a; --yellow:#fde047;
}
html, body, .stApp, [class*="css"] { font-family:'Nunito', system-ui, sans-serif; }
.stApp{
  color:var(--plum);
  background:
    radial-gradient(circle at 0% 100%, #fbd5e3 0, #fbd5e3 13%, transparent 13.2%),
    var(--cream);
  background-attachment: fixed;
}
header[data-testid="stHeader"], #MainMenu, footer { visibility:hidden; height:0; }
.block-container{ padding-top:1.6rem; padding-bottom:2rem; max-width:1250px; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--plum); }

/* ---- hero ---- */
.hero{ display:flex; flex-direction:column; gap:.1rem; margin-bottom:1rem; }
.hero .shield{ width:58px; margin-bottom:.4rem; }
.hero h1{ font-family:'Lora', Georgia, serif; font-weight:700; font-size:4.4rem; line-height:1.02;
  color:var(--plum) !important; margin:0; padding:0; }
.hero .tag{ font-family:'Lora', Georgia, serif; font-style:italic; font-size:1.45rem; color:var(--purple); margin:.35rem 0 .2rem; }
.hero .sub{ font-size:1.15rem; color:var(--plum); margin:.1rem 0 .7rem; }
.pills{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.6rem; }
.pill{ border:2px solid var(--purple); color:var(--purple); background:#fff; border-radius:999px;
  padding:.18rem .95rem; font-weight:800; font-size:.95rem; }
.team{ font-size:.9rem; color:var(--plum-soft); }
.team b{ color:var(--plum); }

/* ---- inputs ---- */
.stApp h3{ font-family:'Lora', Georgia, serif; font-weight:700; }
div[data-testid="stTextArea"] textarea{
  background:#fff; color:var(--plum); border:2px solid var(--plum); border-radius:18px; font-size:1.02rem; }
div[data-testid="stTextArea"] textarea:focus{ border-color:var(--purple); box-shadow:0 0 0 3px #d9c7ff; }
.stButton > button, .stButton > button p{ font-family:'Nunito', sans-serif; }
.stButton > button{ border-radius:999px; font-weight:800; padding:.55rem 1.1rem; transition:transform .12s ease, box-shadow .12s ease; }
.stButton > button:hover{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(59,42,80,.18); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--purple); border:2px solid var(--purple); color:#fff !important; }
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p{ color:#fff !important; }
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"]{
  background:#fff; border:2px solid var(--purple); color:var(--purple) !important; }
.stButton > button[kind="secondary"] p, .stButton > button[data-testid="stBaseButton-secondary"] p{ color:var(--purple) !important; }

/* ---- lavender result panel (right column) ---- */
div[data-testid="stColumn"]:has(.rp-marker), div[data-testid="column"]:has(.rp-marker){
  background:var(--lav); border-radius:34px; padding:1.6rem 1.5rem 1.4rem; box-shadow:0 10px 30px rgba(109,63,208,.12); }
div[data-testid="stColumn"]:has(.rp-marker) .stButton > button[kind="primary"],
div[data-testid="column"]:has(.rp-marker) .stButton > button[kind="primary"],
div[data-testid="stColumn"]:has(.rp-marker) .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--red); border-color:var(--red); font-size:1.1rem; }

.rcard{ background:#fff; border:4px solid var(--red); border-radius:28px; padding:1.2rem 1.4rem 1.1rem;
  transform:rotate(-1deg); box-shadow:0 16px 30px rgba(59,42,80,.16); margin-bottom:1.2rem; }
.rcard.safe{ border-color:var(--green); }
.rcard.idle{ border-color:var(--purple); text-align:center; padding:2.4rem 1.4rem; }
.rhead{ display:flex; align-items:center; gap:.8rem; margin-bottom:.7rem; flex-wrap:wrap; }
.badge{ background:var(--red); color:#fff; font-weight:800; font-size:1.35rem; border-radius:999px; padding:.15rem 1.1rem; }
.rcard.safe .badge{ background:var(--green); }
.conf{ color:var(--plum-soft); font-size:1.05rem; font-weight:600; }
.rtext{ font-size:1.1rem; line-height:2.0; color:var(--plum); word-break:break-word; }
.mk{ border-radius:8px; padding:2px 4px; box-decoration-break:clone; -webkit-box-decoration-break:clone; }
.bar-bg{ background:#eee6f7; border-radius:999px; height:12px; overflow:hidden; margin:.9rem 0 .3rem; }
.bar-fill{ height:100%; border-radius:999px; }
.level{ font-size:.85rem; color:var(--plum-soft); }

.chip{ display:inline-block; padding:.12rem .7rem; border-radius:999px; font-size:.85rem; font-weight:800; margin:.15rem; }
.chip-red{ background:#fff3b0; color:#6b4e00; border:1.5px solid #e9c400; }
.chip-green{ background:#dcfce7; color:#14532d; border:1.5px solid #22c55e; }
.link-row{ display:flex; justify-content:space-between; align-items:center; gap:.6rem; padding:.55rem .9rem;
  border-radius:14px; background:#fff; margin-bottom:.45rem; word-break:break-all; border:1.5px solid #d9c7ff; }
.vote{ background:#fff; border-radius:14px; padding:.55rem .9rem .35rem; margin-bottom:.45rem; border:1.5px solid #d9c7ff; }
.vote .bar-bg{ margin:.35rem 0 .3rem; }

/* tabs */
button[data-baseweb="tab"]{ font-weight:800; color:var(--plum-soft); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--purple); }
div[data-baseweb="tab-highlight"]{ background-color:var(--purple) !important; }
div[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p{ color:var(--plum); }

/* sidebar */
section[data-testid="stSidebar"]{ background:#fff; border-right:2px solid #f1d4e1; }
section[data-testid="stSidebar"] *{ color:var(--plum); }

/* stat tiles */
.tiles{ display:flex; gap:.8rem; flex-wrap:wrap; margin-top:1.4rem; }
.tile{ flex:1; min-width:150px; background:#fff; border:2px solid #f1d4e1; border-radius:20px; padding:.7rem .9rem; }
.tile b{ display:block; font-family:'Lora', Georgia, serif; font-size:1.6rem; color:var(--purple); line-height:1.1; }
.tile span{ font-size:.82rem; color:var(--plum-soft); }

@media (max-width: 900px){ .hero h1{ font-size:3rem; } }
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLES = {
    "🏦 Bank alert": "URGENT: Your bank account has been suspended. Verify your password now at http://secure-login-bank-update.com/verify to avoid permanent closure.",
    "🎁 Free prize": "Congratulations! You won a free $1000 gift card. Click http://free-bonus-wallet.xyz/claim now before it expires!",
    "💼 Fake job": "Dear candidate, you are selected for a work-from-home job paying $500 per day. Send your bank details and a small registration fee to start today.",
    "✅ Normal message": "Hi, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "📧 Work email": "Hi team, please find the meeting notes attached. The next sync is on Thursday at 3 pm. Thanks, Sara",
}

SHIELD = (
    '<svg class="shield" viewBox="0 0 64 72" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M32 4 L58 14 V36 C58 52 46 64 32 68 C18 64 6 52 6 36 V14 Z" fill="#f9c6da" '
    'stroke="#3b2a50" stroke-width="4" stroke-linejoin="round"/>'
    '<rect x="29" y="19" width="6" height="23" rx="3" fill="#3b2a50"/>'
    '<circle cx="32" cy="52" r="4.2" fill="#3b2a50"/></svg>'
)


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    md = config.MODEL_DIR
    res = {
        "vec": joblib.load(md / "tfidf.joblib"),
        "nb": joblib.load(md / "nb.joblib"),
        "lr": joblib.load(md / "lr_tuned.joblib"),
        "svm": joblib.load(md / "svm_cal.joblib"),
        "xgb": joblib.load(md / "xgb_bundle.joblib"),
        "stack": joblib.load(md / "stack_meta.joblib"),
        "threshold": float(common.load_json(md / "threshold.json")["threshold"]),
        "scorer": common.UrlScorer(),
        "bert": None,
    }
    res["feats"] = np.array(res["vec"].get_feature_names_out())
    bert_dir = md / "distilbert"
    if "with_bert" in res["stack"] and (bert_dir / "config.json").exists():
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(bert_dir))
            mdl = AutoModelForSequenceClassification.from_pretrained(str(bert_dir)).eval()
            res["bert"] = (tok, mdl, torch)
        except Exception:
            res["bert"] = None
    return res


def bert_prob(res, text):
    tok, mdl, torch = res["bert"]
    enc = tok([text], truncation=True, max_length=config.BERT_MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits[0]
    return float(torch.softmax(logits, dim=-1)[1])


def analyze(res, raw):
    masked, urls = common.prepare_message(raw)
    X = res["vec"].transform([masked])
    xgb = res["xgb"]
    base = {
        "TF-IDF + Naive Bayes (tuned)": float(res["nb"].predict_proba(X)[0, 1]),
        "TF-IDF + Logistic Regression (tuned)": float(res["lr"].predict_proba(X)[0, 1]),
        "TF-IDF + Linear SVM (tuned, calibrated)": float(res["svm"].predict_proba(X)[0, 1]),
        "TF-IDF + XGBoost (tuned)": float(xgb["model"].predict_proba(xgb["selector"].transform(X))[0, 1]),
    }
    url_scores = res["scorer"].score(urls) if urls else np.zeros(0)
    urlf = common.url_features_from_lists([urls], res["scorer"])[0]

    use_bert = res["bert"] is not None
    entry = res["stack"]["with_bert"] if use_bert else res["stack"]["no_bert"]
    bp = bert_prob(res, masked) if use_bert else None
    if use_bert:
        base["DistilBERT"] = bp

    row = []
    for n in entry["names"]:
        if n.startswith("classical::"):
            row.append(common.logit(base[n.split("::", 1)[1]]))
        elif n.startswith("url::"):
            row.append(float(urlf[common.URL_META_NAMES.index(n.split("::", 1)[1])]))
        elif n == "bert":
            row.append(common.logit(bp))
    p = float(entry["model"].predict_proba(np.array([row]))[0, 1])
    return {"p": p, "masked": masked, "urls": urls, "url_scores": url_scores, "base": base, "X": X, "use_bert": use_bert}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def highlight(res, masked, X, show_red=True):
    """Yellow marker = pushes towards scam, green marker = pushes towards safe."""
    coef = res["lr"].coef_[0]
    feats = res["feats"]
    row = X.tocsr()
    contrib = {}
    for j, v in zip(row.indices, row.data):
        if " " not in feats[j]:
            contrib[feats[j]] = float(coef[j] * v)
    mx = max([abs(v) for v in contrib.values()] + [1e-9])

    toks = re.findall(r"\S+|\s+", masked[:1500])
    marks = []  # (token, sign, strength) ; sign 0 = none
    for tok in toks:
        if tok.strip().lower() == "urltoken":
            marks.append((tok, "link", 0))
            continue
        m = re.search(r"[A-Za-z0-9']+", tok)
        key = m.group(0).lower() if m else None
        v = contrib.get(key)
        if v is None or abs(v) < 0.15 * mx or (v > 0 and not show_red):
            marks.append((tok, 0, 0))
        else:
            marks.append((tok, 1 if v > 0 else -1, min(abs(v) / mx, 1.0)))

    # join neighbouring highlighted words across the space between them
    for i in range(1, len(marks) - 1):
        t, s, _ = marks[i]
        if t.isspace() and marks[i - 1][1] in (1, -1) and marks[i - 1][1] == marks[i + 1][1]:
            marks[i] = (t, marks[i - 1][1], min(marks[i - 1][2], marks[i + 1][2]))

    parts = []
    for tok, sign, strength in marks:
        if sign == "link":
            parts.append('<b style="color:#6d3fd0">🔗 [link]</b>')
        elif sign == 0:
            parts.append(esc(tok))
        else:
            a = 0.55 + 0.45 * strength
            color = f"rgba(253,224,71,{a:.2f})" if sign == 1 else f"rgba(134,239,172,{a:.2f})"
            parts.append(f'<span class="mk" style="background:{color}">{esc(tok)}</span>')

    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    scam_words = [w for w, v in ranked if v > 0][:6] if show_red else []
    safe_words = [w for w, v in ranked[::-1] if v < 0][:6]
    return "".join(parts), scam_words, safe_words


def bar(pct, color):
    return f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def risk_color(p):
    if p >= 0.7:
        return "#e11d2e"
    if p >= 0.4:
        return "#f59e0b"
    return "#16a34a"


# ------------------------------------------------------------------ state
res = load_all()
thr = res["threshold"]

st.session_state.setdefault("msg", "")
st.session_state.setdefault("out", None)


def set_example(text):
    st.session_state["msg"] = text
    st.session_state["out"] = None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛡️ ChaunkUp")
    st.markdown("Hybrid scam detector that reads the **message text** and checks every **link** inside it.")
    st.markdown("**Models used**")
    st.markdown("- TF-IDF + Naive Bayes, Logistic Regression, SVM, XGBoost\n- URL model (lexical features + char n-grams)")
    if res["bert"] is not None:
        st.markdown("- DistilBERT (fine-tuned)")
    st.markdown("- Stacked meta-learner with cost-sensitive threshold")
    st.markdown("---")
    mode = st.radio("Sensitivity", ["Balanced (fewer false alarms)", "Cautious (cost-based threshold)"], index=0)
    balanced = mode.startswith("Balanced")
    active_thr = 0.5 if balanced else thr
    st.markdown(f"**Alert threshold:** {active_thr:.2f}")
    st.caption(f"Cost-based threshold from training: {thr:.2f}. A missed scam is treated as 5x worse than a false alarm, which lowers it.")

# ------------------------------------------------------------------ layout
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown(
        f"""
<div class="hero">
  {SHIELD}
  <h1>ChaunkUp</h1>
  <div class="tag">Chaunk up. Don't fall for it.</div>
  <div class="sub">An AI scam shield that reads the message and checks every link.</div>
  <div class="pills"><span class="pill">Text</span><span class="pill">Links</span><span class="pill">Explained</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### Paste a message")
    st.text_area("Message", key="msg", height=170, label_visibility="collapsed", placeholder="Paste an email, SMS or chat message here...")
    go = st.button("🔍 Check this message", type="primary", use_container_width=True)
    st.markdown("**Or try an example**")
    cols = st.columns(2)
    for i, (label, text) in enumerate(EXAMPLES.items()):
        cols[i % 2].button(label, on_click=set_example, args=(text,), use_container_width=True, key=f"ex{i}")
    st.markdown(
        """
<div class="tiles">
  <div class="tile"><b>99.46%</b><span>F1-score on 12,288 unseen test emails</span></div>
  <div class="tile"><b>99.7%</b><span>of scams caught in Cautious mode</span></div>
  <div class="tile"><b>Why?</b><span>every verdict shows the words behind it</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

if go and st.session_state["msg"].strip():
    with st.spinner("Analysing..."):
        st.session_state["out"] = analyze(res, st.session_state["msg"])

with right:
    st.markdown('<span class="rp-marker"></span>', unsafe_allow_html=True)
    out = st.session_state["out"]

    if out is None:
        st.markdown(
            '<div class="rcard idle"><div style="font-size:3rem">🕵️</div>'
            '<p style="font-size:1.15rem;margin:.3rem 0 0"><b>Your result appears here</b><br>'
            'Paste a message or pick an example, then press <b>Check this message</b>.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        p = out["p"]
        short = len(out["urls"]) == 0 and len(out["masked"].split()) < 8
        need = max(active_thr, 0.9) if (short and balanced) else active_thr
        is_scam = p >= need

        html_text, scam_words, safe_words = highlight(res, out["masked"], out["X"], show_red=is_scam)
        bar_color = risk_color(p) if is_scam else "#16a34a"
        badge = "Scam" if is_scam else "Looks safe"
        cls = "" if is_scam else "safe"
        st.markdown(
            f'<div class="rcard {cls}"><div class="rhead"><span class="badge">{badge}</span>'
            f'<span class="conf">Scam probability: {p*100:.1f}%</span></div>'
            f'<div class="rtext">{html_text}</div>{bar(p * 100, bar_color)}'
            f'<div class="level">Alert level: {need*100:.0f}% &nbsp;·&nbsp; mode: {"Balanced" if balanced else "Cautious"}</div></div>',
            unsafe_allow_html=True,
        )
        if short and balanced:
            st.caption("Short message without a link: the models were trained on longer emails, so a scam verdict needs very high confidence here.")

        tab1, tab2, tab3, tab4 = st.tabs(["💡 Why?", "🔗 Links", "🤖 Model votes", "🧰 Stay safe"])

        with tab1:
            st.caption("Yellow = pushes towards scam, green = pushes towards safe.")
            if scam_words:
                st.markdown("**Suspicious words** " + "".join(f'<span class="chip chip-red">{w}</span>' for w in scam_words), unsafe_allow_html=True)
            if safe_words:
                st.markdown("**Reassuring words** " + "".join(f'<span class="chip chip-green">{w}</span>' for w in safe_words), unsafe_allow_html=True)

        with tab2:
            if len(out["urls"]) == 0:
                st.info("No links found in this message.")
            for u, s in zip(out["urls"], out["url_scores"]):
                c = risk_color(float(s))
                label = "RISKY" if s >= 0.5 else "OK"
                st.markdown(
                    f'<div class="link-row"><span>🔗 {esc(u)}</span>'
                    f'<span class="chip" style="background:{c}22;color:{c};border:1.5px solid {c}">{label} {s*100:.0f}%</span></div>',
                    unsafe_allow_html=True,
                )

        with tab3:
            for name, v in out["base"].items():
                nm = name.replace("TF-IDF + ", "")
                st.markdown(f'<div class="vote"><b>{nm}</b> &nbsp; {v*100:.1f}%{bar(v * 100, risk_color(v))}</div>', unsafe_allow_html=True)
            st.caption("The final verdict combines all of these with the link scores.")

        with tab4:
            if is_scam:
                st.markdown(
                    "- 🚫 Do **not** click any link or open attachments\n"
                    "- 🔑 Never share passwords, OTP codes or card details\n"
                    "- 📞 Contact the company using the number on its official website\n"
                    "- 🗑️ Report the message and delete it\n"
                    "- 🔒 If you already clicked, change your passwords right away"
                )
            else:
                st.markdown(
                    "- 👍 No strong scam signs found, but no tool is perfect\n"
                    "- 🔍 Still double-check unexpected requests for money or data\n"
                    "- 🔗 Hover over links before clicking them"
                )
