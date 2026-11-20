import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("CHAUNKUP_RAW", ROOT / "data" / "raw"))
WORK = Path(os.environ.get("CHAUNKUP_WORK", ROOT / "work"))
PROC_DIR = WORK / "processed"
MODEL_DIR = WORK / "models"
REPORT_DIR = WORK / "reports"
FIG_DIR = REPORT_DIR / "figures"

for d in (PROC_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

DATASET_A_FILE = RAW_DIR / "dataset_a_phishing_email_big.csv"
DATASET_B_FILE = RAW_DIR / "dataset_b_phishing_email_detection.csv"
DATASET_C_FILE = RAW_DIR / "dataset_c_malicious_urls.csv"
DATASET_D_FILE = RAW_DIR / "dataset_d_sms_spam.csv"

TEXT_COLUMN_CANDIDATES = ["text_combined", "Email Text", "text", "body", "message", "Message", "email_text", "v2"]
LABEL_COLUMN_CANDIDATES = ["label", "Email Type", "class", "target", "Category", "v1", "type"]
URL_COLUMN_CANDIDATES = ["url", "URL", "link"]
URL_LABEL_CANDIDATES = ["type", "label", "Label", "class"]

SCAM_LABELS = {"phishing email", "phishing", "spam", "scam", "1", "1.0", "malicious", "fraud"}
SAFE_LABELS = {"safe email", "ham", "legitimate", "safe", "benign", "0", "0.0", "normal"}

URL_MALICIOUS_TYPES = {"phishing", "malware"}
URL_BENIGN_TYPES = {"benign"}

MIN_TEXT_CHARS = 15
MAX_TEXT_CHARS = 2000

TEST_SIZE = 0.15
VAL_SIZE = 0.15

TFIDF_MAX_FEATURES = 200000
TUNE_ITER = 4
TUNE_CV = 3
XGB_FEATURES = 5000

URL_TFIDF_FEATURES = 150000
URL_MAX_ROWS = 100000

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LEN = 256
BERT_EPOCHS = 2
BERT_BATCH = 32
BERT_LR = 3e-5
BERT_TRAIN_ROWS = 40000

ROBUST_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]

COST_FN = 5.0
COST_FP = 1.0
