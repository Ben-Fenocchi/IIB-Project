from pathlib import Path
import re
import time
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from joblib import dump

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

from tqdm import tqdm


# ============================
# Config
# ============================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAINING_XLSX = BASE_DIR / "data" / "interim" / "disruption_master_10k_multiexpert_labelled.xlsx"
SHEET_NAME = "data"  # master sheet name
OUTPUT_DIR = "models/disruption_v2_experts"

USE_URL_FALLBACK = True
REQUIRE_TEXT = True

THRESHOLD_GENERAL = 0.40
THRESHOLD_EXPERT = 0.40

ROW_ORIGIN_COL = "row_origin"
GOLD_ORIGIN_VALUE = "gold_manual"

URL_COL = "url_normalized"
TITLE_COL = "title"
META_COL = "meta_description"

# Gold columns (NO prefix)
GOLD_TYPES = [
    "flood",
    "drought",
    "cyclone_huricane",
    "extreme_heat",
    "landslide",
    "earthquake",
    "mine_accident",
    "labour_strike",
    "protests",
    "trade_embargo",
    "country_relations",
    "tariffs",
]
GOLD_GENERAL_COL = "disruption"

# Weak labels (ChatGPT columns WITH prefix)
WEAK_PREFIX = "chatgpt_"  # as requested

BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!", "msn", "bot"
]


# ============================
# Utilities
# ============================
def to_int01(x) -> int:
    """Map mixed (0/1, '0'/'1', booleans, true/false) to {0,1}."""
    if pd.isna(x):
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f"}:
        return 0
    return 0


def looks_like_garbage(s: str) -> bool:
    if not isinstance(s, str):
        return True
    s = s.lower().strip()
    if len(s) < 15:
        return True
    return any(p in s for p in BAD_TEXT_PATTERNS)


def url_to_text(url: str) -> str:
    """Convert URL path into text tokens (slug -> words)."""
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        path = urlparse(url).path
    except Exception:
        return ""
    path = path.replace("/", " ")
    path = re.sub(r"[-_]+", " ", path)
    path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"\b\d+\b", " ", path)
    path = re.sub(r"\s+", " ", path).strip().lower()
    return path


def build_text(row: pd.Series) -> str:
    title = "" if pd.isna(row.get(TITLE_COL)) else str(row.get(TITLE_COL))
    desc = "" if pd.isna(row.get(META_COL)) else str(row.get(META_COL))
    url = "" if pd.isna(row.get(URL_COL)) else str(row.get(URL_COL))

    main = f"{title}. {desc}".strip()
    main = " ".join(main.split())

    if USE_URL_FALLBACK and looks_like_garbage(main):
        return url_to_text(url)
    return main


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


def format_seconds(secs: float) -> str:
    secs = max(0.0, float(secs))
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


# ============================
# Label construction (gold + chatgpt)
# ============================
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold rows (row_origin == gold_manual):
      - experts from gold columns (e.g., 'flood', ...)
      - general from gold 'disruption'
    Non-gold rows:
      - experts from chatgpt_* columns
      - general from chatgpt_disruption IF present, else OR over chatgpt experts
    """
    is_gold = (df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE).values

    # Ensure weak label columns exist for all types
    weak_cols = [f"{WEAK_PREFIX}{t}" for t in GOLD_TYPES]
    ensure_columns(df, weak_cols)

    # General weak label: prefer chatgpt_disruption if present
    weak_general_col = f"{WEAK_PREFIX}{GOLD_GENERAL_COL}"
    has_weak_general = weak_general_col in df.columns

    y = pd.DataFrame(index=df.index)

    # Expert targets
    for t in GOLD_TYPES:
        gold_col = t                # NO prefix
        weak_col = f"{WEAK_PREFIX}{t}"

        if gold_col not in df.columns:
            raise ValueError(f"Gold column missing: '{gold_col}'")
        if weak_col not in df.columns:
            raise ValueError(f"Weak label column missing: '{weak_col}'")

        gold_vals = df[gold_col].map(to_int01).values
        weak_vals = df[weak_col].map(to_int01).values
        y[t] = np.where(is_gold, gold_vals, weak_vals).astype(int)

    # General target
    if GOLD_GENERAL_COL not in df.columns:
        raise ValueError(f"Gold general column missing: '{GOLD_GENERAL_COL}'")

    gold_general = df[GOLD_GENERAL_COL].map(to_int01).values

    if has_weak_general:
        weak_general = df[weak_general_col].map(to_int01).values
    else:
        # fall back to OR over weak experts (from chatgpt_*)
        weak_general = (df[weak_cols].applymap(to_int01).sum(axis=1).values > 0).astype(int)

    y[GOLD_GENERAL_COL] = np.where(is_gold, gold_general, weak_general).astype(int)

    return y


# ============================
# Training util
# ============================
def train_one_binary(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_name: str,
    threshold: float,
    random_state: int = 42,
):
    """
    Train a calibrated linear SVM on precomputed embeddings.
    Writes:
      - error_review.xlsx (all_test / false_pos / false_neg) with original columns
      - model bundle joblib
    """
    if len(np.unique(y)) < 2:
        print(f"Skipping {model_name}: only one class present.")
        return

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    X_train_emb = embeddings[idx_train]
    X_test_emb = embeddings[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5)
    clf.fit(X_train_emb, y_train)

    probs = clf.predict_proba(X_test_emb)[:, 1]
    preds = (probs > threshold).astype(int)

    print(f"\n==== {model_name} ====")
    print("Average precision:", average_precision_score(y_test, probs))
    print(f"Classification report @ threshold={threshold}:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    out_dir.mkdir(parents=True, exist_ok=True)

    review_df = df.loc[idx_test].copy()
    review_df["p_disruption"] = probs
    review_df["y_true"] = y_test
    review_df["y_pred"] = preds
    review_df["error_type"] = ""
    review_df.loc[(review_df["y_true"] == 0) & (review_df["y_pred"] == 1), "error_type"] = "FALSE_POSITIVE"
    review_df.loc[(review_df["y_true"] == 1) & (review_df["y_pred"] == 0), "error_type"] = "FALSE_NEGATIVE"

    false_pos = review_df[review_df["error_type"] == "FALSE_POSITIVE"].sort_values("p_disruption", ascending=False)
    false_neg = review_df[review_df["error_type"] == "FALSE_NEGATIVE"].sort_values("p_disruption", ascending=True)

    review_path = out_dir / f"{model_name}_error_review.xlsx"
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        review_df.sort_values("p_disruption", ascending=False).to_excel(writer, index=False, sheet_name="all_test")
        false_pos.to_excel(writer, index=False, sheet_name="false_positives")
        false_neg.to_excel(writer, index=False, sheet_name="false_negatives")

    dump(
        {
            "embed_model": "all-MiniLM-L6-v2",
            "classifier": clf,
            "threshold": threshold,
            "label": model_name,
            "use_url_fallback": USE_URL_FALLBACK,
        },
        out_dir / f"{model_name}.joblib",
    )

    print(f"Wrote: {review_path}")
    print(f"Saved model: {out_dir / f'{model_name}.joblib'}")
    print("False positives:", len(false_pos), "| False negatives:", len(false_neg))


# ============================
# Main
# ============================
print("Loading Excel:", TRAINING_XLSX)
df = pd.read_excel(TRAINING_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
print("Rows loaded:", len(df))

# Core required columns
ensure_columns(df, [ROW_ORIGIN_COL, URL_COL, TITLE_COL, META_COL])
ensure_columns(df, GOLD_TYPES + [GOLD_GENERAL_COL])

# Build text
df["text"] = df.apply(build_text, axis=1)
if REQUIRE_TEXT:
    before = len(df)
    df = df[df["text"].astype(str).str.len() > 0].copy().reset_index(drop=True)
    print(f"Dropped empty-text rows: {before} -> {len(df)}")
else:
    df = df.reset_index(drop=True)

# Build targets (gold + chatgpt)
y_df = build_targets(df)
print("Gold rows:", int((df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE).sum()))
print("General label counts:", y_df[GOLD_GENERAL_COL].value_counts().to_dict())

# Embed once (shared across all models) + visible progress bar
print("Loading embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding all texts (once)...")
X_text = df["text"].astype(str).tolist()
embeddings = embedder.encode(X_text, normalize_embeddings=True, show_progress_bar=True)

# Train 13 models (1 general + 12 experts) with overall progress + ETA
root = Path(OUTPUT_DIR)
root.mkdir(parents=True, exist_ok=True)

tasks = [("general", GOLD_GENERAL_COL, THRESHOLD_GENERAL)] + [(f"expert_{t}", t, THRESHOLD_EXPERT) for t in GOLD_TYPES]

ema_per_model = None
alpha = 0.25  # EMA smoothing
start_all = time.time()

pbar = tqdm(tasks, desc="Training models", unit="model")
for i, (subdir, label_col, thr) in enumerate(pbar, start=1):
    t0 = time.time()

    model_name = "disruption_general" if label_col == GOLD_GENERAL_COL else f"disruption_{label_col}"
    out_dir = root / subdir

    y = y_df[label_col].values.astype(int)
    train_one_binary(df=df, embeddings=embeddings, y=y, out_dir=out_dir, model_name=model_name, threshold=thr)

    dt = time.time() - t0
    ema_per_model = dt if ema_per_model is None else (alpha * dt + (1 - alpha) * ema_per_model)

    remaining = (len(tasks) - i) * (ema_per_model if ema_per_model is not None else 0.0)
    elapsed = time.time() - start_all

    pbar.set_postfix_str(f"last={format_seconds(dt)} | elapsed={format_seconds(elapsed)} | ETA={format_seconds(remaining)}")

pbar.close()

print("\nAll done.")
print("Models saved under:", root)
