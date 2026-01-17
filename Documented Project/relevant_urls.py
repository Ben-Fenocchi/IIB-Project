import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from joblib import load
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
# We'll save the "Gold" output here
GOLD_BASE_DIR = Path("data/processed/model_scored_daily")
# We get our input from the cleaned "Step 4" state
STATE_DIR = Path("data/interim/_state")

BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!"
]

# -----------------------------
# HELPERS (Unchanged)
# -----------------------------
def looks_like_garbage(s: str) -> bool:
    if not isinstance(s, str): return True
    s = s.lower().strip()
    if len(s) < 15: return True
    return any(p in s for p in BAD_TEXT_PATTERNS)

def url_to_text(url: str) -> str:
    if not isinstance(url, str) or not url.strip(): return ""
    try:
        path = urlparse(url).path
        path = path.replace("/", " ")
        path = re.sub(r"[-_]+", " ", path)
        path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path, flags=re.IGNORECASE)
        path = re.sub(r"\b\d+\b", " ", path)
        return re.sub(r"\s+", " ", path).strip().lower()
    except: return ""

def build_text(row: pd.Series, use_url_fallback: bool = True) -> str:
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    desc = str(row.get("meta_description", "")) if pd.notna(row.get("meta_description")) else ""
    main = " ".join(f"{title}. {desc}".split())
    if use_url_fallback and looks_like_garbage(main):
        return url_to_text(str(row.get("url_normalized", "")))
    return main

# -----------------------------
# MAIN SCORING LOGIC
# -----------------------------
def main(target_date: str, threshold_override: float = None, top_k: int = 0):
    # 1. Setup Nested Path: Year/Month/Day
    year, month, day = target_date[:4], target_date[4:6], target_date[6:8]
    out_dir = GOLD_BASE_DIR / year / month / day
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Find Input (Fixed Cache from Step 4)
    in_csv = STATE_DIR / f"url_title_meta_cache_{target_date}_fixed.csv"
    model_path = Path("models/disruption_v1/disruption_model.joblib")

    if not in_csv.exists():
        print(f"Skipping Step 5: Cleaned cache {in_csv.name} not found.")
        return

    # 3. Load Model and Data
    print(f"\n--- Step 5: Scoring Relevance for {target_date} ---")
    bundle = load(model_path)
    clf = bundle["classifier"]
    threshold = threshold_override if threshold_override is not None else float(bundle.get("threshold", 0.5))
    use_url_fallback = bool(bundle.get("use_url_fallback", True))

    df = pd.read_csv(in_csv, encoding="utf-8", engine="python")
    
    # 4. Process
    df["text"] = df.apply(lambda r: build_text(r, use_url_fallback=use_url_fallback), axis=1)
    embedder = SentenceTransformer(bundle.get("embed_model", "all-MiniLM-L6-v2"))

    print(f"Embedding {len(df)} rows...")
    X = embedder.encode(df["text"].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)
    
    # 5. Scoring
    probs = clf.predict_proba(X)[:, 1]
    df["p_disruption"] = probs
    df["keep"] = df["p_disruption"] >= threshold

    # 6. Save results to the Nested Directory
    kept = df[df["keep"]].sort_values("p_disruption", ascending=False)
    if top_k > 0: kept = kept.head(top_k)

    scored_path = out_dir / f"{target_date}_scored.csv"
    urls_path_txt = out_dir / f"{target_date}_interesting_urls.txt"
    urls_path_csv = out_dir / f"{target_date}_interesting_urls_only.csv"

    df.to_csv(scored_path, index=False)
    kept["url_normalized"].to_csv(urls_path_txt, index=False, header=False)
    kept[["url_normalized"]].to_csv(urls_path_csv, index=False)

    print(f"Success! Folder created: {out_dir}")
    print(f"Total Scored: {len(df)} | Kept: {len(kept)}")
    print(f"Final CSV: {urls_path_csv.name}\n")

if __name__ == "__main__":
    day = input("Enter date to score (YYYYMMDD): ").strip()
    main(day)