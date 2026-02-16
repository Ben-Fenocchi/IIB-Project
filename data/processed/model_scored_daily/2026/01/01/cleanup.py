#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


# ------------------ PATH SETUP ------------------ #

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "20260101_experts_scored.csv"
OUTPUT_FILE = BASE_DIR / "cleaned.csv"


# ------------------ LOAD ------------------ #

df = pd.read_csv(INPUT_FILE)


# ------------------ FILTER ------------------ #

filtered = df[
    df["top_expert"].notna() &
    (df["top_expert"].astype(str).str.strip() != "")
]


# ------------------ SELECT COLUMNS ------------------ #

result = filtered[
    [
        "url_normalized",
        "title",
        "meta_description",
        "top_expert",
        "top_expert_p"
    ]
]


# ------------------ SAVE ------------------ #

result.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(result)} rows to {OUTPUT_FILE}")