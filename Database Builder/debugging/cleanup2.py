'''
This script will append to the first cleanup script the top expert score for that URL to the database so we can see
how the expert model classifier compares to the LLM at classing the disruption types - to see if we should rely on this rather
than the LLM

'''

import pandas as pd
from pathlib import Path


# ------------------ CONFIG ------------------ #

BASE_DIR = Path(__file__).resolve().parent

EXTRACTION_FILE = BASE_DIR / "cleanup.csv"
EXPERT_FILE     = BASE_DIR / "expert_database.csv"

OUTPUT_FILE     = BASE_DIR / "merged_database.csv"


# ------------------ LOAD ------------------ #

df_extraction = pd.read_csv(EXTRACTION_FILE)
df_expert     = pd.read_csv(EXPERT_FILE)


# ------------------ STANDARDISE URL COLUMN ------------------ #
# Ensure both have column named "url"

if "url" not in df_extraction.columns:
    raise ValueError("Extraction DB missing 'url' column")

if "url" not in df_expert.columns:
    raise ValueError("Expert DB missing 'url' column")


# ------------------ MERGE (INNER JOIN) ------------------ #

merged = pd.merge(
    df_extraction,
    df_expert,
    on="url",
    how="inner"
)


# ------------------ SAVE ------------------ #

merged.to_csv(OUTPUT_FILE, index=False)

print(f"Merged rows: {len(merged)}")
print(f"Saved to: {OUTPUT_FILE}")