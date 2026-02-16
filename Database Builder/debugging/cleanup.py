'''
This script will extract the url title and disruption type columns from the LLM extracted records (pre consolidation)
'''

import json
import pandas as pd
from pathlib import Path


# ------------------ CONFIG ------------------ #

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "weekly_extractions_202601.jsonl"      # change filename
OUTPUT_FILE = BASE_DIR / "cleanup.csv"


# ------------------ LOAD ------------------ #

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            records.append({
                "url": obj.get("url", ""),
                "title": obj.get("source_title", ""),
                "disruption_type": obj.get("disruption_type", "")
            })


df = pd.DataFrame(records)


# ------------------ SAVE ------------------ #

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_FILE}")