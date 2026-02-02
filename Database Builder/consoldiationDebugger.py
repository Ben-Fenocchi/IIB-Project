from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


# ------------------ PATHS ------------------ #

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DEDUPED_CSV = RESULTS_DIR / "dedupedExtractions.csv"
DEDUPED_JSONL = RESULTS_DIR / "dedupedExtractions.jsonl"


# ------------------ LOAD ------------------ #

def load_deduped() -> pd.DataFrame:
    """
    Load deduped extractions from CSV or JSONL.
    """
    if DEDUPED_CSV.exists():
        df = pd.read_csv(DEDUPED_CSV)

    elif DEDUPED_JSONL.exists():
        records = []
        with open(DEDUPED_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)

    else:
        raise FileNotFoundError("No dedupedExtractions.csv or .jsonl found")

    df = df.fillna("")

    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")

    return df


# ------------------ DISPLAY ------------------ #

def inspect_events(
    disruption_type: str,
    max_rows: int = 30,
    title_words: int = 10,
    location_words: int = 10,
):
    """
    Display a neat table of deduplicated events for a given disruption type.

    Date formatting:
    - YYYY-MM-DD  -> extracted event date (from article text)
    - YYYY\MM\DD  -> publication date (metadata proxy, slash)
    """
    df = load_deduped()

    disruption_type = disruption_type.lower().strip()
    df = df[df["disruption_type"] == disruption_type]

    if df.empty:
        print(f"No events found for disruption type: {disruption_type}")
        return

    def shorten(text: str, n: int) -> str:
        words = text.split()
        if len(words) <= n:
            return text
        return " ".join(words[:n]) + " ..."

    def format_date(event_date, publish_date) -> str:
        """
        Prefer event_date; fall back to publish_date.
        """
        if pd.notna(event_date):
            return event_date.strftime("%Y-%m-%d")  # hyphen
        if pd.notna(publish_date):
            return publish_date.strftime("%Y/%m/%d")  # slash
        return ""

    out = pd.DataFrame({
        "title": df["source_title"].apply(lambda s: shorten(s, title_words)),
        "date": [
            format_date(ed, pd_)
            for ed, pd_ in zip(df["event_date"], df["publish_date"])
        ],
        "location": df["location_name"].apply(lambda s: shorten(s, location_words)),
    })

    # ------------------ PRINT ------------------ #

    print(
        f"\n=== {disruption_type.upper()} ({len(df)} events) ===\n"
        "Date notation:\n"
        "  YYYY-MM-DD  = event date extracted from article text\n"
        "  YYYY\MM\DD  = article publication date (metadata proxy)\n"
    )

    print(out.head(max_rows).to_string(index=False))

    if len(df) > max_rows:
        print(f"\n... ({len(df) - max_rows} more rows not shown) ...")


# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    inspect_events("earthquake", max_rows=40)
