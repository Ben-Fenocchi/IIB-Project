#!/usr/bin/env python3
"""
Event-level deduplication for extracted disruption events.

This script collapses multiple news articles referring to the same
underlying disruption event into a single canonical record.

Matching logic (conservative by design):
- disruption_type must match and must not be "unknown"
- temporal proximity:
    * prefer event_date (LLM-extracted)
    * fall back to publish_date (article metadata)
- dates must be within +/- DATE_TOLERANCE_DAYS
- location_name strings must share at least one geographic token

Merging rules:
- URLs are preserved as a list
- num_articles counts contributing articles
- indicator values (extras) are merged LOSSLESSLY:
    * single value -> scalar
    * multiple distinct values -> list
- no overwriting of conflicting information

The script also prints detailed statistics showing the impact of
deduplication (before vs after counts).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


# ------------------ CONFIG ------------------ #

DATE_TOLERANCE_DAYS = 1  # +/- days allowed when matching events

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_JSONL = RESULTS_DIR / "extractions.jsonl"
INPUT_CSV = RESULTS_DIR / "extractions.csv"

OUTPUT_JSONL = RESULTS_DIR / "dedupedExtractions.jsonl"
OUTPUT_CSV = RESULTS_DIR / "dedupedExtractions.csv"


# ------------------ LOAD ------------------ #

def load_extractions() -> pd.DataFrame:
    """
    Load extraction outputs from JSONL or CSV and normalise fields.
    """
    if INPUT_JSONL.exists():
        records = []
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)

    elif INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)

    else:
        raise FileNotFoundError("No extractions.jsonl or extractions.csv found")

    df = df.fillna("")

    # Parse dates safely (NaT if missing / malformed)
    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")

    # Normalise disruption type
    df["disruption_type"] = df["disruption_type"].str.lower().str.strip()

    return df


# ------------------ HELPERS ------------------ #

def location_tokens(location: str) -> set[str]:
    """
    Convert a raw location string into a set of meaningful tokens.

    Example:
        "Sumatra (Aceh), Indonesia"
        -> {"sumatra", "aceh", "indonesia"}
    """
    if not location:
        return set()

    location = location.lower()
    location = re.sub(r"\(.*?\)", "", location)
    location = re.sub(r"[^a-z\s]", " ", location)

    return {t for t in location.split() if len(t) > 2}


def dates_close(d1: pd.Timestamp, d2: pd.Timestamp) -> bool:
    """
    Check whether two dates are within the allowed tolerance window.
    """
    if pd.isna(d1) or pd.isna(d2):
        return False

    return abs((d1 - d2).days) <= DATE_TOLERANCE_DAYS


def choose_match_date(record: Dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Select the strongest available temporal signal for matching.

    Priority:
      1) event_date (explicit event timing)
      2) publish_date (article metadata proxy)
    """
    if not pd.isna(record.get("event_date")):
        return record["event_date"]

    if not pd.isna(record.get("publish_date")):
        return record["publish_date"]

    return None


# ------------------ MERGING ------------------ #

def merge_cluster(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a list of extraction records referring to the same event.
    """
    merged: Dict[str, Any] = {}

    merged["disruption_type"] = cluster[0]["disruption_type"]

    # Preserve explicit event_date if any exist
    event_dates = [r["event_date"] for r in cluster if not pd.isna(r["event_date"])]
    merged["event_date"] = min(event_dates) if event_dates else None

    # Preserve earliest publish date (useful provenance)
    publish_dates = [r["publish_date"] for r in cluster if not pd.isna(r["publish_date"])]
    merged["publish_date"] = min(publish_dates) if publish_dates else None

    # Choose the most specific (longest) location string
    merged["location_name"] = max(
        (r["location_name"] for r in cluster if r["location_name"]),
        key=len,
        default=""
    )

    # Provenance
    merged["urls"] = sorted({r["url"] for r in cluster if r["url"]})
    merged["num_articles"] = len(cluster)

    merged["source_title"] = max(
        (r["source_title"] for r in cluster if r["source_title"]),
        key=len,
        default=""
    )

    # Duration: preserve first non-null (can be upgraded later if needed)
    merged["duration_hours"] = next(
        (r["duration_hours"] for r in cluster if r["duration_hours"] not in ("", None)),
        ""
    )

    # -------- LOSSLESS MERGE OF INDICATORS (extras) -------- #
    extras: Dict[str, List[Any]] = {}

    for r in cluster:
        if not isinstance(r["extras"], dict):
            continue

        for k, v in r["extras"].items():
            if v in ("", None):
                continue

            extras.setdefault(k, [])

            if v not in extras[k]:
                extras[k].append(v)

    # Collapse singletons back to scalars
    merged["extras"] = {
        k: (vals[0] if len(vals) == 1 else vals)
        for k, vals in extras.items()
    }

    # Combine evidence strings
    merged["evidence"] = [
        e for r in cluster if isinstance(r["evidence"], list) for e in r["evidence"]
    ]

    merged["confidence"] = max((r["confidence"] for r in cluster), default=0.0)
    merged["method"] = sorted({r["method"] for r in cluster if r["method"]})

    return merged


# ------------------ DEDUPLICATION ------------------ #

def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy clustering using:
    - disruption type
    - hierarchical temporal matching
    - token-based location overlap
    """
    records = df.to_dict(orient="records")

    clusters: List[List[Dict[str, Any]]] = []
    passthrough: List[Dict[str, Any]] = []

    for record in records:
        # Never merge unknown disruption types
        if record["disruption_type"] == "unknown":
            passthrough.append(record)
            continue

        rec_tokens = location_tokens(record["location_name"])
        rec_date = choose_match_date(record)

        matched = False

        for cluster in clusters:
            rep = cluster[0]

            if record["disruption_type"] != rep["disruption_type"]:
                continue

            rep_date = choose_match_date(rep)

            if rec_date is None or rep_date is None:
                continue

            if not dates_close(rec_date, rep_date):
                continue

            if rec_tokens & location_tokens(rep["location_name"]):
                cluster.append(record)
                matched = True
                break

        if not matched:
            clusters.append([record])

    merged_events = [merge_cluster(c) for c in clusters]
    final_records = merged_events + passthrough

    return pd.DataFrame(final_records)


# ------------------ STATISTICS ------------------ #

def print_dedupe_stats(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """
    Print summary statistics showing the impact of deduplication.
    """
    print("\n================ DEDUPLICATION SUMMARY ================\n")

    n_before = len(df_before)
    n_after = len(df_after)

    print(f"Total records before dedupe : {n_before}")
    print(f"Total records after dedupe  : {n_after}")

    if n_before > 0:
        reduction = 100.0 * (n_before - n_after) / n_before
        print(f"Overall reduction           : {reduction:.1f}%")

    print("\n---- By disruption type ----")

    before_counts = df_before["disruption_type"].value_counts().sort_index()
    after_counts = df_after["disruption_type"].value_counts().sort_index()

    all_types = sorted(set(before_counts.index) | set(after_counts.index))

    for dtype in all_types:
        if dtype == "unknown":
            continue

        b = before_counts.get(dtype, 0)
        a = after_counts.get(dtype, 0)

        if b == 0:
            continue

        abs_red = b - a
        pct_red = 100.0 * abs_red / b if b > 0 else 0.0

        print(
            f"{dtype:20s} "
            f"{b:5d} → {a:5d}  "
            f"({abs_red:4d} removed, {pct_red:5.1f}%)"
        )

    if "num_articles" in df_after.columns:
        merged = df_after[df_after["num_articles"] > 1]

        if not merged.empty:
            print("\n---- Merge intensity ----")
            print(f"Mean articles per event   : {merged['num_articles'].mean():.2f}")
            print(f"Median articles per event : {merged['num_articles'].median():.1f}")
            print(f"Max articles per event    : {merged['num_articles'].max()}")

    print("\n=======================================================\n")


# ------------------ SAVE ------------------ #

def save_outputs(df: pd.DataFrame):
    """
    Save deduplicated outputs to CSV and JSONL.
    """
    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()

            for k in ("event_date", "publish_date"):
                if isinstance(record.get(k), pd.Timestamp):
                    record[k] = record[k].isoformat()
                elif pd.isna(record.get(k)):
                    record[k] = None

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------ MAIN ------------------ #

def main():
    df_before = load_extractions()
    df_after = dedupe_events(df_before)

    save_outputs(df_after)
    print_dedupe_stats(df_before, df_after)

    print("Saved dedupedExtractions.csv and dedupedExtractions.jsonl")


if __name__ == "__main__":
    main()
