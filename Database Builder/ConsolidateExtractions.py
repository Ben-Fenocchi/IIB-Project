#!/usr/bin/env python3
"""
Event-level deduplication for extracted disruption events.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


# ------------------ CONFIG ------------------ #

EVENT_EVENT_TOLERANCE_DAYS = 1
EVENT_PUBLISH_TOLERANCE_DAYS = 2
PUBLISH_PUBLISH_TOLERANCE_DAYS = 3

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_JSONL = RESULTS_DIR / "extractions.jsonl"
INPUT_CSV = RESULTS_DIR / "extractions.csv"

OUTPUT_JSONL = RESULTS_DIR / "consolidatedExtractions.jsonl"
OUTPUT_CSV = RESULTS_DIR / "consolidatedExtractions.csv"


# ------------------ LOAD ------------------ #

def load_extractions() -> pd.DataFrame:
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

    # ---- DATE PARSING (trust upstream normalisation) ----
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce", utc=True).dt.tz_convert(None)

    df["disruption_type"] = (
        df["disruption_type"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# ------------------ HELPERS ------------------ #

def location_tokens(location: str) -> set[str]:
    if not location:
        return set()

    location = location.lower()
    location = re.sub(r"\(.*?\)", "", location)
    location = re.sub(r"[^a-z\s]", " ", location)

    return {t for t in location.split() if len(t) > 2}


def choose_match_date(record: Dict[str, Any]) -> Optional[Tuple[pd.Timestamp, str]]:
    ed = record.get("event_date")
    if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
        return ed, "event"

    pd_ = record.get("publish_date")
    if isinstance(pd_, pd.Timestamp) and not pd.isna(pd_):
        return pd_, "publish"

    return None


def dates_close_asymmetric(
    d1: pd.Timestamp, src1: str,
    d2: pd.Timestamp, src2: str
) -> bool:
    delta_days = abs((d1 - d2).days)

    if src1 == "event" and src2 == "event":
        tol = EVENT_EVENT_TOLERANCE_DAYS
    elif src1 != src2:
        tol = EVENT_PUBLISH_TOLERANCE_DAYS
    else:
        tol = PUBLISH_PUBLISH_TOLERANCE_DAYS

    return delta_days <= tol


# ------------------ MERGING ------------------ #

def merge_cluster(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    merged["disruption_type"] = cluster[0]["disruption_type"]

    event_dates = [r["event_date"] for r in cluster if pd.notna(r["event_date"])]
    merged["event_date"] = min(event_dates) if event_dates else None

    publish_dates = [r["publish_date"] for r in cluster if pd.notna(r["publish_date"])]
    merged["publish_date"] = min(publish_dates) if publish_dates else None

    merged["location_name"] = max(
        (r.get("location_name", "") for r in cluster if r.get("location_name")),
        key=len,
        default=""
    )

    merged["urls"] = sorted({r.get("url") for r in cluster if r.get("url")})
    merged["num_articles"] = len(cluster)

    merged["source_title"] = max(
        (r.get("source_title", "") for r in cluster if r.get("source_title")),
        key=len,
        default=""
    )

    merged["duration_hours"] = next(
        (r.get("duration_hours") for r in cluster if r.get("duration_hours") is not None),
        None
    )

    extras: Dict[str, List[Any]] = {}
    for r in cluster:
        if isinstance(r.get("extras"), dict):
            for k, v in r["extras"].items():
                if v is not None:
                    extras.setdefault(k, []).append(v)

    merged["extras"] = {k: vals[0] if len(vals) == 1 else vals for k, vals in extras.items()}

    merged["evidence"] = [
        e for r in cluster if isinstance(r.get("evidence"), list) for e in r["evidence"]
    ]

    merged["confidence"] = max((r.get("confidence", 0.0) for r in cluster), default=0.0)
    merged["method"] = sorted({r.get("method") for r in cluster if r.get("method")})

    return merged


# ------------------ DEDUPLICATION ------------------ #

def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict(orient="records")

    clusters: List[List[Dict[str, Any]]] = []
    passthrough: List[Dict[str, Any]] = []

    for record in records:
        if record["disruption_type"] == "unknown":
            passthrough.append(record)
            continue

        rec_tokens = location_tokens(record.get("location_name", ""))
        rec_match = choose_match_date(record)

        matched = False

        for cluster in clusters:
            rep = cluster[0]

            if record["disruption_type"] != rep["disruption_type"]:
                continue

            rep_match = choose_match_date(rep)
            if rec_match is None or rep_match is None:
                continue

            rec_date, rec_src = rec_match
            rep_date, rep_src = rep_match

            if not dates_close_asymmetric(rec_date, rec_src, rep_date, rep_src):
                continue

            if rec_tokens & location_tokens(rep.get("location_name", "")):
                cluster.append(record)
                matched = True
                break

        if not matched:
            clusters.append([record])

    merged_events = [merge_cluster(c) for c in clusters]
    return pd.DataFrame(merged_events + passthrough)


# ------------------ SAVE ------------------ #

def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            for k in ("event_date", "publish_date"):
                if isinstance(record.get(k), pd.Timestamp):
                    record[k] = record[k].isoformat()
                elif record.get(k) is None or pd.isna(record.get(k)):
                    record[k] = None
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------ MAIN ------------------ #

def main():
    df_before = load_extractions()
    df_after = dedupe_events(df_before)

    save_outputs(df_after)
    print("Saved consolidatedExtractions.csv and consolidatedExtractions.jsonl")


if __name__ == "__main__":
    main()
