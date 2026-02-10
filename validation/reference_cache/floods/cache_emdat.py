"""
Cache flood events from the EM-DAT dataset.

Filters EM-DAT flood records to a caller-specified date window and writes
a cleaned JSON cache file.

Assumes local access to EM-DAT data (CSV).
"""

from datetime import date
from pathlib import Path
import csv
import json


def cache_emdat_floods(
    start_date: date,
    end_date: date,
    source_csv: Path,
    output_path: Path,
) -> None:
    cleaned = []

    with source_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Disaster Type") != "Flood":
                continue

            start = row.get("Start Date")
            end = row.get("End Date")

            if not start:
                continue

            if start > end_date.isoformat() or start < start_date.isoformat():
                continue

            cleaned.append(
                {
                    "disaster_no": row.get("Dis No"),
                    "start_date": start,
                    "end_date": end,
                    "country": row.get("Country"),
                    "location": row.get("Location"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
