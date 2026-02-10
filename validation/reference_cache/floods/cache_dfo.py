"""
Cache flood events from the Dartmouth Flood Observatory (DFO) dataset.

Reads DFO flood records, filters to a caller-specified date window,
and writes a cleaned JSON cache file.

Assumes local access to the DFO dataset (CSV or JSON).
"""

from datetime import date
from pathlib import Path
import csv
import json


def cache_dfo_floods(
    start_date: date,
    end_date: date,
    source_csv: Path,
    output_path: Path,
) -> None:
    cleaned = []

    with source_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row.get("Began")
            end = row.get("Ended")

            if not start:
                continue

            # Simple string-date comparison (ISO-like in DFO)
            if start > end_date.isoformat() or start < start_date.isoformat():
                continue

            cleaned.append(
                {
                    "id": row.get("ID"),
                    "start_date": start,
                    "end_date": end,
                    "location": row.get("Location"),
                    "country": row.get("Country"),
                    "lat": row.get("Lat"),
                    "lon": row.get("Lon"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
