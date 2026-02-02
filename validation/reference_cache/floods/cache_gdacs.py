"""
Cache flood events from the GDACS API.

Fetches flood alerts for a caller-specified date window and writes a
cleaned JSON cache file.

Date window selection is handled upstream.
"""

from datetime import date
from pathlib import Path
import json
import requests


API_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist"


def cache_gdacs_floods(
    start_date: date,
    end_date: date,
    output_path: Path,
) -> None:
    params = {
        "eventtype": "FL",
        "fromdate": start_date.isoformat(),
        "todate": end_date.isoformat(),
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()

    data = response.json().get("features", [])

    cleaned = []
    for f in data:
        props = f.get("properties", {})
        geom = f.get("geometry", {})

        cleaned.append(
            {
                "id": props.get("eventid"),
                "name": props.get("eventname"),
                "fromdate": props.get("fromdate"),
                "todate": props.get("todate"),
                "country": props.get("country"),
                "lat": geom.get("coordinates", [None, None])[1],
                "lon": geom.get("coordinates", [None, None])[0],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
