"""
Cache flood events from the ReliefWeb API.

This script retrieves flood-related disasters from ReliefWeb for a
caller-specified time window and writes a cleaned JSON cache file.

Date window selection is handled upstream (e.g. in run_validation).
This script assumes start_date and end_date are already final.

Flowchart role:
- Reference acquisition (pre-validation)
"""

from datetime import date
from pathlib import Path
import json
import requests


API_URL = "https://api.reliefweb.int/v2/disasters"


def cache_reliefweb_floods(
    start_date: date,
    end_date: date,
    output_path: Path,
    appname: str,
) -> None:
    """
    Fetch and cache flood disasters from ReliefWeb for a given date window.
    """

    params = {
        "appname": appname,
        "limit": 1000,
        "query": {
            "operator": "AND",
            "value": [
                {
                    "field": "type.name",
                    "value": "Flood",
                },
                {
                    "field": "date.created",
                    "value": {
                        "from": start_date.isoformat(),
                        "to": end_date.isoformat(),
                    },
                },
            ],
        },
        "fields": {
            "include": [
                "id",
                "name",
                "date",
                "country",
                "type",
            ]
        },
    }

    response = requests.post(API_URL, json=params)
    response.raise_for_status()

    data = response.json().get("data", [])

    cleaned = []
    for r in data:
        cleaned.append(
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "date": r.get("date", {}).get("created"),
                "country": (
                    r.get("country", [{}])[0].get("name")
                    if r.get("country")
                    else None
                ),
                "type": r.get("type", {}).get("name"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cleaned, indent=2),
        encoding="utf-8",
    )
