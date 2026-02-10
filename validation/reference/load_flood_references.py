"""
Flood reference dataset loaders.

This file loads flood events from previously smoke-tested reference datasets:
- DFO
- GDACS
- EM-DAT
- ReliefWeb

It assumes reference data has already been downloaded and cached.
No API calls are made here.

Flowchart role:
- 'Reference Loading' stage on the reference-data stream
"""

from pathlib import Path
from typing import List
from datetime import datetime, date
import json

from validation.models import RefEvent


# ------------------ HELPERS ------------------ #

def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:19]).date()
    except Exception:
        return None


# ------------------ DFO ------------------ #

def load_dfo(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"DFO_{row.get('id')}",
                dataset="DFO",
                ref_type="flood",
                date_start=_parse_date(row.get("start_date")),
                date_end=_parse_date(row.get("end_date")),
                location_name=row.get("location"),
                country=row.get("country"),
                lat=row.get("lat"),
                lon=row.get("lon"),
                text=row.get("description"),
                raw=row,
            )
        )
    return out


# ------------------ GDACS ------------------ #

def load_gdacs(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"GDACS_{row.get('id')}",
                dataset="GDACS",
                ref_type="flood",
                date_start=_parse_date(row.get("fromdate")),
                date_end=_parse_date(row.get("todate")),
                location_name=row.get("country"),
                country=row.get("country"),
                lat=row.get("lat"),
                lon=row.get("lon"),
                text=row.get("name"),
                raw=row,
            )
        )
    return out


# ------------------ EM-DAT ------------------ #

def load_emdat(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"EMDAT_{row.get('disaster_no')}",
                dataset="EM-DAT",
                ref_type="flood",
                date_start=_parse_date(row.get("start_date")),
                date_end=_parse_date(row.get("end_date")),
                location_name=row.get("location"),
                country=row.get("country"),
                lat=None,
                lon=None,
                text=row.get("event_name"),
                raw=row,
            )
        )
    return out


# ------------------ RELIEFWEB ------------------ #

def load_reliefweb(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"RELIEFWEB_{row.get('id')}",
                dataset="ReliefWeb",
                ref_type="flood",
                date_start=_parse_date(row.get("date")),
                date_end=None,
                location_name=row.get("country"),
                country=row.get("country"),
                lat=None,
                lon=None,
                text=row.get("name"),
                raw=row,
            )
        )
    return out


# ------------------ AGGREGATOR ------------------ #

def load_all_flood_references(
    dfo_path: Path,
    gdacs_path: Path,
    emdat_path: Path,
    reliefweb_path: Path,
) -> List[RefEvent]:
    """
    Load and concatenate all flood reference datasets.
    """
    refs: List[RefEvent] = []
    refs.extend(load_dfo(dfo_path))
    refs.extend(load_gdacs(gdacs_path))
    refs.extend(load_emdat(emdat_path))
    refs.extend(load_reliefweb(reliefweb_path))
    return refs
