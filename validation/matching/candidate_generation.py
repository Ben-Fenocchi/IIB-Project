"""
Candidate generation for validation matching.

This file generates plausible extracted–reference event pairs
to avoid expensive all-to-all matching.

Flowchart role:
- 'Candidate Generation' stage joining extracted and reference streams

Only coarse filters are applied here (type and time overlap).
No scoring or decisions are made in this module.
"""

from datetime import timedelta, date
from typing import Dict, List

from validation.models import CanonicalEvent


# ------------------ CONFIG ------------------ #

DEFAULT_MAX_DAYS_APART = 7
MAX_CANDIDATES_PER_EVENT = 200


# ------------------ HELPERS ------------------ #

def _date_overlap(
    d1_start: date | None,
    d1_end: date | None,
    d2_start: date | None,
    d2_end: date | None,
    max_days: int,
) -> bool:
    """
    Check whether two date ranges overlap within a tolerance window.
    """
    if not d1_start or not d2_start:
        return False

    d1_end = d1_end or d1_start
    d2_end = d2_end or d2_start

    return (
        d1_start <= d2_end + timedelta(days=max_days)
        and d1_end >= d2_start - timedelta(days=max_days)
    )


# ------------------ CANDIDATE GENERATION ------------------ #

def generate_candidates(
    extracted: List[CanonicalEvent],
    references: List[CanonicalEvent],
    max_days_apart: int = DEFAULT_MAX_DAYS_APART,
) -> Dict[str, List[str]]:
    """
    Generate candidate reference IDs for each extracted event.
    """
    # Index reference events by type
    refs_by_type: Dict[str, List[CanonicalEvent]] = {}
    for r in references:
        refs_by_type.setdefault(r.kind, []).append(r)

    candidates: Dict[str, List[str]] = {}

    for e in extracted:
        pool = refs_by_type.get(e.kind, [])
        matched: List[str] = []

        for r in pool:
            if _date_overlap(
                e.date_start,
                e.date_end,
                r.date_start,
                r.date_end,
                max_days_apart,
            ):
                matched.append(r.id)

                if len(matched) >= MAX_CANDIDATES_PER_EVENT:
                    break

        candidates[e.id] = matched

    return candidates
