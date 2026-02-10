"""
Scoring functions for candidate extracted–reference matches.

This file assigns similarity scores to candidate pairs produced by
candidate generation.

Flowchart role:
- 'Match Scoring' stage

This module does not decide matches; it only computes scores.
"""

from typing import Dict
from datetime import date

from validation.models import CanonicalEvent, CandidateMatch


# ------------------ SCORING HELPERS ------------------ #

def _time_score(e: CanonicalEvent, r: CanonicalEvent) -> float:
    """
    Score temporal proximity in [0, 1].
    """
    if not e.date_start or not r.date_start:
        return 0.0

    e_d = e.date_start
    r_start = r.date_start
    r_end = r.date_end or r.date_start

    if r_start <= e_d <= r_end:
        return 1.0

    delta = min(abs((e_d - r_start).days), abs((e_d - r_end).days))
    return 1.0 / (1.0 + delta)


def _location_text_score(e: CanonicalEvent, r: CanonicalEvent) -> float:
    """
    Weak location agreement using string overlap.
    """
    if not e.location_name or not r.location_name:
        return 0.0

    e_loc = e.location_name.lower()
    r_loc = r.location_name.lower()

    if e_loc in r_loc or r_loc in e_loc:
        return 1.0

    return 0.0


def _text_score(e: CanonicalEvent, r: CanonicalEvent) -> float:
    """
    Simple token overlap score.
    """
    e_tokens = set((e.text or "").lower().split())
    r_tokens = set((r.text or "").lower().split())

    if not e_tokens or not r_tokens:
        return 0.0

    return len(e_tokens & r_tokens) / len(e_tokens | r_tokens)



def score_candidate(
    extracted: CanonicalEvent,
    reference: CanonicalEvent,
) -> CandidateMatch:
    """
    Compute similarity features and combined score for one candidate pair.
    """
    t = _time_score(extracted, reference)
    l = _location_text_score(extracted, reference)
    x = _text_score(extracted, reference)

    score = 0.5 * t + 0.3 * l + 0.2 * x

    return CandidateMatch(
        extracted_id=extracted.id,
        ref_id=reference.id,
        dataset=reference.meta.get("dataset", "unknown"),
        features={
            "time": t,
            "location": l,
            "text": x,
        },
        score=score,
    )
