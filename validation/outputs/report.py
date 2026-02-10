"""
Reporting utilities for validation results.

This file aggregates match decisions from dual-gate verification into
summary statistics and diagnostic tables.

Flowchart role:
- 'Validation Outputs' stage

This module:
- Computes high-level metrics (forward/inverse match rates)
- Produces row-oriented outputs for inspection and analysis
- Does not write files or rerun validation logic
"""

from typing import Dict, List
from collections import Counter

from validation.models import MatchDecision


# ------------------ SUMMARY METRICS ------------------ #

def build_summary(
    forward: List[MatchDecision],
    inverse: List[MatchDecision],
) -> Dict[str, float]:
    """
    Compute high-level validation metrics.
    """
    f_total = len(forward)
    i_total = len(inverse)

    f_pass = sum(1 for d in forward if d.passed)
    i_pass = sum(1 for d in inverse if d.passed)

    return {
        "forward_total": f_total,
        "forward_matched": f_pass,
        "forward_match_rate": f_pass / f_total if f_total else 0.0,
        "inverse_total": i_total,
        "inverse_matched": i_pass,
        "inverse_match_rate": i_pass / i_total if i_total else 0.0,
    }


# ------------------ ROW-LEVEL OUTPUTS ------------------ #

def decisions_to_rows(decisions: List[MatchDecision]) -> List[dict]:
    """
    Convert MatchDecision objects into flat dictionaries
    suitable for CSV/JSON output.
    """
    rows = []
    for d in decisions:
        rows.append(
            {
                "source_id": d.source_id,
                "matched_id": d.matched_id,
                "matched_dataset": d.matched_dataset,
                "score": d.score,
                "passed": d.passed,
                "reason": d.reason,
            }
        )
    return rows


# ------------------ DIAGNOSTICS ------------------ #

def failure_reasons(decisions: List[MatchDecision]) -> Dict[str, int]:
    """
    Count failure reasons for unmatched events.
    """
    reasons = Counter()
    for d in decisions:
        if not d.passed:
            reasons[d.reason] += 1
    return dict(reasons)


def dataset_breakdown(decisions: List[MatchDecision]) -> Dict[str, int]:
    """
    Count successful matches by reference dataset.
    """
    counts = Counter()
    for d in decisions:
        if d.passed and d.matched_dataset:
            counts[d.matched_dataset] += 1
    return dict(counts)
