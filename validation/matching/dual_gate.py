"""
Dual-gate verification for validation matching.

This file converts scored candidate matches into final match decisions.
It implements two complementary passes:

1) Forward validation:
   For each extracted event, find the best-matching reference event.
   Used as a proxy for precision ("did this extracted event correspond
   to a known real-world event?").

2) Inverse validation:
   For each reference event, find the best-matching extracted event.
   Used as a proxy for coverage ("was this known event captured?").

Flowchart role:
- 'Dual-Gate Verification' stage (Forward + Inverse)

This module does not:
- generate candidates
- compute similarity scores
- write outputs to disk
"""

from typing import Dict, List, Optional

from validation.models import CandidateMatch, MatchDecision


# ------------------ CONFIG ------------------ #

DEFAULT_SCORE_THRESHOLD = 0.6


# ------------------ FORWARD VALIDATION ------------------ #

def forward_validation(
    scored_candidates: List[CandidateMatch],
    threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[MatchDecision]:
    """
    For each extracted event, select the highest-scoring reference match
    and decide whether it passes the score threshold.
    """
    # Group candidate matches by extracted event
    by_extracted: Dict[str, List[CandidateMatch]] = {}
    for c in scored_candidates:
        by_extracted.setdefault(c.extracted_id, []).append(c)

    decisions: List[MatchDecision] = []

    for extracted_id, matches in by_extracted.items():
        # Select best-scoring reference for this extracted event
        best = max(matches, key=lambda m: m.score)

        if best.score >= threshold:
            decisions.append(
                MatchDecision(
                    source_id=extracted_id,
                    matched_id=best.ref_id,
                    matched_dataset=best.dataset,
                    score=best.score,
                    passed=True,
                    reason="score_above_threshold",
                )
            )
        else:
            decisions.append(
                MatchDecision(
                    source_id=extracted_id,
                    matched_id=None,
                    matched_dataset=None,
                    score=best.score,
                    passed=False,
                    reason="score_below_threshold",
                )
            )

    return decisions


# ------------------ INVERSE VALIDATION ------------------ #

def inverse_validation(
    scored_candidates: List[CandidateMatch],
    threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[MatchDecision]:
    """
    For each reference event, select the highest-scoring extracted match
    and decide whether it passes the score threshold.
    """
    # Group candidate matches by reference event
    by_reference: Dict[str, List[CandidateMatch]] = {}
    for c in scored_candidates:
        by_reference.setdefault(c.ref_id, []).append(c)

    decisions: List[MatchDecision] = []

    for ref_id, matches in by_reference.items():
        # Select best-scoring extracted event for this reference
        best = max(matches, key=lambda m: m.score)

        if best.score >= threshold:
            decisions.append(
                MatchDecision(
                    source_id=ref_id,
                    matched_id=best.extracted_id,
                    matched_dataset=best.dataset,
                    score=best.score,
                    passed=True,
                    reason="score_above_threshold",
                )
            )
        else:
            decisions.append(
                MatchDecision(
                    source_id=ref_id,
                    matched_id=None,
                    matched_dataset=None,
                    score=best.score,
                    passed=False,
                    reason="score_below_threshold",
                )
            )

    return decisions


# ------------------ ORCHESTRATOR ------------------ #

def run_dual_gate_validation(
    scored_candidates: List[CandidateMatch],
    threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Dict[str, List[MatchDecision]]:
    """
    Run both forward and inverse validation passes.

    Returns a dictionary with two lists of MatchDecision objects:
    - 'forward': extracted -> reference decisions
    - 'inverse': reference -> extracted decisions
    """
    return {
        "forward": forward_validation(scored_candidates, threshold),
        "inverse": inverse_validation(scored_candidates, threshold),
    }
