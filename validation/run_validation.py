"""
Validation pipeline runner.

This script orchestrates the full validation workflow for flood events.
It is the single place where control flow lives.

High-level responsibilities:
1) Load extracted events
2) Decide the reference caching time window
3) Cache reference datasets (optional)
4) Run validation against cached references
5) Write validation outputs

All heavy logic lives in imported modules.
This file only sequences steps in the correct order.
"""

from pathlib import Path
from datetime import timedelta

# ------------------ I/O ------------------ #

from io import (
    load_extracted_events,
    write_json,
    write_csv,
)

# ------------------ EXTRACTED EVENT PROCESSING ------------------ #

from extracted.filtering import filter_by_type
from extracted.profiling import profile_extracted_events
from extracted.metadata_inference import to_canonical_extracted

# ------------------ REFERENCE DATA ------------------ #

from reference.load_flood_references import load_all_flood_references
from reference.standardise import standardise_reference_events

# ------------------ MATCHING ------------------ #

from matching.candidate_generation import generate_candidates
from matching.scoring import score_candidate
from matching.dual_gate import run_dual_gate_validation

# ------------------ REPORTING ------------------ #

from outputs.report import (
    build_summary,
    decisions_to_rows,
    failure_reasons,
    dataset_breakdown,
)

# ------------------ REFERENCE CACHING ------------------ #

from reference_cache.floods.cache_reliefweb import cache_reliefweb_floods
from reference_cache.floods.cache_gdacs import cache_gdacs_floods
from reference_cache.floods.cache_dfo import cache_dfo_floods
from reference_cache.floods.cache_emdat import cache_emdat_floods


# ------------------ CONFIG ------------------ #

# Number of days added on either side of the extracted-event window
# when caching reference datasets
CACHE_BUFFER_DAYS = 14


# ------------------ WINDOW SELECTION ------------------ #

def compute_cache_window(canonical_events):
    """
    Determine the time window for reference caching.

    Uses the earliest and latest dates present in the extracted events,
    ignoring events with missing dates, and expands the window by a
    fixed buffer to avoid edge effects.
    """
    dates = [
        e.date_start
        for e in canonical_events
        if e.date_start is not None
    ]

    if not dates:
        raise RuntimeError("No dated extracted events available")

    start = min(dates) - timedelta(days=CACHE_BUFFER_DAYS)
    end = max(dates) + timedelta(days=CACHE_BUFFER_DAYS)

    return start, end


# ------------------ MAIN PIPELINE ------------------ #

def run_validation(
    extracted_path: Path,
    reference_cache_dir: Path,
    output_dir: Path,
    *,
    run_caching: bool = True,
    reliefweb_appname: str | None = None,
    dfo_source_csv: Path | None = None,
    emdat_source_csv: Path | None = None,
) -> None:
    """
    Run flood validation end-to-end.
    """

    # --------------------------------------------------
    # 1) Load extracted disruption events from disk
    # --------------------------------------------------
    extracted_raw = load_extracted_events(extracted_path)

    # --------------------------------------------------
    # 2) Restrict validation scope to floods only
    # --------------------------------------------------
    extracted_floods = filter_by_type(extracted_raw, {"flood"})

    # --------------------------------------------------
    # 3) Profile extracted events for diagnostics
    # (no effect on matching)
    # --------------------------------------------------
    extracted_profile = profile_extracted_events(extracted_floods)

    # --------------------------------------------------
    # 4) Infer and normalise extracted metadata
    # (convert to canonical representation)
    # --------------------------------------------------
    extracted_canonical = [
        to_canonical_extracted(e)
        for e in extracted_floods
    ]

    # --------------------------------------------------
    # 5) Decide reference caching time window
    # (single source of truth for all datasets)
    # --------------------------------------------------
    cache_start, cache_end = compute_cache_window(extracted_canonical)

    # --------------------------------------------------
    # 6) Cache reference datasets (optional step)
    # This performs all API calls / disk reads
    # --------------------------------------------------
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    if run_caching:
        if not reliefweb_appname:
            raise ValueError("ReliefWeb appname required for caching")

        cache_reliefweb_floods(
            start_date=cache_start,
            end_date=cache_end,
            output_path=reference_cache_dir / "reliefweb.json",
            appname=reliefweb_appname,
        )

        cache_gdacs_floods(
            start_date=cache_start,
            end_date=cache_end,
            output_path=reference_cache_dir / "gdacs.json",
        )

        if not dfo_source_csv:
            raise ValueError("DFO source CSV required for caching")

        cache_dfo_floods(
            start_date=cache_start,
            end_date=cache_end,
            source_csv=dfo_source_csv,
            output_path=reference_cache_dir / "dfo.json",
        )

        if not emdat_source_csv:
            raise ValueError("EM-DAT source CSV required for caching")

        cache_emdat_floods(
            start_date=cache_start,
            end_date=cache_end,
            source_csv=emdat_source_csv,
            output_path=reference_cache_dir / "emdat.json",
        )

    # --------------------------------------------------
    # 7) Load cached reference datasets
    # (no network access from this point onwards)
    # --------------------------------------------------
    reference_raw = load_all_flood_references(
        dfo_path=reference_cache_dir / "dfo.json",
        gdacs_path=reference_cache_dir / "gdacs.json",
        emdat_path=reference_cache_dir / "emdat.json",
        reliefweb_path=reference_cache_dir / "reliefweb.json",
    )

    # --------------------------------------------------
    # 8) Standardise reference events
    # (convert to canonical representation)
    # --------------------------------------------------
    reference_canonical = standardise_reference_events(reference_raw)

    # --------------------------------------------------
    # 9) Generate candidate extracted–reference pairs
    # --------------------------------------------------
    candidate_map = generate_candidates(
        extracted_canonical,
        reference_canonical,
    )

    # --------------------------------------------------
    # 10) Score all candidate pairs
    # --------------------------------------------------
    ext_lookup = {e.id: e for e in extracted_canonical}
    ref_lookup = {r.id: r for r in reference_canonical}

    scored_candidates = []
    for extracted_id, ref_ids in candidate_map.items():
        e = ext_lookup[extracted_id]
        for ref_id in ref_ids:
            scored_candidates.append(
                score_candidate(e, ref_lookup[ref_id])
            )

    # --------------------------------------------------
    # 11) Dual-gate validation
    # (forward + inverse passes)
    # --------------------------------------------------
    decisions = run_dual_gate_validation(scored_candidates)
    forward = decisions["forward"]
    inverse = decisions["inverse"]

    # --------------------------------------------------
    # 12) Aggregate validation outputs
    # --------------------------------------------------
    summary = build_summary(forward, inverse)

    diagnostics = {
        "forward_failure_reasons": failure_reasons(forward),
        "inverse_failure_reasons": failure_reasons(inverse),
        "forward_dataset_breakdown": dataset_breakdown(forward),
    }

    # --------------------------------------------------
    # 13) Write outputs to disk
    # --------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "extracted_profile.json", extracted_profile)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "diagnostics.json", diagnostics)

    write_csv(output_dir / "forward_matches.csv", decisions_to_rows(forward))
    write_csv(output_dir / "inverse_matches.csv", decisions_to_rows(inverse))


# ------------------ CLI ------------------ #

if __name__ == "__main__":
    run_validation(
        extracted_path=Path("results/extractions.jsonl"),
        reference_cache_dir=Path("cache/floods"),
        output_dir=Path("validation_outputs"),
        run_caching=True,
        reliefweb_appname="your-approved-appname",
        dfo_source_csv=Path("raw_data/dfo.csv"),
        emdat_source_csv=Path("raw_data/emdat.csv"),
    )
