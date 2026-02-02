from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class ILOSTATPanelConfig:
    geo_col: str = "ref_area"      # ISO-3 code for countries in bulk tables (ref_area). :contentReference[oaicite:6]{index=6}
    time_col: str = "time"
    value_col: str = "obs_value"


def build_country_time_panel(
    df: pd.DataFrame,
    *,
    cfg: Optional[ILOSTATPanelConfig] = None,
    filters: Optional[Dict[str, str]] = None,
    keep_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Returns tidy table: [ref_area, time, obs_value] + optional kept dimensions.

    filters: exact-match filters for dimensions (e.g. {"sex": "SEX_T", "classif1": "AGE_Y15-24"}).
    keep_cols: additional columns to retain (e.g. ["indicator", "sex", "classif1", "classif2", "source"]).
    """
    cfg = cfg or ILOSTATPanelConfig()
    out = df.copy()

    # Apply dimension filters
    if filters:
        for k, v in filters.items():
            if k in out.columns:
                out = out[out[k].astype(str) == str(v)]
            else:
                raise ValueError(f"Filter column '{k}' not found in dataset columns.")

    base_cols = [cfg.geo_col, cfg.time_col, cfg.value_col]
    for c in base_cols:
        if c not in out.columns:
            raise ValueError(f"Required column '{c}' not found. Available: {list(out.columns)[:30]} ...")

    # Keep selection
    cols = base_cols.copy()
    if keep_cols:
        cols += [c for c in keep_cols if c in out.columns and c not in cols]

    out = out[cols].copy()

    # Coerce types
    out[cfg.geo_col] = out[cfg.geo_col].astype(str).str.upper().str.strip()
    out[cfg.value_col] = pd.to_numeric(out[cfg.value_col], errors="coerce")

    return out.sort_values([cfg.geo_col, cfg.time_col]).reset_index(drop=True)
