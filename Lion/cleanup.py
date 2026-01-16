# -*- coding: utf-8 -*-
# fix_url_title_meta_cache_20240115.py
#
# One last "belt + braces" cleaner:
# - Safely handles NaN/pd.NA
# - Unescapes HTML entities
# - Runs ftfy
# - Applies BOTH: (a) exact unicode-escape replacements (your confirmed case)
#                (b) control-byte variants (â€\x93 / â€\x94 etc)
#                (c) a re-decode fallback for classic mojibake (cp1252/latin1 -> utf-8)
# - Normalises whitespace
#
# Usage:
#   pip install pandas ftfy
#   python fix_url_title_meta_cache_20240115.py

from pathlib import Path
import html

import pandas as pd
from ftfy import fix_text


# --- Exact mojibake sequences using unicode escapes (avoids editor encoding issues) ---
# You observed: â€”Chi -> [0xe2, 0x20ac, 0x201d, ...] => "\u00e2\u20ac\u201d"
EM_DASH_BAD = "\u00e2\u20ac\u201d"  # mojibake for —
EN_DASH_BAD = "\u00e2\u20ac\u201c"  # mojibake for –

LSQUOTE_BAD = "\u00e2\u20ac\u02dc"  # sometimes appears; kept as fallback
RSQUOTE_BAD = "\u00e2\u20ac\u2122"
LDQUOTE_BAD = "\u00e2\u20ac\u0153"
RDQUOTE_BAD = "\u00e2\u20ac\u009d"

# Note: quote sequences vary across mangling; we'll also handle control-byte forms below.

# --- Control-byte variants (often present as â€\x93 / â€\x94, etc.) ---
CONTROL_MAP = {
    "â€\x91": "‘",
    "â€\x92": "’",
    "â€\x93": "–",
    "â€\x94": "—",
    "â€\x85": "…",
    "â€\x9c": "“",
    "â€\x9d": "”",
}

# --- Common literal variants (sometimes truly present) ---
LITERAL_MAP = {
    "â€“": "–",
    "â€”": "—",
    "â€˜": "‘",
    "â€™": "’",
    "â€œ": "“",
    "â€": "”",
    "â€¦": "…",
    "â„¢": "™",
    "Â£": "£",
    "Â€": "€",
    "Â ": " ",     # NBSP artifact
    "\u00A0": " ", # NBSP
}


def _try_redecode(s: str) -> str:
    """
    Attempt to reverse classic mojibake:
    text that was UTF-8 bytes decoded as cp1252/latin-1.
    Only returns the new string if it looks improved.
    """
    # Heuristic: mojibake often contains 'â', 'Â' sequences.
    suspicious = ("â" in s) or ("Â" in s) or ("Ã" in s)
    if not suspicious:
        return s

    # Try cp1252 -> utf-8
    try:
        s2 = s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
        # accept if it reduces mojibake markers
        if s2.count("â") + s2.count("Â") + s2.count("Ã") < s.count("â") + s.count("Â") + s.count("Ã"):
            return s2
    except Exception:
        pass

    # Try latin-1 -> utf-8
    try:
        s2 = s.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        if s2.count("â") + s2.count("Â") + s2.count("Ã") < s.count("â") + s.count("Â") + s.count("Ã"):
            return s2
    except Exception:
        pass

    return s


def fix_meta_str(x):
    """Fix a single cell (title/description)."""
    if pd.isna(x):
        return x

    s = str(x)

    # 1) HTML entities
    s = html.unescape(s)

    # 2) ftfy (one pass)
    s = fix_text(s)

    # 3) Exact unicode-escape replacements for your confirmed dash case
    s = s.replace(EM_DASH_BAD, "—").replace(EN_DASH_BAD, "–")

    # 4) Handle control-byte variants
    for bad, good in CONTROL_MAP.items():
        s = s.replace(bad, good)

    # 5) Handle common literal variants + NBSP
    for bad, good in LITERAL_MAP.items():
        s = s.replace(bad, good)

    # 6) If still suspicious, attempt re-decode + another ftfy pass, then repeat dash cleanup
    if ("â" in s) or ("Â" in s) or ("Ã" in s):
        s2 = _try_redecode(s)
        if s2 != s:
            s = fix_text(s2)
            s = s.replace(EM_DASH_BAD, "—").replace(EN_DASH_BAD, "–")
            for bad, good in CONTROL_MAP.items():
                s = s.replace(bad, good)
            for bad, good in LITERAL_MAP.items():
                s = s.replace(bad, good)

    # 7) Whitespace normalisation
    s = " ".join(s.split())
    return s


def clean_meta_fields_csv(
    in_csv: str | Path,
    out_csv: str | Path,
    *,
    title_col: str = "title",
    desc_col: str = "meta_description",
) -> None:
    in_csv = Path(in_csv)
    out_csv = Path(out_csv)

    df = pd.read_csv(in_csv)

    if title_col in df.columns:
        df[title_col] = df[title_col].apply(fix_meta_str)

    if desc_col in df.columns:
        df[desc_col] = df[desc_col].apply(fix_meta_str)

    df.to_csv(out_csv, index=False)

    # Quick sanity print: how many suspicious remnants remain?
    t_bad = 0
    d_bad = 0
    if title_col in df.columns:
        t_bad = df[title_col].astype(str).str.contains(r"[âÂÃ]", na=False).sum()
    if desc_col in df.columns:
        d_bad = df[desc_col].astype(str).str.contains(r"[âÂÃ]", na=False).sum()

    print(f"✅ Wrote cleaned CSV -> {out_csv}")
    print(f"🔎 Remaining suspicious markers: title={t_bad}, meta_description={d_bad}")


if __name__ == "__main__":
    clean_meta_fields_csv(
        in_csv="data/interim/_state/url_title_meta_cache_20240303.csv",
        out_csv="data/interim/_state/url_title_meta_cache_20240303_fixed.csv",
        title_col="title",
        desc_col="meta_description",
    )