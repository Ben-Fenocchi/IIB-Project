import pandas as pd
from pathlib import Path


def _load_extractions(base_dir: Path) -> pd.DataFrame:
    csv_path = base_dir / "results" / "consolidatedExtractions.csv"
    jsonl_path = base_dir / "results" / "consolidatedExtractions.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)
    else:
        raise FileNotFoundError(
            f"Could not find either {csv_path.name} or {jsonl_path.name} in {base_dir / 'results'}"
        )


def _truncate(s: str, max_chars: int) -> str:
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _print_coverage_stats(df: pd.DataFrame):
    """
    Print coverage statistics for dates and locations
    (known disruption types only).
    """
    known = df[df["disruption_type"] != "unknown"].copy()
    total = len(known)

    if total == 0:
        print("\nNo known disruption types found — skipping coverage stats.\n")
        return

    known["event_date"] = pd.to_datetime(known.get("event_date"), errors="coerce")
    known["publish_date"] = pd.to_datetime(known.get("publish_date"), errors="coerce")

    has_event_date = known["event_date"].notna()
    has_publish_only = known["event_date"].isna() & known["publish_date"].notna()
    has_no_date = known["event_date"].isna() & known["publish_date"].isna()

    has_location = known["location_name"].astype(str).str.strip().ne("")

    print("\n=== Coverage diagnostics (known disruption types only) ===\n")

    print("Date coverage:")
    print(f"- Event date extracted       : {has_event_date.sum():5d} / {total} "
          f"({100 * has_event_date.mean():5.1f}%)")
    print(f"- Publish date only (proxy)  : {has_publish_only.sum():5d} / {total} "
          f"({100 * has_publish_only.mean():5.1f}%)")
    print(f"- No date available          : {has_no_date.sum():5d} / {total} "
          f"({100 * has_no_date.mean():5.1f}%)")

    print("\nLocation coverage:")
    print(f"- Location extracted         : {has_location.sum():5d} / {total} "
          f"({100 * has_location.mean():5.1f}%)")

    print("\n=========================================================\n")


def view_extractions(
    max_rows: int = 30,
    title_max_chars: int = 40,
    location_max_chars: int = 40,
):
    base_dir = Path(__file__).resolve().parent
    df = _load_extractions(base_dir)
    df = df.fillna("")

    # ---- Explanation of date formats ----
    print(
        "\nDate notation used below:\n"
        "- YYYY-MM-DD : event date extracted from article text (preferred)\n"
        "- YYYY/MM/DD : article publish date (metadata proxy, used when no event date)\n"
    )

    # ---- Coverage diagnostics ----
    _print_coverage_stats(df)

    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce", utc=True).dt.tz_convert(None)
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce", utc=True).dt.tz_convert(None)

    def display_date(row) -> str:
        if pd.notna(row["event_date"]):
            return row["event_date"].strftime("%Y-%m-%d")
        if pd.notna(row["publish_date"]):
            return row["publish_date"].strftime("%Y/%m/%d")
        return ""

    df["display_date"] = df.apply(display_date, axis=1)

    df["title_short"] = df["source_title"].apply(lambda s: _truncate(s, title_max_chars))
    df["location_short"] = df["location_name"].apply(lambda s: _truncate(s, location_max_chars))

    view = df[[
        "title_short",
        "disruption_type",
        "display_date",
        "location_short",
        "duration_hours",
        "confidence",
    ]].rename(columns={
        "title_short": "title",
        "disruption_type": "type",
        "display_date": "date",
        "location_short": "location",
        "duration_hours": "duration_h",
    })

    view["confidence"] = _to_numeric(view["confidence"])
    view["duration_h"] = _to_numeric(view["duration_h"])

    view = view.sort_values("confidence", ascending=False, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", None)

    print("\n=== Disruption Type Counts ===\n")
    counts = view["type"].value_counts().sort_index()
    print(counts.to_frame(name="count").to_string())

    # ---- Print floods first, then the rest ----
    ordered_types = []
    if "flood" in view["type"].unique():
        ordered_types.append("flood")
    ordered_types += sorted(t for t in view["type"].unique() if t not in ("flood", "unknown"))

    for dtype in ordered_types:
        group = view[view["type"] == dtype]
        if len(group) == 0:
            continue

        print(f"\n=== {dtype.upper()} ({len(group)} events) ===\n")

        if len(group) <= max_rows:
            print(group.to_string(index=False))
        else:
            print(group.head(max_rows).to_string(index=False))
            print(f"\n... ({len(group) - max_rows} more rows not shown) ...\n")


if __name__ == "__main__":
    view_extractions()
