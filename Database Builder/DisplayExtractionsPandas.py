import pandas as pd
from pathlib import Path


def _load_extractions(base_dir: Path) -> pd.DataFrame:
    #csv_path = base_dir / "results" / "dedupedExtractions.csv"
    #jsonl_path = base_dir / "results" / "dedupedExtractions.jsonl"
    csv_path = base_dir / "results" / "extractions.csv"
    jsonl_path = base_dir / "results" / "extractions.jsonl"

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


def _save_plots(df: pd.DataFrame, out_dir: Path, top_n_types: int = 12):
    """
    Produces three plots:
      1) Disruption type counts (excluding 'unknown')
      2) Known vs unknown disruption counts
      3) Confidence histogram (excluding 'unknown')
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["type"] = df["disruption_type"].astype(str).str.strip()
    df["confidence"] = _to_numeric(df.get("confidence"))

    known = df[df["type"] != "unknown"]
    type_counts = known["type"].value_counts()

    if len(type_counts) > top_n_types:
        top = type_counts.iloc[:top_n_types]
        remainder = type_counts.iloc[top_n_types:].sum()
        type_counts = pd.concat([top, pd.Series({"other": remainder})])

    plt.figure(figsize=(8, 4.2))
    type_counts.sort_values(ascending=False).plot(kind="bar")
    plt.xlabel("")
    plt.ylabel("count")
    plt.title("Extracted disruption events by type (excluding unknown)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "disruption_type_counts_known.png", dpi=300)
    plt.close()

    known_count = (df["type"] != "unknown").sum()
    unknown_count = (df["type"] == "unknown").sum()

    plt.figure(figsize=(4.5, 4.2))
    plt.bar(["known", "unknown"], [known_count, unknown_count])
    plt.ylabel("count")
    plt.title("Known vs unknown extractions")
    plt.tight_layout()
    plt.savefig(out_dir / "known_vs_unknown.png", dpi=300)
    plt.close()

    conf = known["confidence"].dropna()
    if len(conf) > 0:
        plt.figure(figsize=(8, 4.2))
        plt.hist(conf, bins=20)
        plt.xlabel("confidence")
        plt.ylabel("count")
        plt.title("Confidence score distribution (known events)")
        plt.tight_layout()
        plt.savefig(out_dir / "confidence_histogram_known.png", dpi=300)
        plt.close()

    print("\n=== Plots saved ===")
    print(f"- {out_dir / 'disruption_type_counts_known.png'}")
    print(f"- {out_dir / 'known_vs_unknown.png'}")
    if len(conf) > 0:
        print(f"- {out_dir / 'confidence_histogram_known.png'}")


def view_extractions(
    max_rows: int = 30,
    title_max_chars: int = 40,
    location_max_chars: int = 40,
    make_plots: bool = True,
    plots_dirname: str = "plots",
):
    base_dir = Path(__file__).resolve().parent
    df = _load_extractions(base_dir)
    df = df.fillna("")

    # ---- Coverage diagnostics ----
    _print_coverage_stats(df)

    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")

    def display_date(row) -> str:
        """
        Prefer extracted event_date.
        Fall back to publish_date (YYYY/MM/DD).
        """
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

    for dtype, group in view.groupby("type"):
        if dtype == "unknown" or len(group) == 0:
            continue

        print(f"\n=== {dtype.upper()} ({len(group)} events) ===\n")

        if len(group) <= max_rows:
            print(group.to_string(index=False))
        else:
            print(group.head(max_rows).to_string(index=False))
            print(f"\n... ({len(group) - max_rows} more rows not shown) ...\n")

    if make_plots:
        _save_plots(df, base_dir / plots_dirname)


if __name__ == "__main__":
    view_extractions()
