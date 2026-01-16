import pandas as pd
from pathlib import Path


def _load_extractions(base_dir: Path) -> pd.DataFrame:
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


def view_extractions(
    max_rows: int = 30,
    title_max_chars: int = 40,
    location_max_chars: int = 40,
):
    base_dir = Path(__file__).resolve().parent
    df = _load_extractions(base_dir)

    # Fill NaNs for nicer display
    df = df.fillna("")

    # Truncate title + location to keep tables readable
    df["title_short"] = df["source_title"].apply(lambda s: _truncate(s, title_max_chars))
    df["location_short"] = df["location_name"].apply(lambda s: _truncate(s, location_max_chars))

    # Select and rename columns for display
    view = df[[
        "title_short",
        "disruption_type",
        "event_date",
        "location_short",
        "duration_hours",
        "confidence",
    ]].rename(columns={
        "title_short": "title",
        "disruption_type": "type",
        "event_date": "date",
        "location_short": "location",
        "duration_hours": "duration_h",
    })

    # Sort by confidence descending (most reliable first)
    view = view.sort_values("confidence", ascending=False)

    # Display settings for terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", None)

    print("\n=== Disruption Type Counts ===\n")
    counts = view["type"].value_counts().sort_index()
    print(counts.to_frame(name="count").to_string())

    # Print per-type tables, skipping unknowns
    for dtype, group in view.groupby("type"):
        if dtype == "unknown" or len(group) == 0:
            continue

        print(f"\n=== {dtype.upper()} ({len(group)} events) ===\n")

        if len(group) <= max_rows:
            print(group.to_string(index=False))
        else:
            print(group.head(max_rows).to_string(index=False))
            print(f"\n... ({len(group) - max_rows} more rows not shown) ...\n")


if __name__ == "__main__":
    view_extractions()
