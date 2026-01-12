import pandas as pd
from pathlib import Path


def view_extractions(max_rows: int = 30, title_words: int = 8):
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results" / "extractions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    # Fill NaNs for nicer display
    df = df.fillna("")

    # Create a shortened title column
    def short_title(s: str) -> str:
        words = str(s).split()
        return " ".join(words[:title_words]) + ("…" if len(words) > title_words else "")

    df["title_short"] = df["source_title"].apply(short_title)

    # Select and rename columns for display
    view = df[[
        "title_short",
        "disruption_type",
        "event_date",
        "location_name",
        "duration_hours",
        "confidence",
    ]].rename(columns={
        "title_short": "title",
        "disruption_type": "type",
        "event_date": "date",
        "location_name": "location",
        "duration_hours": "duration_h",
    })

    # Sort by confidence descending (most reliable first)
    view = view.sort_values("confidence", ascending=False)

    # Display settings for terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 40)

    print("\n=== Extraction Summary ===\n")

    if len(view) <= max_rows:
        print(view.to_string(index=False))
    else:
        print(view.head(max_rows).to_string(index=False))
        print(f"\n... ({len(view) - max_rows} more rows not shown) ...\n")


if __name__ == "__main__":
    view_extractions()
