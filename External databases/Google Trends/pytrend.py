from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from pytrends.request import TrendReq


# -----------------------------
# SETTINGS
# -----------------------------
MAX_KW_PER_REQUEST = 5
DAILY_WINDOW_DAYS = 269          # safe "≤ ~9 months"
SLEEP_SECONDS = 2.0              # be polite (avoid rate limiting)
HL = "en-GB"
TZ = 0


# -----------------------------
# HELPERS
# -----------------------------
def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def make_daily_windows(start_date: str, end_date: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build consecutive windows that are short enough to return DAILY data.

    We use inclusive endpoints in the timeframe string, and to avoid overlap,
    we advance the next window start by +1 day from the previous window end.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if end < start:
        raise ValueError("end_date must be >= start_date")

    windows = []
    cur = start

    while cur <= end:
        win_end = min(cur + pd.Timedelta(days=DAILY_WINDOW_DAYS), end)
        windows.append((cur, win_end))
        cur = win_end + pd.Timedelta(days=1)  # ensures NO overlap in dates

    return windows


def assert_daily_index(df: pd.DataFrame, context: str = "") -> None:
    """
    Quick sanity check: verify df index increments in 1-day steps.
    (Allows small edge cases if df is tiny.)
    """
    if df.empty or len(df.index) < 3:
        return
    idx = pd.to_datetime(df.index).sort_values()
    deltas = idx.to_series().diff().dropna()
    if not ((deltas == pd.Timedelta(days=1)).all()):
        raise ValueError(f"Non-daily spacing detected {context}. Got deltas: {deltas.value_counts().head()}")


def fetch_iot_daily(
    pytrends: TrendReq,
    country: str,
    keywords: List[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch interest_over_time for a short window that should yield DAILY data.
    Returns a dataframe indexed by date, columns = keywords.
    """
    timeframe = f"{window_start.strftime('%Y-%m-%d')} {window_end.strftime('%Y-%m-%d')}"
    pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=country)
    df = pytrends.interest_over_time()

    if df is None or df.empty:
        return pd.DataFrame()

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df.index = pd.to_datetime(df.index)
    # Ensure columns are exactly the keywords we asked for (sometimes empty cols can appear)
    df = df[[c for c in df.columns if c in keywords]]

    # Sanity check (optional but useful)
    assert_daily_index(df, context=f"(country={country}, timeframe={timeframe}, keywords={keywords})")
    return df


# -----------------------------
# BUILD MASTER DATASET
# -----------------------------
def build_daily_master(
    countries: List[str],
    keywords: List[str],
    start_date: str,
    end_date: str,
    out_csv: str | Path,
    out_xlsx: str | Path,
    sleep_s: float = SLEEP_SECONDS,
) -> pd.DataFrame:
    """
    Produces:
      1) master CSV: country, date, keyword, value  (NO repeated (country,keyword,date))
      2) Excel workbook: one sheet per country (dates as rows, keywords as columns)

    All data pulled in DAILY windows.
    """
    pytrends = TrendReq(hl=HL, tz=TZ)
    kw_batches = chunk_list(keywords, MAX_KW_PER_REQUEST)
    windows = make_daily_windows(start_date, end_date)

    rows = []

    for country in countries:
        for (ws, we) in windows:
            for kw_batch in kw_batches:
                df_wide = fetch_iot_daily(pytrends, country, kw_batch, ws, we)

                if df_wide.empty:
                    time.sleep(sleep_s)
                    continue

                # Wide -> long
                df_long = (
                    df_wide.reset_index()
                          .rename(columns={"index": "date"})
                          .melt(id_vars=["date"], var_name="keyword", value_name="value")
                )
                df_long["country"] = country
                rows.append(df_long)

                time.sleep(sleep_s)

    if not rows:
        master = pd.DataFrame(columns=["country", "date", "keyword", "value"])
    else:
        master = pd.concat(rows, ignore_index=True)
        master["date"] = pd.to_datetime(master["date"])

        # HARD GUARANTEE: no repeats of the same day for the same (country, keyword)
        master = (
            master.sort_values(["country", "keyword", "date"])
                  .drop_duplicates(subset=["country", "keyword", "date"], keep="last")
                  .reset_index(drop=True)
        )

    # -----------------------------
    # Export CSV
    # -----------------------------
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)

    # -----------------------------
    # Export Excel (tabs per country)
    # -----------------------------
    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for country, sub in master.groupby("country"):
            wide = (
                sub.pivot_table(index="date", columns="keyword", values="value", aggfunc="mean")
                   .sort_index()
            )
            # Excel sheet name limit: 31 chars
            wide.to_excel(writer, sheet_name=country[:31])

    return master


# -----------------------------
# EXAMPLE RUN
# -----------------------------
if __name__ == "__main__":
    countries = ["CL"]  # Chile
    keywords = ["earthquake", "flood", "protests"]

    master = build_daily_master(
        countries=countries,
        keywords=keywords,
        start_date="2020-01-01",
        end_date="2020-12-31",
        out_csv="External databases/Google Trends/master_daily.csv",
        out_xlsx="External databases/Google Trends/master_daily.xlsx",
        sleep_s=2.0,
    )

    print(master.head())
    print("Rows:", len(master))
    print("Unique (country,keyword,date):", master.drop_duplicates(["country", "keyword", "date"]).shape[0])
