import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure outputs save next to this script (not the current working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set plot style
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")


# ------------------0) PRICING (USD per 1M tokens) ------------------#
# Source: OpenAI pricing table (Standard tier). Update if you change models/tier.
# Missing models will be warned + skipped in the estimate. :contentReference[oaicite:3]{index=3}
PRICES_PER_1M = {
    # GPT-5 family
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},

    # Common others (examples from pricing page; extend as needed)
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}

def _resolve_price_key(model_name: str) -> str | None:
    """
    Try to map a concrete model name (e.g., gpt-4o-mini-2024-07-18)
    to a base pricing key (e.g., gpt-4o-mini).
    """
    if not model_name:
        return None
    if model_name in PRICES_PER_1M:
        return model_name

    # Common versioned model names
    for k in PRICES_PER_1M.keys():
        if model_name == k or model_name.startswith(k + "-"):
            return k

    # chat-latest / codex aliases sometimes appear
    # e.g. gpt-5.1-chat-latest -> gpt-5.1
    for prefix in ["-chat-latest", "-codex", "-codex-mini", "-codex-max", "-search-api"]:
        if model_name.endswith(prefix):
            base = model_name[: -len(prefix)]
            if base in PRICES_PER_1M:
                return base

    return None


def estimate_cost_usd(input_tokens: int, input_cached_tokens: int, output_tokens: int, model_name: str) -> float | None:
    """
    Estimate USD cost for a single model given token usage.
    Uses: (non_cached_input * input_price + cached_input * cached_price + output * output_price) / 1e6
    """
    key = _resolve_price_key(model_name)
    if not key:
        return None

    p = PRICES_PER_1M[key]
    in_total = int(input_tokens or 0)
    in_cached = int(input_cached_tokens or 0)
    out_total = int(output_tokens or 0)

    in_non_cached = max(in_total - in_cached, 0)

    usd = (
        (in_non_cached * p["input"])
        + (in_cached * p["cached_input"])
        + (out_total * p["output"])
    ) / 1_000_000.0

    return float(usd)


# ------------------1) API HELPERS------------------#
def get_admin_headers():
    """Build headers using the admin key from environment variables."""
    OPENAI_ADMIN_KEY = os.getenv("OPENAI_ADMIN_KEY")
    if not OPENAI_ADMIN_KEY:
        raise ValueError("OPENAI_ADMIN_KEY not found in environment variables")
    return {
        "Authorization": f"Bearer {OPENAI_ADMIN_KEY}",
        "Content-Type": "application/json",
    }


def get_data(url, params):
    """Reusable function for retrieving paginated data from the API"""
    headers = get_admin_headers()

    # Initialize an empty list to store all data
    all_data = []
    # Initialize pagination cursor
    page_cursor = None

    # Loop to handle pagination
    while True:
        request_params = dict(params)  # don't mutate caller
        if page_cursor:
            request_params["page"] = page_cursor

        response = requests.get(url, headers=headers, params=request_params)

        if response.status_code == 200:
            data_json = response.json()
            all_data.extend(data_json.get("data", []))
            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break
        else:
            # Common edge-case: API can sometimes return an invalid next_page token
            # If that happens, stop pagination and keep what we already fetched.
            if response.status_code == 400 and "page token is invalid" in response.text.lower():
                print("Warning: pagination token invalid; stopping pagination and using data fetched so far.")
                break

            print(f"Error: {response.status_code}")
            print(response.text)
            break

    if all_data:
        print("Data retrieved successfully!")
    else:
        print("Issue: No data available to retrieve.")

    return all_data


def get_project_id(project_name):
    """Fetches the project ID for a given project name."""
    url = "https://api.openai.com/v1/organization/projects"
    headers = get_admin_headers()

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            projects = response.json().get("data", [])
            for project in projects:
                if project.get("name") == project_name:
                    return project.get("id")
            print(f"Project '{project_name}' not found.")
            return None
        else:
            print(f"Error fetching projects: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception fetching projects: {e}")
        return None


# ------------------2) TIME WINDOW FETCH ------------------#
def fetch_daily_buckets_in_windows(url, base_params, start_time, end_time):
    """
    The API enforces max limit=31 for bucket_width=1d.
    To fetch from Oct 1st to now, we split into 31-day windows and concatenate.
    """
    SECONDS_PER_DAY = 24 * 60 * 60
    MAX_DAYS_PER_CALL = 31

    all_buckets = []
    cursor_start = start_time

    while cursor_start < end_time:
        window_end = min(end_time, cursor_start + MAX_DAYS_PER_CALL * SECONDS_PER_DAY)

        params = dict(base_params)
        params["start_time"] = int(cursor_start)
        params["end_time"] = int(window_end)
        params["bucket_width"] = "1d"
        params["limit"] = MAX_DAYS_PER_CALL

        window_data = get_data(url, params)
        all_buckets.extend(window_data)

        # Move to next window (avoid overlap)
        cursor_start = window_end

        if window_end == end_time:
            break

    # De-duplicate buckets by (start_time,end_time) just in case overlapping windows occur
    dedup = {}
    for b in all_buckets:
        key = (b.get("start_time"), b.get("end_time"))
        dedup[key] = b
    return [dedup[k] for k in sorted(dedup.keys())]


# ------------------3) COST: CUMULATIVE PLOT ONLY (OFFICIAL) ------------------#
def analyze_costs_cumulative(project_id=None):
    """
    Produces: a plot of cumulative cost from Oct 1st 2025 to present day.
    Returns: daily cost dataframe with columns [date_utc, official_daily_cost_usd, cumulative_cost_usd]
    """
    print("\nFetching Cost Data (official Costs endpoint)...")
    url = "https://api.openai.com/v1/organization/costs"

    # Fixed start time: 1st October 2025 00:00 UTC
    start_dt = datetime.datetime(2025, 10, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_time = int(start_dt.timestamp())

    # End time: now
    end_time = int(time.time())

    base_params = {}
    if project_id:
        base_params["project_ids"] = [project_id]

    cost_buckets = fetch_daily_buckets_in_windows(url, base_params, start_time, end_time)
    if not cost_buckets:
        print("No cost data available.")
        return pd.DataFrame(columns=["date_utc", "official_daily_cost_usd", "cumulative_cost_usd"])

    # Parse cost data into daily totals
    rows = []
    for bucket in cost_buckets:
        bucket_start = bucket.get("start_time")
        results = bucket.get("results", [])
        day_cost = 0.0
        for r in results:
            day_cost += float(r.get("amount", {}).get("value", 0) or 0.0)

        rows.append({
            "start_time_unix_s": bucket_start,
            "official_daily_cost_usd": day_cost
        })

    daily = pd.DataFrame(rows)
    daily["date_utc"] = pd.to_datetime(daily["start_time_unix_s"], unit="s", utc=True).dt.date

    # Combine any accidental duplicates
    daily = daily.groupby("date_utc", as_index=False)["official_daily_cost_usd"].sum()

    # Ensure explicit row for every day (missing => 0.0)
    start_date = pd.to_datetime(start_time, unit="s", utc=True).date()
    end_date = pd.to_datetime(end_time, unit="s", utc=True).date()
    full_days = pd.date_range(start=start_date, end=end_date, freq="D").date

    daily = (
        daily.set_index("date_utc")
             .reindex(full_days, fill_value=0.0)
             .rename_axis("date_utc")
             .reset_index()
    )

    daily["official_daily_cost_usd"] = daily["official_daily_cost_usd"].round(3)

    # Cumulative
    daily = daily.sort_values("date_utc").reset_index(drop=True)
    daily["cumulative_cost_usd"] = daily["official_daily_cost_usd"].cumsum().round(3)

    # Plot: cumulative total cost since Oct 1 2025
    plt.figure(figsize=(12, 6))
    plt.plot(daily["date_utc"], daily["cumulative_cost_usd"], linewidth=2.0)
    plt.xlabel("Date (UTC)", fontsize=12)
    plt.ylabel("Cumulative cost (USD)", fontsize=12)
    plt.title(f"Cumulative Total Cost Since Oct 1 2025{' (Filtered)' if project_id else ''}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="both", linestyle="--", alpha=0.5)

    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))

    plt.tight_layout()
    out_png = os.path.join(BASE_DIR, "cost_cumulative_chart.png")
    print(f"Saving Cumulative Cost Chart to {out_png}...")
    plt.savefig(out_png, dpi=300)
    plt.close()

    return daily[["date_utc", "official_daily_cost_usd", "cumulative_cost_usd"]]


# ------------------4) USAGE + COST TABLE (OFFICIAL + ESTIMATED) ------------------#
def usage_cost_table(project_id=None):
    """
    Produces: ONE table (pandas) of each day where there was usage.
    Adds:
      - official_daily_cost_usd (Costs endpoint, can lag today)
      - estimated_daily_cost_usd (Usage tokens * pricing, updates intraday)
    Saves: daily_usage_cost_table.csv
    """
    print("\nFetching Usage Data...")
    url = "https://api.openai.com/v1/organization/usage/completions"

    # Fixed start time: 1st October 2025 00:00 UTC
    start_dt = datetime.datetime(2025, 10, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_time = int(start_dt.timestamp())

    # End time: now (exclusive in API) :contentReference[oaicite:4]{index=4}
    end_time = int(time.time())

    base_params = {
        # We want model so we can price tokens -> dollars
        "group_by": ["model", "project_id"],
        "bucket_width": "1d",
    }
    if project_id:
        base_params["project_ids"] = [project_id]

    usage_buckets = fetch_daily_buckets_in_windows(url, base_params, start_time, end_time)
    if not usage_buckets:
        print("No usage data available.")
        empty = pd.DataFrame(columns=[
            "date_utc", "requests_count", "input_tokens", "output_tokens",
            "official_daily_cost_usd", "estimated_daily_cost_usd"
        ])
        print("\nDaily usage/cost table is empty.")
        return empty

    # --- Parse usage buckets into (date, model) rows so we can estimate costs ---
    model_rows = []
    for bucket in usage_buckets:
        bucket_start = bucket.get("start_time")
        date_utc = pd.to_datetime(bucket_start, unit="s", utc=True).date()

        for r in bucket.get("results", []):
            model_rows.append({
                "date_utc": date_utc,
                "model": r.get("model"),
                "requests_count": int(r.get("num_model_requests", 0) or 0),
                "input_tokens": int(r.get("input_tokens", 0) or 0),
                "input_cached_tokens": int(r.get("input_cached_tokens", 0) or 0),
                "output_tokens": int(r.get("output_tokens", 0) or 0),
            })

    usage_by_model = pd.DataFrame(model_rows)
    if usage_by_model.empty:
        print("Usage buckets returned but contained no results.")
        empty = pd.DataFrame(columns=[
            "date_utc", "requests_count", "input_tokens", "output_tokens",
            "official_daily_cost_usd", "estimated_daily_cost_usd"
        ])
        return empty

    # Aggregate in case of duplicates
    usage_by_model = (
        usage_by_model
        .groupby(["date_utc", "model"], as_index=False)[
            ["requests_count", "input_tokens", "input_cached_tokens", "output_tokens"]
        ].sum()
    )

    # Estimate USD per (date, model)
    missing_models = set()
    est_costs = []
    for _, row in usage_by_model.iterrows():
        est = estimate_cost_usd(
            input_tokens=row["input_tokens"],
            input_cached_tokens=row["input_cached_tokens"],
            output_tokens=row["output_tokens"],
            model_name=row["model"] or "",
        )
        if est is None:
            missing_models.add(row["model"])
            est = 0.0
        est_costs.append(est)

    usage_by_model["estimated_cost_usd_model"] = est_costs

    if missing_models:
        missing_models_str = ", ".join(sorted([m for m in missing_models if m]))
        print("\nWARNING: Some models were not found in PRICES_PER_1M and were skipped in the estimate:")
        print(f"  {missing_models_str}")
        print("Add them to PRICES_PER_1M at the top of the script to improve estimate accuracy.")

    # Collapse to daily totals across models
    usage_daily = (
        usage_by_model
        .groupby("date_utc", as_index=False)[
            ["requests_count", "input_tokens", "output_tokens"]
        ].sum()
    )

    estimated_daily = (
        usage_by_model
        .groupby("date_utc", as_index=False)["estimated_cost_usd_model"]
        .sum()
        .rename(columns={"estimated_cost_usd_model": "estimated_daily_cost_usd"})
    )

    # Keep only days where there was usage
    usage_daily = usage_daily[usage_daily["requests_count"] > 0].copy()
    usage_daily = usage_daily.sort_values("date_utc").reset_index(drop=True)

    # Merge in estimated costs (updates intraday)
    table = usage_daily.merge(estimated_daily, on="date_utc", how="left")
    table["estimated_daily_cost_usd"] = table["estimated_daily_cost_usd"].fillna(0.0).round(4)

    # Fetch official daily cost series and merge
    costs_daily = analyze_costs_cumulative(project_id=project_id)[["date_utc", "official_daily_cost_usd"]].copy()
    table = table.merge(costs_daily, on="date_utc", how="left")
    table["official_daily_cost_usd"] = table["official_daily_cost_usd"].fillna(0.0).round(3)

    # Save CSV next to this script
    out_csv = os.path.join(BASE_DIR, "daily_usage_cost_table.csv")
    print(f"Saving Daily Usage/Cost Table to {out_csv}...")
    table.to_csv(out_csv, index=False)

    # Display a readable table output in the console
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)

    print("\nDaily Usage/Cost Table (days with usage only):")
    if len(table) <= 60:
        print(table.to_string(index=False))
    else:
        print(table.head(30).to_string(index=False))
        print("\n... (snip) ...\n")
        print(table.tail(30).to_string(index=False))

    return table


if __name__ == "__main__":
    target_project_name = "database"
    print(f"Resolving project ID for '{target_project_name}'...")
    project_id = get_project_id(target_project_name)

    if project_id:
        print(f"Found Project ID: {project_id}")

        # cumulative official cost since Oct 1 2025
        analyze_costs_cumulative(project_id)

        # table (pandas) of days with usage + official cost + estimated cost
        usage_cost_table(project_id)

    else:
        print(f"Could not find project '{target_project_name}'. Exiting.")
