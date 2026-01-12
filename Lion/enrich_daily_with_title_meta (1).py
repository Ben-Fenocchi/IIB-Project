"""
pipeline/03_enrich_daily_with_title_meta.py

ONE-DAY-AT-A-TIME enrichment.

For the chosen day (ONLY_DAY), this script:
1) Finds the matching daily *_deduped.csv file(s)
2) Fetches HTML for each URL and extracts:
     - <title>
     - meta description (fallback to og:description)
3) Writes an enriched daily CSV next to the input:
     *_deduped_enriched.csv
4) Writes a DAY-SPECIFIC cache CSV (title/meta lookup) whose *filename includes the day*:
     data/interim/_state/url_title_meta_cache_{ONLY_DAY}.csv

Progress bars:
- Overall progress across files
- Per-file progress across URLs

Requirements:
  pip install requests beautifulsoup4 lxml tqdm
"""

import csv
import time
import random
from pathlib import Path
from typing import Dict, Tuple, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("data/interim/gdelt_event_context_daily")
OUTPUT_SUFFIX = "_enriched.csv"

# Run one day at a time (YYYYMMDD). Change this per run.
ONLY_DAY = "20240215"

# DAY-SPECIFIC CACHE FILE (title/meta) -> filename includes the day
CACHE_PATH = Path(f"data/interim/_state/url_title_meta_cache_{ONLY_DAY}.csv")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (compatible; LithiumQRA/1.0)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

TIMEOUT_S = 20
MAX_RETRIES = 2
SLEEP_BETWEEN_REQ = (0.15, 0.45)

MAX_TITLE_CHARS = 300
MAX_DESC_CHARS = 800


# -----------------------------
# URL NORMALISATION
# -----------------------------
TRACKING_KEYS_EXACT = {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src"}
TRACKING_PREFIXES = ("utm_",)

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    try:
        p = urlparse(url)
        scheme = (p.scheme or "http").lower()
        netloc = (p.netloc or "").lower()
        path = p.path or ""
        fragment = ""  # drop fragments

        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKING_KEYS_EXACT:
                continue
            if any(kl.startswith(pref) for pref in TRACKING_PREFIXES):
                continue
            q.append((k, v))
        query = urlencode(q, doseq=True)

        return urlunparse((scheme, netloc, path, p.params or "", query, fragment))
    except Exception:
        return url


def truncate(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s


# -----------------------------
# DAY CACHE (CSV)
# -----------------------------
CACHE_FIELDS = ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]

def load_cache() -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    if not CACHE_PATH.exists():
        return cache
    with open(CACHE_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            u = (r.get("url_normalized") or "").strip()
            if u:
                cache[u] = r
    return cache


def append_cache_row(row: Dict[str, str]) -> None:
    write_header = not CACHE_PATH.exists()
    with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CACHE_FIELDS})


# -----------------------------
# FETCH TITLE + META
# -----------------------------
def fetch_title_meta(url: str) -> Tuple[str, str, int, str]:
    """
    Returns (title, meta_description, http_status, fetch_error).
    """
    last_err = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S, allow_redirects=True)
            status = resp.status_code

            if status != 200 or not resp.text:
                last_err = f"bad_status_or_empty:{status}"
                if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(0.8 * (attempt + 1) + random.random() * 0.4)
                    continue
                return "", "", status, last_err

            soup = BeautifulSoup(resp.text, "lxml")

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            desc = ""
            tag = soup.find("meta", attrs={"name": "description"})
            if tag and tag.get("content"):
                desc = tag["content"].strip()
            else:
                og = soup.find("meta", attrs={"property": "og:description"})
                if og and og.get("content"):
                    desc = og["content"].strip()

            return truncate(title, MAX_TITLE_CHARS), truncate(desc, MAX_DESC_CHARS), status, ""

        except Exception as e:
            last_err = f"exception:{type(e).__name__}"
            if attempt < MAX_RETRIES:
                time.sleep(0.8 * (attempt + 1) + random.random() * 0.4)
                continue
            return "", "", 0, last_err

    return "", "", 0, last_err


# -----------------------------
# FILE PROCESSING
# -----------------------------
def iter_deduped_daily_files():
    # Only process files for ONLY_DAY
    for f in sorted(INPUT_DIR.rglob("*_deduped.csv")):
        if ONLY_DAY and ONLY_DAY not in f.name:
            continue
        yield f


def enrich_daily_file(in_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    out_path = in_path.with_name(in_path.stem + OUTPUT_SUFFIX)

    # If already enriched, skip
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"SKIP (exists): {out_path}")
        return

    # Read all rows (so tqdm can show accurate total)
    with open(in_path, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            print(f"EMPTY/NO HEADER: {in_path}")
            return
        rows = list(reader)
        base_fieldnames = list(reader.fieldnames)

    # Ensure we have these columns appended
    new_cols = ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]
    out_fieldnames = base_fieldnames[:]
    for c in new_cols:
        if c not in out_fieldnames:
            out_fieldnames.append(c)

    rows_out = []

    for row in tqdm(rows, desc=in_path.name, unit="url"):
        url = (row.get("sourceurl") or "").strip()
        url_norm = normalize_url(url)
        row["url_normalized"] = url_norm

        if not url_norm.startswith("http"):
            row["title"] = ""
            row["meta_description"] = ""
            row["http_status"] = ""
            row["fetch_error"] = "non_http"
            rows_out.append(row)
            continue

        # Cache hit (day-specific cache)
        if url_norm in cache:
            c = cache[url_norm]
            row["title"] = c.get("title", "")
            row["meta_description"] = c.get("meta_description", "")
            row["http_status"] = str(c.get("http_status", ""))
            row["fetch_error"] = c.get("fetch_error", "")
            rows_out.append(row)
            continue

        # Polite jitter
        time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))

        title, desc, status, err = fetch_title_meta(url_norm)

        row["title"] = title
        row["meta_description"] = desc
        row["http_status"] = str(status)
        row["fetch_error"] = err
        rows_out.append(row)

        # Write-through cache (CSV)
        entry = {
            "url_normalized": url_norm,
            "title": title,
            "meta_description": desc,
            "http_status": str(status),
            "fetch_error": err,
        }
        cache[url_norm] = entry
        append_cache_row(entry)

    # Write output file
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows_out:
            for c in out_fieldnames:
                if c not in r:
                    r[c] = ""
            writer.writerow(r)

    print(f"WROTE: {out_path}")
    print(f"WROTE/UPDATED DAY CACHE: {CACHE_PATH}")


def main():
    cache = load_cache()
    print(f"Loaded day cache entries: {len(cache):,}")
    print(f"ONLY_DAY = {ONLY_DAY}")
    print(f"Day cache file = {CACHE_PATH}")

    files = list(iter_deduped_daily_files())
    if not files:
        print(f"No *_deduped.csv files found for day {ONLY_DAY} under {INPUT_DIR}")
        return

    for f in tqdm(files, desc="Daily files", unit="file"):
        enrich_daily_file(f, cache)

    print("Done.")


if __name__ == "__main__":
    main()
