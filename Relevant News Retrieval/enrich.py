import csv
import time
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_DIR = Path("data/interim/gdelt_event_context_daily")
OUTPUT_SUFFIX = "_enriched.csv"
USER_AGENT = "Mozilla/5.0 (compatible; LithiumQRA/1.0)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

TIMEOUT_S = 20
MAX_RETRIES = 2
SLEEP_BETWEEN_REQ = (0.05, 0.15) 
MAX_WORKERS = 30  # Optimized for low 429 error rate

MAX_TITLE_CHARS = 300
MAX_DESC_CHARS = 800


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url: return url
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) 
             if k.lower() not in {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src"} 
             and not any(k.lower().startswith(pref) for pref in ("utm_",))]
        return urlunparse((p.scheme or "http", p.netloc.lower(), p.path, p.params, urlencode(q, doseq=True), ""))
    except: return url

def truncate(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s

def fetch_title_meta(url: str, session: requests.Session) -> Tuple[str, str, int, str]:
    last_err = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_S, allow_redirects=True)
            status = resp.status_code
            if status != 200 or not resp.text:
                if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return "", "", status, f"bad_status:{status}"
            
            soup = BeautifulSoup(resp.text, "lxml")
            title = truncate(soup.title.string if soup.title else "", MAX_TITLE_CHARS)
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            desc = truncate(desc_tag["content"] if desc_tag else "", MAX_DESC_CHARS)
            return title, desc, status, ""
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return "", "", 0, f"exception:{type(e).__name__}"
    return "", "", 0, "failed_after_retries"


CACHE_FIELDS = ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]

def append_cache_row(row: Dict[str, str]):
    write_header = not CACHE_PATH.exists()
    with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in CACHE_FIELDS})

def process_single_row(row: dict, cache: dict, session: requests.Session, existing_progress: dict):
    url = (row.get("sourceurl") or "").strip()
    url_norm = normalize_url(url)
    row["url_normalized"] = url_norm

    if url_norm in existing_progress and existing_progress[url_norm].get("http_status") == "200":
        return existing_progress[url_norm]
    if not url_norm.startswith("http"):
        row.update({"title": "", "meta_description": "", "http_status": "", "fetch_error": "non_http"})
        return row
    if url_norm in cache:
        c = cache[url_norm]
        row.update({"title": c.get("title", ""), "meta_description": c.get("meta_description", ""), 
                    "http_status": str(c.get("http_status", "")), "fetch_error": c.get("fetch_error", "")})
        return row

    time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))
    title, desc, status, err = fetch_title_meta(url_norm, session)
    res = {**row, "title": title, "meta_description": desc, "http_status": str(status), "fetch_error": err}
    append_cache_row({"url_normalized": url_norm, "title": title, "meta_description": desc, "http_status": str(status), "fetch_error": err})
    return res

# -----------------------------
# MAIN PROCESSING
# -----------------------------
def enrich_daily_file(in_path: Path, cache: Dict[str, Dict[str, str]], session: requests.Session) -> None:
    # Output path is created in the SAME folder as the input file
    out_path = in_path.with_name(in_path.stem.replace("_filtered", "") + OUTPUT_SUFFIX)
    
    existing_progress = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("url_normalized"): existing_progress[r["url_normalized"]] = r

    with open(in_path, "r", newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
        if not rows: return
        fieldnames = list(rows[0].keys()) + ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]
        fieldnames = list(dict.fromkeys(fieldnames))

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(lambda r: process_single_row(r, cache, session, existing_progress), rows),
                            total=len(rows), desc=f"Enriching {in_path.parent.name}/{in_path.name}"))

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - start_time
    theoretical_serial = total_elapsed * MAX_WORKERS
    
    print(f"\n--- PERFORMANCE REPORT: {in_path.name} ---")
    print(f"Time Taken: {total_elapsed:.2f}s | Speed Boost: {theoretical_serial/total_elapsed:.1f}x")
    print(f"Saved to: {out_path.parent}\n")

def main(target_date: str):
    global CACHE_PATH
    CACHE_PATH = Path(f"data/interim/_state/url_title_meta_cache_{target_date}.csv")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # RECURSIVE SEARCH: Finds files in Year/Month/Day folders
    files = list(BASE_DIR.rglob(f"*{target_date}*_deduped_filtered.csv"))

    if not files:
        print(f"No filtered files found for {target_date} in {BASE_DIR}")
        return

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = {r["url_normalized"]: r for r in csv.DictReader(f) if r.get("url_normalized")}

    with requests.Session() as session:
        session.headers.update(HEADERS)
        for f in files:
            enrich_daily_file(f, cache, session)

if __name__ == "__main__":
    day = input("Enter date (YYYYMMDD): ").strip()
    main(day)