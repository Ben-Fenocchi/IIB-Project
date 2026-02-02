"""
Debugger for publish-date extraction.

IMPORTANT:
This debugger ONLY evaluates URLs that were classified as
KNOWN disruptions (disruption_type != 'unknown') in extractions.csv.

This makes results directly comparable to the pipeline coverage stats.
"""

from __future__ import annotations

import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import trafilatura
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional Newspaper3k
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


# ------------------ CONSTANTS ------------------ #

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

META_DATE_TAGS = [
    ("property", "article:published_time"),
    ("property", "article:modified_time"),
    ("name", "pubdate"),
    ("name", "publish-date"),
    ("name", "publication_date"),
    ("name", "date"),
    ("itemprop", "datePublished"),
    ("itemprop", "dateModified"),
]


# ------------------ DATE EXTRACTION METHODS ------------------ #

def extract_from_meta(soup: BeautifulSoup) -> Optional[str]:
    for attr, key in META_DATE_TAGS:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            try:
                return dateparser.parse(tag["content"]).isoformat()
            except Exception:
                pass
    return None


def extract_from_time_tag(soup: BeautifulSoup) -> Optional[str]:
    time_tag = soup.find("time", datetime=True)
    if time_tag:
        try:
            return dateparser.parse(time_tag["datetime"]).isoformat()
        except Exception:
            pass
    return None


def extract_from_trafilatura(html: str) -> Optional[str]:
    meta = trafilatura.bare_extraction(html)
    if isinstance(meta, dict):
        raw = (
            meta.get("date")
            or meta.get("published")
            or meta.get("datePublished")
        )
        if raw:
            try:
                return dateparser.parse(raw).isoformat()
            except Exception:
                pass
    return None


def extract_from_newspaper(url: str) -> Optional[str]:
    if not _HAS_NEWSPAPER:
        return None
    try:
        art = _NPArticle(url)
        art.download()
        art.parse()
        if art.publish_date:
            return art.publish_date.isoformat()
    except Exception:
        pass
    return None


# ------------------ SINGLE URL WORKER ------------------ #

def process_url(url: str) -> Tuple[bool, Optional[str]]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        if extract_from_meta(soup):
            return True, "meta_tags"

        if extract_from_time_tag(soup):
            return True, "time_tag"

        if extract_from_trafilatura(html):
            return True, "trafilatura"

        if extract_from_newspaper(url):
            return True, "newspaper3k"

        return False, None

    except Exception:
        return False, None


# ------------------ MAIN DRIVER ------------------ #

def debug_publish_dates_known_only(csv_path: str, max_workers: int = 20):
    df = pd.read_csv(csv_path)

    if not {"url", "disruption_type"}.issubset(df.columns):
        raise ValueError("CSV must contain 'url' and 'disruption_type' columns")

    # ---- FILTER TO KNOWN DISRUPTIONS ----
    known = df[df["disruption_type"] != "unknown"].copy()
    urls = known["url"].dropna().astype(str).tolist()

    print("\n===================================================")
    print("DEBUG MODE: KNOWN DISRUPTIONS ONLY")
    print("---------------------------------------------------")
    print("This debugger will ONLY evaluate URLs where")
    print("disruption_type != 'unknown' in extractions.csv.")
    print("This makes the success rate directly comparable")
    print("to the pipeline coverage diagnostics.")
    print("---------------------------------------------------")
    print(f"Total URLs in CSV           : {len(df)}")
    print(f"Known disruption URLs used : {len(urls)}")
    print("===================================================\n")

    total = len(urls)
    processed = 0
    total_success = 0
    method_hits = Counter()

    def print_status():
        rate = total_success / processed if processed else 0.0
        methods = ", ".join(f"{k}:{v}" for k, v in method_hits.items())
        sys.stdout.write(
            f"\rProcessed {processed}/{total} | "
            f"Publish-date success: {total_success} ({rate:.1%}) | "
            f"{methods}"
        )
        sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_url, u) for u in urls]

        for future in as_completed(futures):
            processed += 1
            success, method = future.result()

            if success:
                total_success += 1
                method_hits[method] += 1

            print_status()

    print("\n\n================ FINAL SUMMARY =================")
    print(f"Known disruption URLs tested : {total}")
    print(f"Publish dates extracted     : {total_success}")
    print(f"Final success rate          : {total_success / total:.1%}\n")

    print("Breakdown by extraction method:")
    for k, v in method_hits.items():
        print(f"  - {k:15s}: {v}")


# ------------------ ENTRY POINT ------------------ #

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "results" / "extractions.csv"

    debug_publish_dates_known_only(
        csv_path=str(csv_path),
        max_workers=20,
    )
