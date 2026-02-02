from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

import pandas as pd
import requests


Directory = Literal["indicator", "ref_area"]
Fmt = Literal[".parquet", ".csv.gz", ".csv", ".feather", ".json", ".tsv", ".dta"]


@dataclass
class ILOSTATConfig:
    """
    Client config for ILOSTAT Bulk downloads.
    Bulk repository base is the rplumber endpoint. 
    """
    bulk_base_url: str = "https://rplumber.ilo.org/data"
    raw_dir: str = "External databases/ILOSTAT/data/raw"
    derived_dir: str = "External databases/ILOSTAT/data/derived"
    timeout_s: int = 180

    # polite client + resilience
    max_attempts_per_format: int = 5     # retries per format
    base_sleep_s: float = 1.0            # exponential backoff base
    jitter_s: float = 0.2                # small random-ish delay (fixed jitter to keep deterministic)
    prefer_formats: tuple[Fmt, ...] = (".parquet", ".csv.gz", ".csv")

    # output saving
    save_parquet: bool = True
    save_csv: bool = True


class ILOSTATClient:
    """
    Robust downloader/reader for ILOSTAT bulk tables.

    Key ideas:
      - Use bulk endpoint: {base}/{directory}/?format={fmt}&id={table_id} 
      - Retry with exponential backoff on transient 5xx errors (common)
      - Prefer parquet when available (often more reliable/faster than csv.gz)
    """

    def __init__(self, cfg: Optional[ILOSTATConfig] = None):
        self.cfg = cfg or ILOSTATConfig()
        self.raw_dir = Path(self.cfg.raw_dir)
        self.derived_dir = Path(self.cfg.derived_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)

        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "lion-ilostat/1.0"})

    # -----------------------------
    # URL building
    # -----------------------------

    def _url(self, table_id: str, directory: Directory, fmt: str) -> str:
        return f"{self.cfg.bulk_base_url}/{directory}/?format={fmt}&id={table_id}"

    # -----------------------------
    # Download (robust)
    # -----------------------------

    def _download_stream(self, url: str, out_path: Path) -> None:
        with self.sess.get(url, stream=True, timeout=self.cfg.timeout_s) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    def download_table(
        self,
        table_id: str,
        *,
        directory: Directory,
        prefer: Optional[Iterable[Fmt]] = None,
        force: bool = False,
    ) -> Path:
        """
        Download a table file to raw_dir and return local path.
        Tries multiple formats with retries/backoff.

        Example:
          download_table("EMP_DWAP_NOC_RT_A", directory="indicator")
        """
        prefer_formats = list(prefer) if prefer is not None else list(self.cfg.prefer_formats)

        last_err: Optional[Exception] = None

        for fmt in prefer_formats:
            out_path = self.raw_dir / f"{table_id}{fmt}"
            if out_path.exists() and not force:
                return out_path

            # Some servers accept "csv.gz" (no leading dot) even though docs show ".csv.gz"
            fmt_variants = [fmt]
            if fmt == ".csv.gz":
                fmt_variants.append("csv.gz")

            for fmt_variant in fmt_variants:
                url = self._url(table_id, directory, fmt_variant)

                for attempt in range(1, self.cfg.max_attempts_per_format + 1):
                    try:
                        self._download_stream(url, out_path)
                        # tiny jitter to reduce bursting
                        time.sleep(self.cfg.jitter_s)
                        return out_path

                    except requests.HTTPError as e:
                        last_err = e
                        status = getattr(e.response, "status_code", None)

                        # 4xx usually means wrong ID/format/permissions — don't keep retrying this format variant
                        if status is not None and 400 <= status < 500:
                            break

                        # 5xx -> retry with exponential backoff
                        sleep_s = self.cfg.base_sleep_s * (2 ** (attempt - 1))
                        time.sleep(sleep_s + self.cfg.jitter_s)

                    except requests.RequestException as e:
                        last_err = e
                        sleep_s = self.cfg.base_sleep_s * (2 ** (attempt - 1))
                        time.sleep(sleep_s + self.cfg.jitter_s)

        raise RuntimeError(
            f"Failed to download ILOSTAT table '{table_id}' (directory={directory}). "
            f"Tried formats={prefer_formats}. Last error: {last_err}"
        )

    # -----------------------------
    # Read into pandas
    # -----------------------------

    def read_table(
        self,
        table_id: str,
        *,
        directory: Directory,
        prefer: Optional[Iterable[Fmt]] = None,
        force_download: bool = False,
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """
        Download (if needed) and read a table into a DataFrame.

        For parquet:
          - uses pd.read_parquet
        For csv.gz / csv:
          - uses pd.read_csv
        """
        path = self.download_table(
            table_id,
            directory=directory,
            prefer=prefer,
            force=force_download,
        )

        suffix = "".join(path.suffixes)  # handles ".csv.gz" as two suffixes
        if suffix.endswith(".parquet"):
            return pd.read_parquet(path)
        elif suffix.endswith(".csv.gz"):
            return pd.read_csv(path, compression="gzip", low_memory=low_memory)
        elif suffix.endswith(".csv"):
            return pd.read_csv(path, low_memory=low_memory)
        else:
            raise ValueError(f"Unsupported file type for reading: {path}")

    # -----------------------------
    # Save derived outputs
    # -----------------------------

    def save_outputs(self, df: pd.DataFrame, name: str) -> Dict[str, Path]:
        """
        Save a derived dataset to derived_dir as parquet and/or csv.
        Returns dict of written paths.
        """
        written: Dict[str, Path] = {}

        if self.cfg.save_parquet:
            p = self.derived_dir / f"{name}.parquet"
            df.to_parquet(p, index=False)
            written["parquet"] = p

        if self.cfg.save_csv:
            c = self.derived_dir / f"{name}.csv"
            df.to_csv(c, index=False)
            written["csv"] = c

        return written

    # -----------------------------
    # Convenience: list cached files
    # -----------------------------

    def list_raw_cache(self) -> list[Path]:
        return sorted(self.raw_dir.glob("*"))

    def list_derived(self) -> list[Path]:
        return sorted(self.derived_dir.glob("*"))
