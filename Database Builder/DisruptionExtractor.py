from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from webscraper import extract_article_text


# ------------------ ENV + OPENAI CLIENT SETUP ------------------ #

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

client = OpenAI(api_key=api_key)

# Single-pass model for structured extraction
DEFAULT_MODEL = "gpt-5-mini"

# Concurrency
MAX_WORKERS = 20  # up to 20 workers


# ------------------ DATA MODEL ------------------ #

@dataclass
class ExtractRecord:
    """
    Structured representation of one extracted disruption event.
    This is what gets serialised to JSONL/CSV downstream.
    """
    url: str
    source_title: str
    disruption_type: str
    event_date: Optional[str]        # extracted event date (LLM)
    publish_date: Optional[str]      # article publication date (scraper)
    location_name: str
    duration_hours: Optional[float]
    extras: Dict[str, Any]
    confidence: float


# ------------------ DATE NORMALISATION ------------------ #
def _normalise_date(value: Any, *, date_only: bool) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    try:
        ts = pd.to_datetime(s, utc=True, errors="raise")
    except Exception:
        return None

    if date_only:
        return ts.date().isoformat()

    return ts.isoformat()


# ------------------ LLM EXTRACTION HELPER ------------------ #

def _call_chatgpt_extractor(
    url: str,
    title: str,
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
) -> Dict[str, Any]:

    system_prompt = """\
You are an information extraction engine for supply chain disruptions.

Your job is to read a news article and extract a REAL physical or policy disruption event.

You MUST output a single JSON object and NOTHING else.
"""

    user_prompt = f"""\
Extract information about a single main supply chain disruption event from the article.

If there is no qualifying disruption, return:
{{
  "disruption_type": "unknown",
  "event_date": null,
  "location_name": "",
  "duration_hours": null,
  "extras": {{}},
  "confidence": 0.0
}}

Allowed disruption_type:
flood, drought, cyclone_hurricane, extreme_heat, landslide, earthquake,
mine_collapse, mine_accident, labour_strike, trade_embargo, tariffs, unknown

Schema:
{{
  "disruption_type": "...",
  "event_date": "YYYY-MM-DD" or null,
  "location_name": "...",
  "duration_hours": number or null,
  "extras": {{ indicator_name: value }},
  "confidence": 0.0
}}

IMPORTANT:
- Our threshold for classing a disruption is a confidence of 0.6
- Ignore metaphorical disruptions e.g. "a flood of criticism".
- "extras" is the ONLY place where indicator values may appear.
- Only include indicators if explicitly mentioned in the article.
- Do not infer, estimate, or fabricate indicator values.
- Do not include indicators not listed below.
- If no indicators are mentioned, extras must be {{}}.

Indicators by disruption type:

flood:
- rainfall_anomaly
- rainfall_intensity
- river_discharge
- soil_saturation
- reservoir_release

drought:
- rainfall_deviation
- reservoir_level
- temperature_anomaly
- water_restrictions

cyclone_hurricane:
- sea_surface_temp_anomaly
- storm_category
- wind_speed

extreme_heat:
- temperature_anomaly
- days_above_35c
- wet_bulb_temp
- power_grid_stress

landslide:
- rainfall_intensity
- soil_moisture
- slope_stability
- deforestation_activity

earthquake:
- seismic_event_count
- max_magnitude
- foreshock_activity

mine_collapse:
- tailings_risk
- inspection_failure
- seepage_or_cracks

mine_accident:
- fatalities
- injuries
- equipment_failure

labour_strike:
- unionization_rate
- inflation_rate
- wage_growth
- strike_vote_result

trade_embargo:
- sanction_count
- trade_restrictiveness_index

tariffs:
- tariff_rate
- affected_products_count
- affected_trade_value

Rules:
- Stay faithful to the text.
- If unsure, leave fields null/empty and lower confidence.
- Output JSON only.

Now process this article:

URL:
{url}

TITLE:
{title}

TEXT:
{text}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        timeout=timeout,
    )

    raw = completion.choices[0].message.content or ""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end + 1])
        else:
            data = {}

    data.setdefault("disruption_type", "unknown")
    data.setdefault("event_date", None)
    data.setdefault("location_name", "")
    data.setdefault("duration_hours", None)
    data.setdefault("extras", {})
    data.setdefault("confidence", 0.0)

    return data


# ------------------ SINGLE-URL ORCHESTRATOR ------------------ #

def extract_from_url_llm_single_pass(url: str, model: str = DEFAULT_MODEL) -> ExtractRecord:
    art = extract_article_text(url)
    title = art.get("title", "") or ""
    body = art.get("text", "") or ""
    publish_date = art.get("publish_date")

    llm_out = _call_chatgpt_extractor(url, title, body, model=model)

    publish_date_norm = _normalise_date(publish_date, date_only=False)
    event_date_norm = _normalise_date(llm_out.get("event_date"), date_only=True)

    return ExtractRecord(
        url=url,
        source_title=title,
        disruption_type=llm_out.get("disruption_type") or "unknown",
        event_date=event_date_norm,
        publish_date=publish_date_norm,
        location_name=llm_out.get("location_name") or "",
        duration_hours=llm_out.get("duration_hours"),
        extras=llm_out.get("extras") or {},
        confidence=round(float(llm_out.get("confidence") or 0.0), 3),
    )


# ------------------ BATCH RUNNER (CONCURRENT) ------------------ #

def run_batch(
    input_csv: str = "test_urls.csv",
    model: str = DEFAULT_MODEL,
    max_workers: int = MAX_WORKERS,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_jsonl = os.path.join(results_dir, "extractions.jsonl")
    out_csv = os.path.join(results_dir, "extractions.csv")
    out_errors = os.path.join(results_dir, "errors.csv")

    input_path = os.path.join(base_dir, input_csv)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_csv} in {base_dir}")

    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "url" not in df.columns:
        raise ValueError("CSV must have a header named 'url'")

    urls = [u.strip() for u in df["url"].dropna().astype(str).tolist() if u.strip()]
    total = len(urls)

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    write_lock = threading.Lock()

    def _worker(u: str) -> Dict[str, Any]:
        rec = extract_from_url_llm_single_pass(u, model=model)
        return rec.__dict__

    print(f"Processing {total} URLs with model={model} using up to {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_worker, u): u for u in urls}

        for fut in tqdm(as_completed(future_to_url), total=total, desc="Extracting"):
            url = future_to_url[fut]
            try:
                data = fut.result()
                results.append(data)

                with write_lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                errors.append({"url": url, "error": str(e)})

    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\nSaved outputs to:\n  {out_jsonl}\n  {out_csv}")

    if errors:
        pd.DataFrame(errors).to_csv(out_errors, index=False, encoding="utf-8")
        print(f"Some URLs failed — see:\n  {out_errors}")
    else:
        print("No errors recorded.")


if __name__ == "__main__":
    run_batch()