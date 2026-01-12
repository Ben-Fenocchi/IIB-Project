from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from webscraper import extract_article_text


#------------------ENV + OPENAI CLIENT SETUP------------------#

# Load .env from project root
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

client = OpenAI(api_key=api_key)

# Default extraction model (chosen for cost/quality tradeoff)
DEFAULT_MODEL = "gpt-5-nano"


#------------------DATA MODEL------------------#

@dataclass
class ExtractRecord:
    """
    Structured representation of one extracted disruption event.
    This is what gets serialised to JSONL/CSV downstream.
    """
    url: str
    source_title: str
    disruption_type: str
    event_date: Optional[str]
    location_name: str
    duration_hours: Optional[float]
    extras: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    method: str  # "llm-only"


#------------------LLM EXTRACTION HELPER------------------#

def _call_chatgpt_extractor(
    url: str,
    title: str,
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Calls the LLM and forces a STRICT JSON object response matching the schema.
    Returns a dict with all expected keys present (defaults filled if missing).
    """

    system_prompt = """
You are an information extraction engine for supply chain disruptions.

Your job is to read a news article and extract a REAL PHYSICAL disruption event
that could affect critical material supply chains. The disruption categories you are interested in are the following:
-Earthquakes, Floods, Cyclones/Hurricanes, Heatwaves ,Droughts, Mine Collapse, Critical Mineral Labour Strikes, Export Restrictions/ Trade Embargos,

Ignore:
- metaphorical or financial 'disruptions' (e.g. 'a storm in markets', 'a flood of criticism'),
- purely opinion/editorial content without a concrete incident,
- purely hypothetical scenarios.

You MUST output a single JSON object with a fixed schema and NOTHING else.
"""

    user_prompt = f"""
Extract information about a single **main disruption event** from the article below.

If the article does NOT contain any qualifying physical disruption, you MUST still
return a JSON object but set:
- "disruption_type": "unknown"
- "event_date": null
- "location_name": ""
- "duration_hours": null
- "extras": {{}}
- "evidence": []
- "confidence": 0.0

Examples of disruptions:
- Flooding of a mine or smelter, cyclone closing a port, drought halting mining,
  landslide or mine collapse, labour strike halting production, trade embargo or sanctions,
  power outage at a key facility, major road/rail closure disrupting shipments.

Allowed disruption_type values:
- "flood"
- "drought"
- "cyclone_hurricane"
- "extreme_heat"
- "landslide"
- "earthquake"
- "mine_collapse"
- "labour_strike"
- "trade_embargo"
- "power_outage"
- "fire"
- "road_closure"
- "rail_disruption"
- "unknown"

Field requirements:
- Stay faithful to text; do NOT fabricate mines, ports, durations, numbers.
- If unsure, use null or "" and reduce confidence.
- Output MUST be a single JSON object only.

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

    raw = completion.choices[0].message.content

    # Robust JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end + 1])
        else:
            data = {}

    # Ensure keys exist
    data.setdefault("disruption_type", "unknown")
    data.setdefault("event_date", None)
    data.setdefault("location_name", "")
    data.setdefault("duration_hours", None)
    data.setdefault("extras", {})
    data.setdefault("evidence", [])
    data.setdefault("confidence", 0.0)

    return data


#------------------SINGLE-URL ORCHESTRATOR------------------#

def extract_from_url_llm_only(url: str, model: str = DEFAULT_MODEL) -> ExtractRecord:
    """
    1) Fetch article text/title
    2) Call ChatGPT (LLM-only)
    3) Return ExtractRecord
    """
    art = extract_article_text(url)
    title = art.get("title", "") or ""
    body = art.get("text", "") or ""

    llm_out = _call_chatgpt_extractor(url, title, body, model=model)

    return ExtractRecord(
        url=url,
        source_title=title,
        disruption_type=llm_out.get("disruption_type") or "unknown",
        event_date=llm_out.get("event_date"),
        location_name=llm_out.get("location_name") or "",
        duration_hours=llm_out.get("duration_hours"),
        extras=llm_out.get("extras") or {},
        evidence=llm_out.get("evidence") or [],
        confidence=round(float(llm_out.get("confidence") or 0.0), 3),
        method="llm-only",
    )


#------------------BATCH RUNNER (for multiple URL's at once)------------------#

def run_batch(
    input_csv: str = "test_urls.csv",
    model: str = DEFAULT_MODEL,
):
    """
    Loop through all URLs in input_csv and save outputs into ./results/
    """
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

    urls = df["url"].dropna().astype(str).tolist()

    results = []
    errors = []

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    print(f"Processing {len(urls)} URLs with model={model}...\n")

    for i, url in enumerate(urls, start=1):
        url = url.strip()
        if not url:
            continue

        print(f"[{i}/{len(urls)}] {url}")
        try:
            rec = extract_from_url_llm_only(url, model=model)
            data = rec.__dict__

            results.append(data)

            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  ERROR: {e}")
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
