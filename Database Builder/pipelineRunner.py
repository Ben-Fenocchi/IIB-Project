"""
Master post-processing pipeline.

Flow:
1) Load raw extractions from results/
2) Run consolidation
3) Save consolidated files
4) Run debugger + metrics
5) Display consolidated extractions
"""

from pathlib import Path
import sys

# ---- Project paths ---- #

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# ---- Import helper modules ---- #

from helper_scripts.consolidateExtractions import (load_extractions,run_consolidation)
from helper_scripts.debuggerAndMetrics import run_debugger_and_metrics
from helper_scripts.DisplayExtractionsPandas import run_display_extractions
from helper_scripts.plotDisruptions import run_plots


# ------------------ MAIN PIPELINE ------------------ #

def run_pipeline(input_filename: str):

    input_path = RESULTS_DIR / input_filename

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    print(f"\nRunning pipeline for: {input_filename}\n")

    # ---- 1) Load raw extractions ---- #
    df_before = load_extractions(input_path)

    # ---- 2) Consolidate ---- #
    df_after = run_consolidation(input_path)

    # ---- 3) Debug + Metrics ---- #
    run_debugger_and_metrics(df_before, df_after)

    # ---- 4) Display ---- #
    run_display_extractions(df_after, df_before=df_before)

    #-----5) plots ----#
    run_plots(df_after, project_root = BASE_DIR)


    print("\nPipeline complete.\n")


if __name__ == "__main__":
    run_pipeline("weekly_extractions_202601.jsonl")