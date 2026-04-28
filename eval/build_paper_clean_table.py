#!/usr/bin/env python3
"""Build a paper-friendly merged metrics table.

Usage:
  python3 build_paper_clean_table.py
"""

import csv
from pathlib import Path
from typing import Dict


EVAL_DIR = Path(__file__).resolve().parent


def normalize_method(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def read_csv_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def macrel_from_per_file(macrel_dir: Path, method: str) -> Dict[str, str]:
    """Compute MACREL mean stats from per-sequence csv for a method."""
    file_path = macrel_dir / f"{method}_macrel.csv"
    if not file_path.exists():
        return {}

    rows = read_csv_rows(file_path)
    if not rows:
        return {}

    amp_vals = [float(r["macrel_amp"]) for r in rows if r.get("macrel_amp", "") != ""]
    hemo_vals = [float(r["macrel_hemo"]) for r in rows if r.get("macrel_hemo", "") != ""]
    if not amp_vals or not hemo_vals:
        return {}

    amp_mean = sum(amp_vals) / len(amp_vals)
    hemo_mean = sum(hemo_vals) / len(hemo_vals)
    return {
        "MACREL_AMP": f"{amp_mean:.4f}",
        "MACREL_Hemo": f"{hemo_mean:.4f}",
    }


def main():
    amp_scanner = EVAL_DIR / "amp-scanner_results" / "amp_scanner_summary.csv"
    amplify = EVAL_DIR / "amplify_results" / "amplify_results_summary.csv"
    apex = EVAL_DIR / "apex_results" / "apex_summary.csv"
    hydramp = EVAL_DIR / "hydramp_results" / "hydramp_summary.csv"
    macrel_summary = EVAL_DIR / "macrel_results" / "macrel_summary_stats.csv"
    macrel_dir = EVAL_DIR / "macrel_results"

    output_path = EVAL_DIR / "paper_metrics_clean_table.csv"

    methods_order = ["dit_amp", "ampgan", "pepcvae", "hydramp", "ampdiffusion"]
    rows_by_method: Dict[str, Dict[str, str]] = {m: {"method": m} for m in methods_order}

    for r in read_csv_rows(amp_scanner):
        m = normalize_method(r["model"])
        if m in rows_by_method:
            rows_by_method[m]["AMP-Scanner_AMP_Rate"] = f'{float(r["amp_rate_mean"]):.4f}'
            rows_by_method[m]["AMP-Scanner_Prob"] = f'{float(r["probability_mean"]):.4f}'

    for r in read_csv_rows(amplify):
        fname = r["file_name"]
        method = fname
        if method.startswith("amplify_"):
            method = method[len("amplify_") :]
        if method.endswith(".tsv"):
            method = method[:-4]
        m = normalize_method(method)
        if m in rows_by_method:
            rows_by_method[m]["AMPlify_AMP_Rate"] = f'{float(r["AMP_prediction_rate"]):.4f}'
            rows_by_method[m]["AMPlify_Prob"] = f'{float(r["Probability_score_mean"]):.4f}'

    for r in read_csv_rows(apex):
        m = normalize_method(r["dataset"])
        if m in rows_by_method:
            rows_by_method[m]["APEX_MIC"] = f'{float(r["apex_mic_mean"]):.4f}'

    for r in read_csv_rows(hydramp):
        m = normalize_method(r["model"])
        if m in rows_by_method:
            rows_by_method[m]["HydrAMP_AMP"] = f'{float(r["hydramp_amp_mean"]):.4f}'
            rows_by_method[m]["HydrAMP_MIC"] = f'{float(r["hydramp_mic_mean"]):.4f}'

    # Fill MACREL metrics from summary when present.
    for r in read_csv_rows(macrel_summary):
        m = normalize_method(r["dataset"])
        if m in rows_by_method:
            rows_by_method[m]["MACREL_AMP"] = f'{float(r["amp_mean"]):.4f}'
            rows_by_method[m]["MACREL_Hemo"] = f'{float(r["hemo_mean"]):.4f}'

    # Backfill missing MACREL metrics from per-file outputs (e.g., ampdiffusion).
    for m in methods_order:
        if "MACREL_AMP" not in rows_by_method[m] or "MACREL_Hemo" not in rows_by_method[m]:
            rows_by_method[m].update(macrel_from_per_file(macrel_dir, m))

    fieldnames = [
        "method",
        "AMP-Scanner_AMP_Rate",
        "AMP-Scanner_Prob",
        "AMPlify_AMP_Rate",
        "AMPlify_Prob",
        "APEX_MIC",
        "HydrAMP_AMP",
        "HydrAMP_MIC",
        "MACREL_AMP",
        "MACREL_Hemo",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in methods_order:
            row = rows_by_method[m]
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
