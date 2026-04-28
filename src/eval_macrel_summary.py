"""Compute MACREL means from sample FASTA and update summary CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
from Bio import SeqIO
from macrel.run_macrel import run_macrel

SUMMARY_FIELDS = [
    "name",
    "Probability_score_mean",
    "AMPlify_log_scaled_score_mean",
    "macrel_amp_mean",
    "macrel_hemo_mean",
    "apex_mic_mean",
]


def read_sequences_from_fasta(path):
    return [str(rec.seq) for rec in SeqIO.parse(str(path), "fasta")]


def name_from_fasta(path):
    name = path.stem
    if name.startswith("sampled_amps_"):
        name = name[len("sampled_amps_") :]
    return name


def find_sample_fastas(samples_dir, results_dir, name=None):
    if name is not None:
        return [find_sample_fasta(samples_dir, results_dir, name)]

    fasta_paths = []
    fasta_paths.extend(results_dir.rglob("*.fasta"))
    fasta_paths.extend(results_dir.rglob("*.fa"))
    fasta_paths.extend(samples_dir.glob("*.fasta"))
    fasta_paths.extend(samples_dir.glob("*.fa"))

    # Prefer FASTA files from results if the same sample exists in both places.
    by_name = {}
    for fasta_path in sorted(fasta_paths):
        sample_name = name_from_fasta(fasta_path)
        by_name.setdefault(sample_name, fasta_path)

    return [by_name[key] for key in sorted(by_name)]


def find_sample_fasta(samples_dir, results_dir, name):
    candidates = [
        results_dir / f"{name}.fasta",
        results_dir / f"{name}.fa",
        results_dir / f"sampled_amps_{name}.fasta",
        results_dir / f"sampled_amps_{name}.fa",
        samples_dir / f"{name}.fasta",
        samples_dir / f"{name}.fa",
        samples_dir / f"sampled_amps_{name}.fasta",
        samples_dir / f"sampled_amps_{name}.fa",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = sorted(results_dir.rglob(f"*{name}*.fasta"))
    if not matches:
        matches = sorted(results_dir.rglob(f"*{name}*.fa"))
    if not matches:
        matches = sorted(samples_dir.glob(f"*{name}*.fasta"))
    if not matches:
        matches = sorted(samples_dir.glob(f"*{name}*.fa"))
    if not matches:
        raise FileNotFoundError(f"No FASTA found for name={name} in {samples_dir}")
    return matches[0]


def read_summary_rows(summary_csv):
    summary_rows = []
    if summary_csv.is_file():
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("name"):
                    continue
                summary_rows.append({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return summary_rows


def upsert_summary_row(summary_rows, name, values):
    for row in summary_rows:
        if row["name"] == name:
            row.update(values)
            return

    row = {field: "" for field in SUMMARY_FIELDS}
    row["name"] = name
    row.update(values)
    summary_rows.append(row)


def write_summary_rows(summary_csv, summary_rows):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def compute_summary_for_fasta(fasta_path):
    sequences = read_sequences_from_fasta(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    macrel_amp_vals = []
    macrel_hemo_vals = []
    for seq in sequences:
        try:
            amp_prob, hemo_prob = run_macrel(seq)
            macrel_amp_vals.append(float(amp_prob))
            macrel_hemo_vals.append(float(hemo_prob))
        except Exception:
            continue

    macrel_amp_mean = float(np.mean(macrel_amp_vals)) if macrel_amp_vals else float("nan")
    macrel_hemo_mean = float(np.mean(macrel_hemo_vals)) if macrel_hemo_vals else float("nan")

    print("FASTA:", fasta_path)
    print("macrel_amp mean:", macrel_amp_mean)
    print("macrel_hemo mean:", macrel_hemo_mean)

    return {
        "macrel_amp_mean": macrel_amp_mean,
        "macrel_hemo_mean": macrel_hemo_mean,
    }


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    samples_dir = project_root / "samples"
    results_dir = samples_dir / "results"
    summary_csv = samples_dir / "results" / "summary.csv"
    fasta_paths = find_sample_fastas(samples_dir, results_dir, args.name)
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found in {results_dir} or {samples_dir}")

    summary_rows = read_summary_rows(summary_csv)
    for fasta_path in fasta_paths:
        name = args.name if args.name is not None else name_from_fasta(fasta_path)
        values = compute_summary_for_fasta(fasta_path)
        upsert_summary_row(summary_rows, name, values)

    write_summary_rows(summary_csv, summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)
