# run with amplify env!
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from apex.APEX_predict import predict

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


def compute_summary_for_fasta(fasta_path, name, samples_dir):
    result_folder = samples_dir / "results" / name
    tsv_files = sorted(result_folder.glob("*.tsv"))
    if tsv_files:
        tsv_file = tsv_files[0]
        df = pd.read_csv(tsv_file, sep="\t")
        means = df[["Probability_score", "AMPlify_log_scaled_score"]].mean()
        prob_mean = float(means["Probability_score"])
        log_scaled_mean = float(means["AMPlify_log_scaled_score"])
    else:
        print(f"No .tsv file found in {result_folder}; leaving AMPlify fields blank")
        prob_mean = ""
        log_scaled_mean = ""

    sequences = read_sequences_from_fasta(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    apex_vals = []

    for seq in sequences:
        try:
            apex_vals.append(predict(seq))
        except Exception as e:
            print(f"Error predicting APEX for sequence: {seq}: {e}")
            continue

    apex_mic_mean = float(np.mean(apex_vals)) if apex_vals else float("nan")

    print("Probability_score mean:", prob_mean)
    print("AMPlify_log_scaled_score mean:", log_scaled_mean)
    print("FASTA:", fasta_path)
    print("apex_mic mean:", apex_mic_mean)

    return {
        "Probability_score_mean": prob_mean,
        "AMPlify_log_scaled_score_mean": log_scaled_mean,
        "apex_mic_mean": apex_mic_mean,
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
        values = compute_summary_for_fasta(fasta_path, name, samples_dir)
        upsert_summary_row(summary_rows, name, values)

    write_summary_rows(summary_csv, summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)