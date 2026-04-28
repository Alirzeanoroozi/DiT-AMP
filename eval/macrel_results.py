import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from src.macrel.run_macrel import run_macrel

fasta_files = {
    "ampgan": "../genbio/ampgan_sequences.fasta",
    "pepcvae": "../genbio/pepcvae_sequences.fasta",
    "hydramp": "../genbio/hydramp_sequences.fasta",
    "dit_amp": "../genbio/dit_amp_sequences.fasta",
    "ampdiffusion": "../genbio/ampdiffusion_sequences.fasta",
}

def read_fasta(path):
    records_raw = list(SeqIO.parse(path, "fasta"))
    return [str(x.seq) for x in records_raw]

def write_per_file_csv(output_csv, rows):
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "macrel_amp", "macrel_hemo"])
        writer.writerows(rows)


def write_summary_csv(path, summary_rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "n_sequences",
                "amp_mean",
                "amp_std",
                "hemo_mean",
                "hemo_std",
            ]
        )
        writer.writerows(summary_rows)


def plot_summary(path, labels, amp_means, amp_stds, hemo_means, hemo_stds):
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        amp_means,
        width,
        yerr=amp_stds,
        capsize=4,
        label="Macrel AMP prob",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        hemo_means,
        width,
        yerr=hemo_stds,
        capsize=4,
        label="Macrel Hemo prob",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Macrel mean ± std across datasets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    summary_rows = []
    labels = []
    amp_means = []
    amp_stds = []
    hemo_means = []
    hemo_stds = []

    for name, fasta_file in fasta_files.items():
        sequences = read_fasta(fasta_file)

        per_file_rows = []
        amp_vals = []
        hemo_vals = []
        skipped = 0
        for seq in sequences:
            try:
                amp_prob, hemo_prob = run_macrel(seq)
                per_file_rows.append([seq, amp_prob, hemo_prob])
                amp_vals.append(float(amp_prob))
                hemo_vals.append(float(hemo_prob))
            except Exception as e:
                # print(f"Error running Macrel for {seq}: {e}")
                skipped += 1
                continue

        output_csv = Path("macrel_results") / f"{name}_macrel.csv"
        write_per_file_csv(output_csv, per_file_rows)

        amp_vals = np.array(amp_vals, dtype=float)
        hemo_vals = np.array(hemo_vals, dtype=float)

        amp_mean = float(np.mean(amp_vals))
        amp_std = float(np.std(amp_vals))
        hemo_mean = float(np.mean(hemo_vals))
        hemo_std = float(np.std(hemo_vals))

        summary_rows.append([name, len(sequences) - skipped, amp_mean, amp_std, hemo_mean, hemo_std])
        labels.append(name)
        amp_means.append(amp_mean)
        amp_stds.append(amp_std)
        hemo_means.append(hemo_mean)
        hemo_stds.append(hemo_std)

        print(f"{name}: n={len(sequences)}")
        print(f"  AMP  mean={amp_mean:.6f}, std={amp_std:.6f}, skipped={skipped}")
        print(f"  Hemo mean={hemo_mean:.6f}, std={hemo_std:.6f}, skipped={skipped}")
        print(f"  wrote {output_csv}")

    summary_csv = Path("macrel_results") / "macrel_summary_stats.csv"
    write_summary_csv(summary_csv, summary_rows)
    print(f"Wrote summary: {summary_csv}")

    plot_path = Path("macrel_results") / "macrel_mean_std_plot.png"
    plot_summary(plot_path, labels, amp_means, amp_stds, hemo_means, hemo_stds)
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
