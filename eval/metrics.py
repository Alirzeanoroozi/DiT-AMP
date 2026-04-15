#!/usr/bin/env python3

import csv
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import random

from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
import numpy as np

fasta_files = {
    "ampgan": "../genbio/ampgan_sequences.fasta",
    "pepcvae": "../genbio/pepcvae_sequences.fasta",
    "hydramp": "../genbio/hydramp_sequences.fasta",
    "ampdiffusion": "../genbio/ampdiffusion_sequences.fasta",
    "dit_amp": "../genbio/dit_amp_sequences.fasta",
}

REFERENCE_FASTA = "../data/data.fasta"
THETA_DEG = 100.0
MAX_DIVERSITY_PAIRS = 20000


H_SCALE = {
    "A": 0.25, "R": -1.80, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54,
}

HX_SCALE = {
    "A": 0.00, "R": 0.21, "N": 0.65, "D": 0.69, "C": 0.68,
    "Q": 0.39, "E": 0.40, "G": 1.00, "H": 0.61, "I": 0.41,
    "L": 0.21, "K": 0.26, "M": 0.24, "F": 0.54, "P": 3.16,
    "S": 0.50, "T": 0.66, "W": 0.49, "Y": 0.53, "V": 0.61,
}


def read_fasta(path):
    sequences = []
    current = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return sequences


def shannon_entropy(sequence):
    if len(sequence) == 0:
        return 0.0
    freq = {}
    for aa in sequence:
        freq[aa] = freq.get(aa, 0) + 1
    entropy = 0.0
    length = len(sequence)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def compute_entropy_stats(sequences):
    values = [shannon_entropy(seq) for seq in sequences]
    avg = sum(values) / len(values) if values else 0.0
    std = float(np.std(np.array(values, dtype=float))) if values else 0.0
    return avg, std


def get_kmers(sequence, k):
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def get_kmer_set(sequences, k):
    kmers = set()
    for seq in sequences:
        kmers.update(get_kmers(seq, k))
    return kmers


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def compute_jaccard_kmer(gen_seqs, ref_seqs, k):
    gen_kmers = get_kmer_set(gen_seqs, k)
    ref_kmers = get_kmer_set(ref_seqs, k)
    return jaccard_similarity(gen_kmers, ref_kmers)


@lru_cache(maxsize=1)
def _pairwise_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    return aligner


def pairwise_alignment_identity_ratio(a, b):
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    aln = _pairwise_aligner().align(a, b)[0]
    identities = aln.counts().identities
    return identities / min_len


def compute_diversity(sequences, max_pairs=MAX_DIVERSITY_PAIRS, seed=42):
    n = len(sequences)
    if n < 2:
        return 0.0, "exact"

    total_pairs = n * (n - 1) // 2
    rng = random.Random(seed)

    if total_pairs <= max_pairs:
        pairs_iter = combinations(sequences, 2)
        mode = "exact"
        pair_count = total_pairs
    else:
        mode = f"sampled_{max_pairs}"
        pair_count = max_pairs

        def sampled_pairs():
            for _ in range(max_pairs):
                i, j = rng.sample(range(n), 2)
                yield sequences[i], sequences[j]

        pairs_iter = sampled_pairs()

    scores = [pairwise_alignment_identity_ratio(s1, s2) for s1, s2 in pairs_iter]
    if not scores:
        return 0.0, mode
    return 100.0 * sum(scores) / pair_count, mode


def compute_uniqueness(sequences):
    n = len(sequences)
    if n == 0:
        return 0.0
    return 100.0 * len(set(sequences)) / n


def compute_novelty(sequences, reference_sequences):
    n = len(sequences)
    if n == 0:
        return 0.0
    ref_set = set(reference_sequences)
    novel_count = sum(1 for s in sequences if s not in ref_set)
    return 100.0 * novel_count / n


def sequence_fitness(seq, theta_deg=THETA_DEG):
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    theta = theta_deg * math.pi / 180.0
    x = 0.0
    y = 0.0
    denom = 0.0
    for i, aa in enumerate(seq, start=1):
        if aa not in H_SCALE or aa not in HX_SCALE:
            return 0.0
        h = H_SCALE[aa]
        hx = HX_SCALE[aa]
        x += h * math.cos(i * theta)
        y += h * math.sin(i * theta)
        denom += math.exp(hx)
    numerator = math.sqrt(x * x + y * y)
    if denom == 0:
        return 0.0
    return numerator / denom


def compute_fitness_score(sequences, theta_deg=THETA_DEG):
    n = len(sequences)
    if n == 0:
        return 0.0
    values = [sequence_fitness(seq, theta_deg=theta_deg) for seq in sequences]
    return sum(values) / n


def save_summary_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "n_sequences",
                "diversity",
                "diversity_mode",
                "uniqueness",
                "novelty",
                "fitness",
                "entropy_mean",
                "entropy_std",
                "js_3",
                "js_6",
            ]
        )
        writer.writerows(rows)


def plot_metric(path: Path, labels, values, ylabel, title, color, ylim=None):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    out_dir = Path("metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_seqs = read_fasta(REFERENCE_FASTA)
    if not ref_seqs:
        print(f"No reference/train sequences found in {REFERENCE_FASTA}")
        return

    summary_rows = []
    labels = []
    series = {
        "diversity": [],
        "uniqueness": [],
        "novelty": [],
        "fitness": [],
        "entropy_mean": [],
        "js_3": [],
        "js_6": [],
    }

    for name, fasta_path in fasta_files.items():
        gen_seqs = read_fasta(fasta_path)
        if not gen_seqs:
            print(f"{name}: no sequences found, skipping")
            continue

        diversity, diversity_mode = compute_diversity(gen_seqs)
        uniqueness = compute_uniqueness(gen_seqs)
        novelty = compute_novelty(gen_seqs, ref_seqs)
        fitness = compute_fitness_score(gen_seqs)
        entropy_mean, entropy_std = compute_entropy_stats(gen_seqs)
        js_3 = compute_jaccard_kmer(gen_seqs, ref_seqs, k=3)
        js_6 = compute_jaccard_kmer(gen_seqs, ref_seqs, k=6)

        summary_rows.append(
            [
                name,
                len(gen_seqs),
                diversity,
                diversity_mode,
                uniqueness,
                novelty,
                fitness,
                entropy_mean,
                entropy_std,
                js_3,
                js_6,
            ]
        )
        labels.append(name)
        series["diversity"].append(diversity)
        series["uniqueness"].append(uniqueness)
        series["novelty"].append(novelty)
        series["fitness"].append(fitness)
        series["entropy_mean"].append(entropy_mean)
        series["js_3"].append(js_3)
        series["js_6"].append(js_6)

        print(f"{name}: n={len(gen_seqs)}")
        print(
            f"  diversity={diversity:.6f} ({diversity_mode}), uniqueness={uniqueness:.6f}, novelty={novelty:.6f}"
        )
        print(f"  fitness={fitness:.6f}, entropy_mean={entropy_mean:.6f}, js_3={js_3:.6f}, js_6={js_6:.6f}")

    if not summary_rows:
        print("No generated datasets were processed.")
        return

    summary_path = out_dir / "metrics_summary.csv"
    save_summary_csv(summary_path, summary_rows)
    print(f"Saved {summary_path}")

    plot_metric(out_dir / "diversity_plot.png", labels, series["diversity"], "Diversity", "Diversity by Dataset", "#72B7B2", ylim=(0.0, 100.0))
    plot_metric(out_dir / "uniqueness_plot.png", labels, series["uniqueness"], "Uniqueness (%)", "Uniqueness by Dataset", "#E45756", ylim=(0.0, 100.0))
    plot_metric(out_dir / "novelty_plot.png", labels, series["novelty"], "Novelty (%)", "Novelty vs Train Data", "#B279A2", ylim=(0.0, 100.0))
    plot_metric(out_dir / "fitness_plot.png", labels, series["fitness"], "Fitness Score", "Fitness by Dataset", "#9D755D")
    plot_metric(out_dir / "entropy_plot.png", labels, series["entropy_mean"], "Shannon Entropy (mean)", "Mean Shannon Entropy by Dataset", "#54A24B")
    plot_metric(out_dir / "js_3_plot.png", labels, series["js_3"], "Jaccard Similarity (3-mer)", "3-mer Jaccard Similarity vs Train Data", "#4C78A8", ylim=(0.0, 1.05))
    plot_metric(out_dir / "js_6_plot.png", labels, series["js_6"], "Jaccard Similarity (6-mer)", "6-mer Jaccard Similarity vs Train Data", "#F58518", ylim=(0.0, 1.05))
    print(f"Saved metric plots under {out_dir}")


if __name__ == "__main__":
    main()