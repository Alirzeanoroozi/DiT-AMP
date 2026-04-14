#!/usr/bin/env python3

import math
import argparse
from functools import lru_cache
from itertools import combinations

from Bio.Align import PairwiseAligner


# -----------------------------
# Amino-acid scales from table
# -----------------------------
H_SCALE = {
    "A": 0.25, "R": -1.80, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54
}

HX_SCALE = {
    "A": 0.00, "R": 0.21, "N": 0.65, "D": 0.69, "C": 0.68,
    "Q": 0.39, "E": 0.40, "G": 1.00, "H": 0.61, "I": 0.41,
    "L": 0.21, "K": 0.26, "M": 0.24, "F": 0.54, "P": 3.16,
    "S": 0.50, "T": 0.66, "W": 0.49, "Y": 0.53, "V": 0.61
}


# -----------------------------
# FASTA reader
# -----------------------------
def read_fasta(path):
    """
    Read sequences from a FASTA file.
    Returns a list of sequences as uppercase strings.
    """
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


# -----------------------------
# Diversity
# -----------------------------
@lru_cache(maxsize=1)
def _pairwise_aligner():
    """Global protein-style scoring; reused across all pair comparisons."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    return aligner


def pairwise_alignment_identity_ratio(a, b):
    """
    Ratio: identical residues in the optimal global alignment (Biopython
    PairwiseAligner) divided by min(len(a), len(b)).
    """
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    aligner = _pairwise_aligner()
    aln = aligner.align(a, b)[0]
    identities = aln.counts().identities
    return identities / min_len


def compute_diversity(sequences):
    """
    Diversity(S) = 100 * average over unordered pairs:
                   Alignment(si, sj) / min(len(si), len(sj))

    Alignment(si, sj) is the number of identical residue pairs in the optimal
    global alignment from Biopython (not LCS).
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    scores = []
    for s1, s2 in combinations(sequences, 2):
        scores.append(pairwise_alignment_identity_ratio(s1, s2))

    return 100.0 * sum(scores) / len(scores)


# -----------------------------
# Uniqueness
# -----------------------------
def compute_uniqueness(sequences):
    """
    Uniqueness(S) = (number of distinct sequences / N) * 100
    """
    n = len(sequences)
    if n == 0:
        return 0.0
    return 100.0 * len(set(sequences)) / n


# -----------------------------
# Novelty
# -----------------------------
def compute_novelty(sequences, reference_sequences):
    """
    Novelty(S) = percentage of sequences in S that are NOT present
                 in the reference set H.
    """
    n = len(sequences)
    if n == 0:
        return 0.0

    ref_set = set(reference_sequences)
    novel_count = sum(1 for s in sequences if s not in ref_set)
    return 100.0 * novel_count / n


# -----------------------------
# Fitness score
# -----------------------------
def sequence_fitness(seq, theta_deg=100.0):
    """
    Compute fitness for one sequence:
    
        sqrt( (sum_i h(a_i) cos(i*theta))^2 + (sum_i h(a_i) sin(i*theta))^2 )
        ---------------------------------------------------------------------
                           sum_i exp(hx(a_i))

    where theta is 100 degrees converted to radians.

    Index i is taken as 1..L, matching the formula.
    """
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0

    theta = theta_deg * math.pi / 180.0

    x = 0.0
    y = 0.0
    denom = 0.0

    for i, aa in enumerate(seq, start=1):
        if aa not in H_SCALE or aa not in HX_SCALE:
            raise ValueError(f"Invalid amino acid '{aa}' in sequence: {seq}")

        h = H_SCALE[aa]
        hx = HX_SCALE[aa]

        x += h * math.cos(i * theta)
        y += h * math.sin(i * theta)
        denom += math.exp(hx)

    numerator = math.sqrt(x * x + y * y)

    if denom == 0:
        return 0.0

    return numerator / denom


def compute_fitness_score(sequences, theta_deg=100.0):
    """
    Fitness-Score(S) = average sequence fitness over all sequences in S
    """
    n = len(sequences)
    if n == 0:
        return 0.0

    values = [sequence_fitness(seq, theta_deg=theta_deg) for seq in sequences]
    return sum(values) / n


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Diversity, Uniqueness, Novelty, and Fitness score from FASTA files."
    )
    parser.add_argument("fasta", help="FASTA file containing generated sequences")
    parser.add_argument(
        "--reference_fasta",
        help="FASTA file containing reference sequences for novelty calculation",
        default=None
    )
    parser.add_argument(
        "--theta_deg",
        type=float,
        default=100.0,
        help="Theta in degrees for fitness calculation (default: 100)"
    )

    args = parser.parse_args()

    sequences = read_fasta(args.fasta)

    if not sequences:
        print("No sequences found in input FASTA.")
        return

    diversity = compute_diversity(sequences)
    uniqueness = compute_uniqueness(sequences)
    fitness = compute_fitness_score(sequences, theta_deg=args.theta_deg)

    print(f"Number of sequences: {len(sequences)}")
    print(f"Diversity:   {diversity:.6f}")
    print(f"Uniqueness:  {uniqueness:.6f}")
    print(f"Fitness:     {fitness:.6f}")

    if args.reference_fasta is not None:
        reference_sequences = read_fasta(args.reference_fasta)
        novelty = compute_novelty(sequences, reference_sequences)
        print(f"Novelty:     {novelty:.6f}")


if __name__ == "__main__":
    main()