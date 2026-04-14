#!/usr/bin/env python3

import math
import argparse


# -----------------------------
# FASTA reader
# -----------------------------
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


# -----------------------------
# Shannon Entropy
# -----------------------------
def shannon_entropy(sequence):
    """
    H(X) = - sum p(x) log2 p(x)
    """
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
    """
    Returns:
    - list of entropies
    - average entropy
    """
    values = [shannon_entropy(seq) for seq in sequences]
    avg = sum(values) / len(values) if values else 0.0
    return values, avg


# -----------------------------
# k-mer extraction
# -----------------------------
def get_kmers(sequence, k):
    """
    Extract set of k-mers from a sequence
    """
    return {
        sequence[i:i + k]
        for i in range(len(sequence) - k + 1)
    }


def get_kmer_set(sequences, k):
    """
    Get union of k-mers across a set of sequences
    """
    kmers = set()
    for seq in sequences:
        kmers.update(get_kmers(seq, k))
    return kmers


# -----------------------------
# Jaccard similarity
# -----------------------------
def jaccard_similarity(set_a, set_b):
    """
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def compute_jaccard_kmer(gen_seqs, ref_seqs, k):
    """
    Compute Jaccard similarity between generated and reference sequences
    using k-mers.
    """
    gen_kmers = get_kmer_set(gen_seqs, k)
    ref_kmers = get_kmer_set(ref_seqs, k)

    return jaccard_similarity(gen_kmers, ref_kmers)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Shannon entropy and Jaccard similarity (k-mers)."
    )

    parser.add_argument("generated_fasta", help="Generated sequences FASTA")
    parser.add_argument(
        "--reference_fasta",
        help="Reference/training sequences FASTA (for Jaccard)",
        required=True
    )

    args = parser.parse_args()

    gen_seqs = read_fasta(args.generated_fasta)
    ref_seqs = read_fasta(args.reference_fasta)

    if not gen_seqs:
        print("No generated sequences found.")
        return

    # --- Entropy ---
    entropies, avg_entropy = compute_entropy_stats(gen_seqs)

    print("=== Shannon Entropy ===")
    print(f"Average entropy: {avg_entropy:.6f}")

    # --- Jaccard similarity ---
    js_3 = compute_jaccard_kmer(gen_seqs, ref_seqs, k=3)
    js_6 = compute_jaccard_kmer(gen_seqs, ref_seqs, k=6)

    print("\n=== Jaccard Similarity ===")
    print(f"3-mer JS: {js_3:.6f}")
    print(f"6-mer JS: {js_6:.6f}")


if __name__ == "__main__":
    main()