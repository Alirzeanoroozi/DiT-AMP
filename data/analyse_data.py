import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

STANDARD_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta_sequences(fasta_path: Path) -> List[str]:
    sequences: List[str] = []
    current: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
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


def is_standard_sequence(sequence: str) -> bool:
    return all(amino_acid in STANDARD_AMINO_ACIDS for amino_acid in sequence)


def clean_sequences(sequences: List[str], max_length: int) -> Tuple[List[str], Dict[str, int]]:
    seen: Set[str] = set()
    cleaned: List[str] = []
    removed_duplicate = 0
    removed_too_long = 0
    removed_non_standard = 0

    for sequence in sequences:
        if len(sequence) > max_length:
            removed_too_long += 1
            continue
        if not is_standard_sequence(sequence):
            removed_non_standard += 1
            continue
        if sequence in seen:
            removed_duplicate += 1
            continue

        seen.add(sequence)
        cleaned.append(sequence)

    stats: Dict[str, int] = {
        "removed_duplicate": removed_duplicate,
        "removed_too_long": removed_too_long,
        "removed_non_standard": removed_non_standard,
    }
    return cleaned, stats


def write_fasta_sequences(sequences: List[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">{index}\n")
            handle.write(f"{sequence}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean FASTA peptides by removing duplicates, sequences longer than a "
            "maximum length, and sequences containing non-standard amino acids."
        )
    )
    parser.add_argument(
        "--fasta",
        default="data/data.fasta",
        help="Path to FASTA file (default: data/data.fasta)",
    )
    parser.add_argument(
        "--out",
        default="data/data_cleaned.fasta",
        help="Output FASTA path (default: overwrite --fasta file)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum allowed sequence length (default: 30)",
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_path = Path(args.out)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequences = read_fasta_sequences(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    cleaned_sequences, stats = clean_sequences(sequences, max_length=args.max_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta_sequences(cleaned_sequences, out_path)

    print(f"Read {len(sequences)} sequences from {fasta_path}")
    print(f"Kept {len(cleaned_sequences)} sequences")
    print(f"Removed duplicates: {stats['removed_duplicate']}")
    print(f"Removed too long (> {args.max_length}): {stats['removed_too_long']}")
    print(f"Removed non-standard amino acids: {stats['removed_non_standard']}")
    print(f"Saved cleaned FASTA to {out_path}")


if __name__ == "__main__":
    main()
