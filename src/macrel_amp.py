import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


from macrel.run_macrel import run_macrel



TOXINPRED_PATTERNS = [
    re.compile(r"toxinpred(?:_score)?\s*[=:]\s*([-+]?\d*\.?\d+)", re.IGNORECASE),
    re.compile(r"toxin(?:_pred)?\s*[=:]\s*([-+]?\d*\.?\d+)", re.IGNORECASE),
]


def parse_toxinpred_value(description: str) -> float:
    for pattern in TOXINPRED_PATTERNS:
        match = pattern.search(description)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return math.nan
    return math.nan


def load_fasta_records(fasta_path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    current_id = None
    current_description = None
    current_sequence: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    records.append(
                        {
                            "id": current_id,
                            "description": current_description or current_id,
                            "sequence": "".join(current_sequence),
                        }
                    )
                header = line[1:].strip()
                current_description = header
                current_id = header.split()[0] if header else "unknown"
                current_sequence = []
            else:
                current_sequence.append(line)

    if current_id is not None:
        records.append(
            {
                "id": current_id,
                "description": current_description or current_id,
                "sequence": "".join(current_sequence),
            }
        )
    return records


def build_rows(records: List[Dict[str, str]], limit: Optional[int] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected_records = records if limit is None else records[:limit]

    for index, item in enumerate(selected_records, start=1):
        sequence = item["sequence"]
        toxinpred_value = parse_toxinpred_value(item["description"])
        if run_macrel is None:
            macrel_amp, macrel_hemo = math.nan, math.nan
        else:
            try:
                macrel_amp, macrel_hemo = run_macrel(sequence)
            except Exception:
                macrel_amp, macrel_hemo = math.nan, math.nan

        rows.append(
            {
                "id": item["id"],
                "sequence": sequence,
                "toxinpred": toxinpred_value,
                "macrel_amp": macrel_amp,
                "macrel_hemo": macrel_hemo,
            }
        )
        if index % 500 == 0:
            print("Processed {0}/{1}".format(index, len(selected_records)))
    return rows


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "sequence", "toxinpred", "macrel_amp", "macrel_hemo"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CSV with toxinpred (if found in FASTA headers) and "
            "Macrel AMP/Hemo scores for each peptide."
        )
    )
    parser.add_argument(
        "--fasta",
        default="data/data.fasta",
        help="Input FASTA path (default: data/data.fasta)",
    )
    parser.add_argument(
        "--out",
        default="data/toxinpred_macrel.csv",
        help="Output CSV path (default: data/toxinpred_macrel.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of sequences to process for quick testing",
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_path = Path(args.out)
    if not fasta_path.exists():
        raise FileNotFoundError("FASTA file not found: {0}".format(fasta_path))

    records = load_fasta_records(fasta_path)
    if not records:
        raise ValueError("No records found in FASTA: {0}".format(fasta_path))

    rows = build_rows(records, limit=args.limit)
    write_csv(rows, out_path)
    num_with_toxinpred = sum(0 if math.isnan(float(x["toxinpred"])) else 1 for x in rows)
    if run_macrel is None:
        print("Warning: Macrel dependencies not available. Macrel columns are NaN.")

    print("Input records: {0}".format(len(records)))
    print("Processed records: {0}".format(len(rows)))
    print("Rows with toxinpred in header: {0}".format(num_with_toxinpred))
    print("Saved CSV to: {0}".format(out_path))


if __name__ == "__main__":
    main()


# toxinpred3 --i data/data.fasta --out data/toxinpred3.csv