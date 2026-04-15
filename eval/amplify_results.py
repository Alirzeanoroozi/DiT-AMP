import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


REQUIRED_COLUMNS = [
    "Probability_score",
    "AMPlify_log_scaled_score",
    "Prediction",
]


def get_delimiter(file_path: Path) -> str:
    if file_path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_file(file_path: Path) -> dict:
    delimiter = get_delimiter(file_path)
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing required columns: {missing_columns}"
            )

        row_count = 0
        probability_sum = 0.0
        probability_count = 0
        amplify_log_sum = 0.0
        amplify_log_count = 0
        amp_count = 0

        for row in reader:
            row_count += 1
            probability_value = safe_float(row["Probability_score"])
            if probability_value is not None:
                probability_sum += probability_value
                probability_count += 1

            amplify_log_value = safe_float(row["AMPlify_log_scaled_score"])
            if amplify_log_value is not None:
                amplify_log_sum += amplify_log_value
                amplify_log_count += 1

            if str(row["Prediction"]).strip().upper() == "AMP":
                amp_count += 1

    if row_count == 0:
        raise ValueError(f"{file_path.name} has no data rows.")

    return {
        "file_name": file_path.name,
        "num_rows": row_count,
        "Probability_score_mean": probability_sum / probability_count
        if probability_count
        else float("nan"),
        "AMPlify_log_scaled_score_mean": amplify_log_sum / amplify_log_count
        if amplify_log_count
        else float("nan"),
        "AMP_prediction_rate": amp_count / row_count,
    }


def plot_summary(summaries: list, output_path: Path) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping summary plot generation.")
        return

    metrics = [
        "Probability_score_mean",
        "AMPlify_log_scaled_score_mean",
        "AMP_prediction_rate",
    ]

    file_names = [item["file_name"] for item in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, metric in enumerate(metrics):
        values = [item[metric] for item in summaries]
        axes[idx].bar(file_names, values)
        axes[idx].set_title(metric)
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "amplify_results"
    output_summary = script_dir / "metrics" / "amplify_results_summary.csv"
    output_plot = script_dir / "metrics" / "amplify_results_summary.png"

    files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.tsv")))
    if not files:
        raise FileNotFoundError(f"No CSV/TSV files found in {input_dir}")

    summaries = [summarize_file(file_path) for file_path in files]
    summaries = sorted(summaries, key=lambda item: item["file_name"])

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "num_rows",
                "Probability_score_mean",
                "AMPlify_log_scaled_score_mean",
                "AMP_prediction_rate",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    plot_summary(summaries, output_plot)

    print("Saved summary:", output_summary)
    if plt is not None:
        print("Saved plot:", output_plot)
    print("\nPer-file summary:")
    for row in summaries:
        print(
            (
                f"{row['file_name']}: "
                f"Probability_score_mean={row['Probability_score_mean']:.6f}, "
                f"AMPlify_log_scaled_score_mean={row['AMPlify_log_scaled_score_mean']:.6f}, "
                f"AMP_prediction_rate={row['AMP_prediction_rate']:.6f}"
            )
        )


if __name__ == "__main__":
    main()
