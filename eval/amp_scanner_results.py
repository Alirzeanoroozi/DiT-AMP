import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODEL_ORDER = [
    "dit_amp",
    "ampgan",
    "pepcvae",
    "hydramp",
    "ampdiffusion",
]


def _parse_float(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def model_name_from_filename(path):
    name = path.name
    if name.endswith("_sequences.csv"):
        return name[: -len("_sequences.csv")]
    return path.stem


def _is_amp_label(label):
    if label is None:
        return False
    return str(label).strip().upper() == "AMP"


def summarize_amp_scanner_file(path):
    """Compute per-file means for probability and AMP class rate."""
    model = model_name_from_filename(path)
    probs = []
    amp_flags = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prob = _parse_float(row.get("Prediction_Probability"))
            if prob is None:
                continue
            probs.append(prob)
            amp_flags.append(1.0 if _is_amp_label(row.get("Prediction_Class")) else 0.0)

    if not probs:
        return {
            "model": model,
            "n_scored": 0,
            "probability_mean": float("nan"),
            "probability_std": float("nan"),
            "amp_rate_mean": float("nan"),
            "amp_rate_std": float("nan"),
            "path": str(path),
        }

    probs_arr = np.array(probs, dtype=np.float64)
    amp_arr = np.array(amp_flags, dtype=np.float64)
    return {
        "model": model,
        "n_scored": int(probs_arr.size),
        "probability_mean": float(np.mean(probs_arr)),
        "probability_std": float(np.std(probs_arr)),
        "amp_rate_mean": float(np.mean(amp_arr)),
        "amp_rate_std": float(np.std(amp_arr)),
        "path": str(path),
    }


def _sort_key(model):
    try:
        idx = MODEL_ORDER.index(model)
    except ValueError:
        idx = len(MODEL_ORDER)
    return (idx, model)


def summarize_amp_scanner_directory(directory=None, pattern="*_sequences.csv"):
    if directory is None:
        directory = Path(__file__).resolve().parent / "amp-scanner"
    paths = sorted(directory.glob(pattern), key=lambda p: _sort_key(model_name_from_filename(p)))
    return [summarize_amp_scanner_file(p) for p in paths]


def write_summary_csv(path, rows):
    rows = list(rows)
    fieldnames = [
        "model",
        "n_scored",
        "probability_mean",
        "probability_std",
        "amp_rate_mean",
        "amp_rate_std",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(summaries, output_path, title="AMP-Scanner mean +/- std per model"):
    if plt is None:
        print("matplotlib is not installed; skipping plot.")
        return

    labels = [s["model"] for s in summaries]
    prob_means = [s["probability_mean"] for s in summaries]
    prob_stds = [s["probability_std"] for s in summaries]
    amp_rate_means = [s["amp_rate_mean"] for s in summaries]
    amp_rate_stds = [s["amp_rate_std"] for s in summaries]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        prob_means,
        width,
        yerr=prob_stds,
        capsize=4,
        label="Prediction_Probability",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        amp_rate_means,
        width,
        yerr=amp_rate_stds,
        capsize=4,
        label="AMP_rate",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "amp-scanner"
    out_csv = script_dir / "metrics" / "amp_scanner_summary.csv"
    out_png = script_dir / "metrics" / "amp_scanner_mean_std_plot.png"

    summaries = summarize_amp_scanner_directory(input_dir)
    if not summaries:
        raise FileNotFoundError("No *_sequences.csv files found in {}".format(input_dir))

    write_summary_csv(out_csv, summaries)
    plot_summary(summaries, out_png)

    print("Wrote {}".format(out_csv))
    if plt is not None:
        print("Wrote {}".format(out_png))
    for row in summaries:
        print(
            "{}: n_scored={}, probability_mean={:.6f}, amp_rate_mean={:.6f}".format(
                row["model"],
                row["n_scored"],
                row["probability_mean"],
                row["amp_rate_mean"],
            )
        )


if __name__ == "__main__":
    main()
