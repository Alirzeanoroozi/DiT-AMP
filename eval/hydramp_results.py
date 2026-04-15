#!/usr/bin/env python3
"""Summarize and plot HydrAMP hydramp_amp / hydramp_mic means per generated model."""

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


import matplotlib.pyplot as plt


# Plot / table order (matches other eval scripts where possible)
MODEL_ORDER = [
    "dit_amp",
    "ampgan",
    "pepcvae",
    "hydramp",
    "ampdiffusion",
]

_SUFFIX = "_sequences_with_hydramp.csv"


def model_name_from_filename(path: Path) -> str:
    stem = path.name
    if not stem.endswith(_SUFFIX):
        return stem.replace(".csv", "")
    return stem[: -len(_SUFFIX)]


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


def load_hydramp_columns(path):
    """Return (hydramp_amp, hydramp_mic) arrays with only rows where both scores are present."""
    amp_list: list[float] = []
    mic_list: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            a = _parse_float(row.get("hydramp_amp"))
            m = _parse_float(row.get("hydramp_mic"))
            if a is None or m is None:
                continue
            amp_list.append(a)
            mic_list.append(m)
    return (
        np.array(amp_list, dtype=np.float64),
        np.array(mic_list, dtype=np.float64),
    )


def summarize_hydramp_file(path):
    """Per-model stats for one `*_sequences_with_hydramp.csv` file."""
    model = model_name_from_filename(path)
    amp, mic = load_hydramp_columns(path)
    n = int(amp.size)
    if n == 0:
        return {
            "model": model,
            "path": str(path),
            "n_scored": 0,
            "hydramp_amp_mean": float("nan"),
            "hydramp_amp_std": float("nan"),
            "hydramp_mic_mean": float("nan"),
            "hydramp_mic_std": float("nan"),
        }
    return {
        "model": model,
        "path": str(path),
        "n_scored": n,
        "hydramp_amp_mean": float(np.mean(amp)),
        "hydramp_amp_std": float(np.std(amp)),
        "hydramp_mic_mean": float(np.mean(mic)),
        "hydramp_mic_std": float(np.std(mic)),
    }


def _sort_key(model):
    try:
        idx = MODEL_ORDER.index(model)
    except ValueError:
        idx = len(MODEL_ORDER)
    return (idx, model)


def summarize_hydramp_directory(directory=None, pattern="*_sequences_with_hydramp.csv"):
    """Summarize every matching CSV under ``directory`` (default: this file's folder)."""
    if directory is None:
        directory = Path(__file__).resolve().parent
    paths = sorted(directory.glob(pattern), key=lambda p: _sort_key(model_name_from_filename(p)))
    return [summarize_hydramp_file(p) for p in paths]


def write_summary_csv(path, rows):
    rows = list(rows)
    fieldnames = [
        "model",
        "n_scored",
        "hydramp_amp_mean",
        "hydramp_amp_std",
        "hydramp_mic_mean",
        "hydramp_mic_std",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_hydramp_means(summaries, output_path, title="HydrAMP mean ± std per model"):
    """Grouped bar chart: hydramp_amp vs hydramp_mic means per model."""
    if plt is None:
        print("matplotlib is not installed; skipping plot.")
        return
    labels = [s["model"] for s in summaries]
    amp_means = [s["hydramp_amp_mean"] for s in summaries]
    amp_stds = [s["hydramp_amp_std"] for s in summaries]
    mic_means = [s["hydramp_mic_mean"] for s in summaries]
    mic_stds = [s["hydramp_mic_std"] for s in summaries]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        amp_means,
        width,
        yerr=amp_stds,
        capsize=4,
        label="hydramp_amp",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        mic_means,
        width,
        yerr=mic_stds,
        capsize=4,
        label="hydramp_mic",
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


def main() -> None:
    hydramp_dir = Path(__file__).resolve().parent / "hydramp_results"
    out_csv = hydramp_dir / "hydramp_summary.csv"
    out_png = hydramp_dir / "hydramp_mean_std_plot.png"

    summaries = summarize_hydramp_directory(hydramp_dir)
    if not summaries:
        raise FileNotFoundError(f"No *{_SUFFIX} files under {hydramp_dir}")

    write_summary_csv(out_csv, summaries)
    plot_hydramp_means(summaries, out_png)

    print(f"Wrote {out_csv}")
    if plt is not None:
        print(f"Wrote {out_png}")
    for row in summaries:
        print(
            f"{row['model']}: n_scored={row['n_scored']}, "
            f"hydramp_amp_mean={row['hydramp_amp_mean']:.6f}, "
            f"hydramp_mic_mean={row['hydramp_mic_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
