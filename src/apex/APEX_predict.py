import numpy as np
import torch
import glob
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from Bio import SeqIO
from utils import make_vocab, onehot_encoding

model_list = [torch.load(a_model, weights_only=False, map_location='cpu').eval() for a_model in glob.glob('./APEX_pathogen_models/APEX_*')]
max_len = 52 #maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
word2idx, idx2word = make_vocab() #make amino acid vocabulary
#emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

def predict(seq):
	seq = np.array([seq])
	X_seq = torch.LongTensor(onehot_encoding(seq, max_len, word2idx) )

	#Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
	#8 pretrained APEX models are provided, and predictions are averaged
	# Store predictions from each APEX model
	all_preds = []

	for model in model_list:
		AMP_pred = model(X_seq).detach().numpy()  # make predictions
		AMP_pred = 10 ** (6 - AMP_pred)  # transform back to MICs
		all_preds.append(AMP_pred)

	return np.mean(all_preds)

def read_fasta(path):
    records_raw = list(SeqIO.parse(path, "fasta"))
    return [str(x.seq) for x in records_raw]

fasta_files = {
    "ampgan": "../../genbio/ampgan_sequences.fasta",
    "pepcvae": "../../genbio/pepcvae_sequences.fasta",
    "hydramp": "../../genbio/hydramp_sequences.fasta",
    "dit_amp": "../../genbio/dit_amp_sequences.fasta",
    "ampdiffusion": "../../genbio/ampdiffusion_sequences.fasta",
}

MODEL_ORDER = [
    "dit_amp",
    "ampgan",
    "pepcvae",
    "hydramp",
    "ampdiffusion",
]


def write_per_file_csv(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "apex_mic"])
        writer.writerows(rows)


def write_summary_csv(path, summary_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "n_sequences", "apex_mic_mean", "apex_mic_std"])
        writer.writerows(summary_rows)


def plot_summary(path, labels, mic_means, mic_stds):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x,
        mic_means,
        yerr=mic_stds,
        capsize=4,
        color="#4C78A8",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Predicted MIC (uM)")
    ax.set_title("APEX predicted MIC mean +/- std across datasets")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sort_key(name):
    try:
        return MODEL_ORDER.index(name)
    except ValueError:
        return len(MODEL_ORDER)


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parents[1] / "eval" / "apex_results"

    summary_rows = []
    labels = []
    mic_means = []
    mic_stds = []

    for name in sorted(fasta_files.keys(), key=_sort_key):
        fasta_file = fasta_files[name]
        fasta_path = script_dir / fasta_file
        sequences = read_fasta(str(fasta_path))

        per_file_rows = []
        mic_vals = []
        skipped = 0

        for seq in sequences:
            try:
                mic = float(predict(seq))
                per_file_rows.append([seq, mic])
                mic_vals.append(mic)
            except Exception:
                skipped += 1
                continue

        output_csv = out_dir / "{}_apex.csv".format(name)
        write_per_file_csv(output_csv, per_file_rows)

        mic_vals = np.array(mic_vals, dtype=float)
        if mic_vals.size == 0:
            mic_mean = float("nan")
            mic_std = float("nan")
        else:
            mic_mean = float(np.mean(mic_vals))
            mic_std = float(np.std(mic_vals))

        n_scored = int(mic_vals.size)
        summary_rows.append([name, n_scored, mic_mean, mic_std])
        labels.append(name)
        mic_means.append(mic_mean)
        mic_stds.append(mic_std)

        print("{}: total={}, scored={}, skipped={}".format(name, len(sequences), n_scored, skipped))
        print("  apex_mic_mean={:.6f}, apex_mic_std={:.6f}".format(mic_mean, mic_std))
        print("  wrote {}".format(output_csv))

    summary_csv = out_dir / "apex_summary.csv"
    write_summary_csv(summary_csv, summary_rows)
    print("Wrote summary: {}".format(summary_csv))

    plot_path = out_dir / "apex_mean_std_plot.png"
    plot_summary(plot_path, labels, mic_means, mic_stds)
    print("Wrote plot: {}".format(plot_path))

if __name__ == "__main__":
    main()