# Standard 20 amino acids (one-letter)
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

import os
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("aa_freq", exist_ok=True)

fasta_files = {
    "ampgan": "../genbio/ampgan_sequences.fasta",
    "pepcvae": "../genbio/pepcvae_sequences.fasta",
    "hydramp": "../genbio/hydramp_sequences.fasta",
    "dit_amp": "../genbio/dit_amp_sequences.fasta",
}

def read_fasta(path):
    records_raw = list(SeqIO.parse(path, "fasta"))
    return [str(x.seq) for x in records_raw]

def normalized_aa_freq(seqs):
    """Count each AA across all sequences and normalize so frequencies sum to 1."""
    from collections import Counter
    counter = Counter()
    for s in seqs:
        counter.update(c.upper() for c in s if c.isalpha())
    total = sum(counter.values()) or 1
    return {aa: counter.get(aa, 0) / total for aa in AA_ORDER}

if __name__ == "__main__":
    ref_seqs = read_fasta("../data/data.fasta")
    for name, fasta_file in fasta_files.items():
        qry_seqs = read_fasta(fasta_file)
        train_freq = normalized_aa_freq(ref_seqs)
        gen_freq = normalized_aa_freq(qry_seqs)
        x = np.arange(len(AA_ORDER))
        w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w/2, [gen_freq[a] for a in AA_ORDER], width=w, label="Generated", color="#7eb8da", edgecolor="none")
        ax.bar(x + w/2, [train_freq[a] for a in AA_ORDER], width=w, label="Train", color="#86c67c", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(AA_ORDER)
        ax.set_xlabel("Amino Acids")
        ax.set_ylabel("Normalized Frequency")
        ax.set_title(f"{name}: Train vs Generated: Amino Acid Frequency")
        ax.legend(loc="upper right")
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.tight_layout()
        aa_freq_path = os.path.join("aa_freq", f"{name}_aa_freq.png")
        fig.savefig(aa_freq_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {aa_freq_path}")