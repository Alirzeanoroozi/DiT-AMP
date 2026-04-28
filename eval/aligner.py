from Bio.Align import PairwiseAligner
from typing import Tuple, List

aligner = PairwiseAligner()
aligner.mode = "global"     # Needleman–Wunsch style global alignment (globalxx equivalent)

# Equivalent spirit to globalxx: match=1, mismatch=0.
aligner.match_score = 1.0
aligner.mismatch_score = 0.0

# gap penalties = 0 to match globalxx behavior
aligner.open_gap_score = 0.0
aligner.extend_gap_score = 0.0
aligner.target_end_gap_score = 0.0
aligner.query_end_gap_score = 0.0

def nw_globalxx_scores(seq_a: str, seq_b: str) -> Tuple[float, float, float]:
    """Return (raw_score, norm_max_len, norm_alignment_len) for Biopython globalxx settings.

    With zero penalties for mismatches and gaps/end-gaps, this mirrors pairwise2.align.globalxx.
    In this configuration, the first optimal alignment has alignment_len == max(len_a, len_b),
    but we return both normalizations explicitly for clarity.
    """
    if not seq_a or not seq_b:
        return 0.0, 0.0, 0.0

    raw = aligner.score(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b))
    alignment_len = max_len  # globalxx with no gap penalties => alignment length equals max_len

    norm_max = raw / max_len if max_len else 0.0
    norm_align = raw / alignment_len if alignment_len else 0.0
    return raw, norm_max, norm_align

def nw_similarity_percent(seq_a: str, seq_b: str) -> float:
    _, norm_max, _ = nw_globalxx_scores(seq_a, seq_b)
    return 100.0 * norm_max

def max_similarity_to_ref(query_seq: str, ref_seqs: List[str]) -> Tuple[float, int, float, float]:
    """Return (best_raw, best_index, best_norm_max_len, best_norm_alignment_len).
    We keep norm_max_len as the comparator for "best" to stay aligned with the paper's similarity metric.
    """
    best_raw = -1.0
    best_norm = -1.0
    best_align_norm = -1.0
    best_i = -1
    for i, rseq in enumerate(ref_seqs):
        raw, norm_max, norm_align = nw_globalxx_scores(query_seq, rseq)
        if norm_max > best_norm:
            best_raw = raw
            best_norm = norm_max
            best_align_norm = norm_align
            best_i = i
    return best_raw, best_i, best_norm, best_align_norm
