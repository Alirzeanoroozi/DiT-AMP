"""
modlAMP physicochemical metrics utilities.

Provides instability, charge, isoelectric point (pI), hydrophobic ratio,
aromaticity, Boman index, TM tendency, aliphatic index, and significance helpers.
"""

import inspect
from typing import List, Union

import numpy as np
import pandas as pd

from modlamp.descriptors import GlobalDescriptor

METHOD_CANDIDATES = {
    "instability": [
        "instability_index",
        "calculate_instability_index",
        "calculate_instability",
        "instability",
    ],
    "charge": [
        "calculate_charge",
        "charge",
    ],
    "isoelectric_point": [
        "isoelectric_point",
        "calculate_isoelectric_point",
        "pI",
        "calculate_pI",
    ],
    "hydrophobic_ratio": [
        "hydrophobic_ratio",
        "calculate_hydrophobic_ratio",
        "hydrophobicity_ratio",
    ],
    "aromaticity": [
        "aromaticity",
        "calculate_aromaticity",
    ],
    "boman_index": [
        "boman_index",
        "calculate_boman_index",
        "boman",
        "calculate_boman",
    ],
}

TM_TEND_SCALE = {
    "A": 0.38,  "R": -2.57, "N": -1.62, "D": -3.27, "C": -0.30,
    "Q": -1.84, "E": -2.90, "G": -0.19, "H": -1.44, "I": 1.97,
    "L": 1.82,  "K": -3.46, "M": 1.40,  "F": 1.98,  "P": -1.44,
    "S": -0.53, "T": -0.32, "W": 1.53,  "Y": 0.49,  "V": 1.46,
}

# Map non-standard/ambiguous amino acids to standard ones for modlAMP compatibility
# B=Asx(D/N), Z=Glx(E/Q), X=unknown, U=selenocysteine, O=pyrrolysine, J=Leu/Ile
# modlAMP's instability_index uses a DIV matrix with only 20 standard codes
NONSTANDARD_AA_MAP = {
    "B": "N",  # Asx: Asp or Asn -> Asn
    "Z": "Q",  # Glx: Glu or Gln -> Gln
    "X": "A",  # Unknown -> Ala
    "U": "C",  # Selenocysteine -> Cys
    "O": "K",  # Pyrrolysine -> Lys
    "J": "L",  # Leu or Ile -> Leu
}
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def _sanitize_seqs_for_modlamp(seqs: List[str]) -> List[str]:
    """
    Replace non-standard amino acids so modlAMP's descriptors
    (e.g. instability_index) don't raise KeyError.
    """
    result = []
    for s in seqs:
        s = (s or "").strip().upper()
        sanitized = "".join(
            c if c in STANDARD_AAS else NONSTANDARD_AA_MAP.get(c, "A")
            for c in s if c.isalpha()
        )
        result.append(sanitized)
    return result


def _call_method_with_defaults(obj, method_name: str):
    """
    Call obj.method_name with no args if possible.
    If the method requires a pH argument, call it with pH=7.0 (common default).
    """
    method = getattr(obj, method_name)

    # Try no-arg call first
    try:
        return method()
    except TypeError:
        pass

    # If it failed, inspect signature and try pH defaults
    sig = inspect.signature(method)
    params = list(sig.parameters.values())

    # If it has a single required parameter besides self, try pH=7.0
    # (common in charge calculations in many peptide toolkits)
    kwargs = {}
    for p in params:
        if p.name.lower() in ("ph", "ph"):
            kwargs[p.name] = 7.0

    # If we found pH, try calling with it
    if kwargs:
        return method(**kwargs)

    # Otherwise, re-raise with helpful info
    raise TypeError(
        f"Method '{method_name}' exists but couldn't be called without args. "
        f"Signature: {sig}"
    )


def _calc_modlamp_metric(seqs: List[str], metric_key: str) -> np.ndarray:
    """
    Compute one metric for a list of sequences using modlAMP GlobalDescriptor,
    robust to different modlAMP method names.
    Non-standard amino acids (B, Z, X) are mapped to standard ones before calling modlAMP.
    """
    sanitized = _sanitize_seqs_for_modlamp(seqs)
    gd = GlobalDescriptor(sanitized)

    # 1) Try our known candidate method names
    for name in METHOD_CANDIDATES.get(metric_key, []):
        if hasattr(gd, name):
            _call_method_with_defaults(gd, name)
            return np.asarray(gd.descriptor).flatten()

    # 2) Fallback: fuzzy search based on substrings
    # (helps when modlAMP renames things in a future version)
    want = metric_key.lower()
    available = [m for m in dir(gd) if not m.startswith("_")]

    def score_name(m: str) -> int:
        ml = m.lower()
        score = 0
        if want in ml:
            score += 5
        if want == "tm_tend":
            if "tm" in ml:
                score += 2
            if "tend" in ml:
                score += 2
        if want == "isoelectric_point":
            if "iso" in ml:
                score += 2
            if "electric" in ml:
                score += 2
            if "pi" == ml or "pi" in ml:
                score += 2
        if want == "instability":
            if "instab" in ml:
                score += 2
        if want == "hydrophobic_ratio":
            if "hydrophob" in ml:
                score += 2
            if "ratio" in ml:
                score += 1
        return score

    ranked = sorted(available, key=score_name, reverse=True)
    for name in ranked[:25]:
        if score_name(name) <= 0:
            break
        try:
            _call_method_with_defaults(gd, name)
            return np.asarray(gd.descriptor).flatten()
        except Exception:
            continue

    # 3) If still nothing worked, show helpful methods
    hint = [
        m for m in available
        if any(x in m.lower() for x in ["instab", "tm", "tend", "charge", "iso", "pi", "hydrophob", "aroma"])
    ]
    raise AttributeError(
        f"Could not compute metric '{metric_key}'. "
        f"Tried candidates: {METHOD_CANDIDATES.get(metric_key, [])}. "
        f"Relevant available methods: {hint[:50]}"
    )


def tm_tend_score(seq: str) -> float:
    """
    TM_tend score = mean(Zhao&London TM tendency values over residues).
    Non-standard letters are ignored (or you can choose to return NaN).
    """
    seq = (seq or "").strip().upper()
    vals = [TM_TEND_SCALE.get(aa) for aa in seq if aa in TM_TEND_SCALE]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def compute_modlamp_metrics(seqs: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Compute metrics:
    - instability, charge, isoelectric_point, hydrophobic_ratio, aromaticity via modlAMP
    - tm_tend via our fallback implementation (since modlAMP here lacks it)
    """
    seqs = list(seqs) if not isinstance(seqs, list) else seqs
    out = {}

    # modlAMP metrics that exist
    for metric_key in METHOD_CANDIDATES.keys():
        out[metric_key] = _calc_modlamp_metric(seqs, metric_key)

    # TM_tend fallback
    out["tm_tend"] = np.array([tm_tend_score(s) for s in seqs], dtype=float)

    return pd.DataFrame(out)


def aliphatic_index(seq: str) -> float:
    """
    Aliphatic index: 100 * (f_A + 2.9 * f_V + 3.9 * (f_I + f_L)).
    """
    s = "".join(c for c in str(seq).upper() if c.isalpha())
    n = len(s)
    if n == 0:
        return np.nan
    f_a = s.count("A") / n
    f_v = s.count("V") / n
    f_i = s.count("I") / n
    f_l = s.count("L") / n
    return 100.0 * (f_a + 2.9 * f_v + 3.9 * (f_i + f_l))


def compute_boman_index(seqs: List[str]) -> np.ndarray:
    """Compute Boman index using modlAMP GlobalDescriptor."""
    return _calc_modlamp_metric(seqs, "boman_index")


def p_to_star(p: float) -> str:
    """Convert p-value to significance stars (ns, *, **, ***)."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


# Backwards-compatible aliases used in eval_metrics.ipynb
_aliphatic_index = aliphatic_index
_compute_boman_modlamp = compute_boman_index
