"""
mpnn_detect - heuristic screening for de novo / inverse-folding designed proteins.

WHAT THIS IS
------------
A multi-signal screen that estimates whether a protein sequence was produced by
a sequence-design model (ProteinMPNN, LigandMPNN, ESM-IF, and relatives) rather
than by evolution. It replaces the "natural HMM vs designed HMM" approach, which
cannot work: a profile HMM models one family's conserved positions, and
"designed" is not a family. A ProteinMPNN redesign of a natural backbone keeps
the fold and typically 30-50% identity to its parent, so it scores HIGH on the
natural profile and near zero on any "designed" profile built from unrelated
scaffolds.

The signals used here instead:

  1. Composition fingerprint. Inverse-folding models have strong, reproducible
     residue biases (E/K/A/L/R enriched; C/M/W/H strongly depleted).
  2. Profile residual. Build ONE profile HMM for the natural family, then ask a
     two-part question: does the query match the conserved core (fold preserved)
     while the variable surface positions carry the generator's preferences
     rather than the family's? That conjunction is the discriminative signal.
  3. Homolog isolation. Natural sequences have close relatives. A redesign
     matches the family profile but has no near neighbour - "fold match,
     sequence orphan".
  4. Idealization. De novo scaffolds are unusually clean: high helical
     amphipathicity, low entropy, repeated motifs, no cysteines.
  5. Construct motifs. His-tags, GS linkers, TEV sites - near-definitive
     evidence of an expression construct rather than a database entry.

WORKING FROM REAL GENERATOR OUTPUT
----------------------------------
Simulated designs cannot validate this method, and for a long time every number
it reported came from them. `simulate_designed` samples from the profile the
detector scores against and tilts the composition with the fingerprint the
detector tests for, so it cannot produce a design whose failure mode the
detector does not already model. Those numbers are upper bounds.

So point it at the real thing:

    gens    = load_generator_output("pmpnn_out/")      # skips the native record
    designs = generator_sequences(gens)
    report  = cross_validate(family, designed_sequences=designs)
    print(detection_by_group(detector, gens, by="source"))

Three traps are handled here because each of them silently moves the answer:

*   ProteinMPNN writes the NATIVE sequence as the first record of every output
    file. It is a natural sequence. Scored as a design it is a guaranteed miss.
*   A backbone is one domain carved out of a larger protein. Comparing bare
    domains against whole UniProt entries measures domain context, not design.
    `extract_domains()` trims the natural class to the same aligned span, and
    the backbones' own native sequences are the control that says whether it
    worked: if those get flagged, the comparison is still unfair.
*   Designs from one backbone are not independent trials. The pooled rate over
    540 designs from six structures looks like 540 pieces of evidence and is
    six. `detection_by_group()` prints the spread, which is the error bar that
    applies to a protein you have not tried yet.

HONEST LIMITS - READ THIS
-------------------------
*   The shipped composition priors are approximate, drawn from published
    descriptions of ProteinMPNN output at low sampling temperature. Measured
    against real ProteinMPNN output on natural kinase backbones, the prior
    disagrees in DIRECTION on D, G, P, W and Y: those published descriptions
    come from de novo helical bundles, and redesigning a natural backbone is a
    different distribution. Prefer `fit_fingerprint()` on your own output;
    `PROTEINMPNN_MEASURED` is a better default than the prior but was measured
    on six kinase backbones, and composition follows the structure given.
*   The default logistic weights are hand-set priors, not fitted.
*   High-temperature sampling, post-design filtering, or explicit composition
    constraints will blunt signal 1 substantially. A design pipeline that
    optimises against a detector like this one will defeat it. Detection falls
    monotonically with sampling temperature in the measurements here.
*   Some signals come out INVERTED against real output - informative, but
    pointing the wrong way. See `apply_validation(flip_inverted=...)`; whether
    flipping generalises to an unseen backbone is measured, not assumed.
*   This is a triage/prioritisation aid. It is not a biosecurity control and
    must not be used as one on its own.

Dependencies: numpy required. pyhmmer optional but needed for all profile
signals (pip install pyhmmer). No system packages, no MAFFT required.

License: MIT.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import (Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np

__version__ = "5.3.3"

LOG = logging.getLogger("mpnn_detect")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    LOG.addHandler(_h)
    # Colab and Jupyter attach a handler to the root logger, so propagating as
    # well prints every message twice.
    LOG.propagate = False
LOG.setLevel(logging.INFO)

try:
    import pyhmmer
    from pyhmmer.easel import (Alphabet, DigitalSequenceBlock, TextMSA,
                               TextSequence)
    from pyhmmer.plan7 import Background, Builder, Pipeline

    HAVE_PYHMMER = True
except Exception as _exc:  # pragma: no cover - environment dependent
    HAVE_PYHMMER = False
    _PYHMMER_IMPORT_ERROR = _exc
    LOG.warning(
        "pyhmmer unavailable (%s); profile-based signals disabled, "
        "composition-only mode active. pip install pyhmmer to enable them.",
        _exc,
    )


# ===========================================================================
# 1. Sequence handling
# ===========================================================================

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

#: Ambiguous / non-standard residues mapped to a standard representative.
#: U = selenocysteine -> C, O = pyrrolysine -> K. B/Z/J are genuinely ambiguous
#: and become X, which is treated as missing rather than guessed.
AMBIGUOUS_MAP = {"U": "C", "O": "K", "B": "X", "Z": "X", "J": "X"}

#: UniProtKB residue background frequencies (release-to-release stable to ~0.1%).
NATURAL_BACKGROUND = np.array(
    [
        0.0825,  # A
        0.0138,  # C
        0.0545,  # D
        0.0675,  # E
        0.0386,  # F
        0.0707,  # G
        0.0227,  # H
        0.0591,  # I
        0.0581,  # K
        0.0965,  # L
        0.0241,  # M
        0.0406,  # N
        0.0474,  # P
        0.0393,  # Q
        0.0553,  # R
        0.0663,  # S
        0.0535,  # T
        0.0686,  # V
        0.0109,  # W
        0.0292,  # Y
    ]
)
NATURAL_BACKGROUND = NATURAL_BACKGROUND / NATURAL_BACKGROUND.sum()

#: Eisenberg consensus hydrophobicity scale, used for the helical moment.
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

NUCLEOTIDE_CHARS = set("ACGTUN")


class SequenceError(ValueError):
    """Raised when a sequence cannot be used as a protein sequence."""


class HeterogeneousSetError(ValueError):
    """Raised when sequences given as one family are not actually homologous."""


class ModelNotBuiltError(RuntimeError):
    """Raised when a profile-dependent operation runs before the profile exists."""


@dataclass(frozen=True)
class CleanSequence:
    """A validated protein sequence plus a record of what was changed."""

    seq: str
    name: str = "query"
    original_length: int = 0
    notes: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def x_fraction(self) -> float:
        return self.seq.count("X") / max(len(self.seq), 1)


def clean_sequence(
    raw: str,
    name: str = "query",
    min_length: int = 30,
    max_x_fraction: float = 0.10,
) -> CleanSequence:
    """Validate and normalise a protein sequence.

    Handles the inputs that break naive pipelines: FASTA headers pasted along
    with the sequence, whitespace and line numbers, gap characters, trailing
    stop codons, lowercase, selenocysteine, and nucleotide sequences submitted
    by mistake.

    Raises SequenceError with an actionable message rather than failing later
    inside the aligner.
    """
    if raw is None:
        raise SequenceError("Sequence is None.")
    text = str(raw)

    # Drop FASTA headers if a whole record was pasted in.
    lines = [ln for ln in text.splitlines() if not ln.startswith(">")]
    text = "".join(lines)

    original_length = len(re.sub(r"[^A-Za-z]", "", text))
    notes: List[str] = []

    # Strip anything that is not a letter: whitespace, digits, gaps, '*', '.'.
    stripped = re.sub(r"[^A-Za-z]", "", text).upper()
    if not stripped:
        raise SequenceError(f"{name!r}: no residues found after cleaning.")

    # Nucleotide sequences are a common and silent mistake.
    nuc_fraction = sum(1 for c in stripped if c in NUCLEOTIDE_CHARS) / len(stripped)
    if nuc_fraction > 0.92 and len(stripped) >= 60:
        raise SequenceError(
            f"{name!r}: looks like a nucleotide sequence "
            f"({nuc_fraction:.0%} ACGTUN over {len(stripped)} characters). "
            "Translate it before screening."
        )

    out = []
    unknown = Counter()
    for c in stripped:
        if c in AA_INDEX:
            out.append(c)
        elif c in AMBIGUOUS_MAP:
            mapped = AMBIGUOUS_MAP[c]
            out.append(mapped)
            unknown[c] += 1
        elif c == "X":
            out.append("X")
            unknown["X"] += 1
        else:
            out.append("X")
            unknown[c] += 1

    seq = "".join(out)
    for ch, n in sorted(unknown.items()):
        tgt = AMBIGUOUS_MAP.get(ch, "X")
        notes.append(f"{n}x {ch!r} -> {tgt!r}")

    if len(seq) < min_length:
        raise SequenceError(
            f"{name!r}: only {len(seq)} residues; minimum is {min_length}. "
            "Composition statistics are meaningless on fragments this short."
        )

    x_frac = seq.count("X") / len(seq)
    if x_frac > max_x_fraction:
        raise SequenceError(
            f"{name!r}: {x_frac:.0%} unknown residues exceeds the "
            f"{max_x_fraction:.0%} limit."
        )

    return CleanSequence(
        seq=seq, name=name, original_length=original_length, notes=tuple(notes)
    )


def parse_fasta(path_or_text: str, from_text: bool = False) -> List[Tuple[str, str]]:
    """Parse FASTA from a path or a string. Handles wrapping, CRLF, blank lines.

    Transparently reads .gz. Returns [(name, sequence), ...] in file order.

    Comment lines - the original FASTA ';' and the common '#' - are dropped.
    Without that, a leading comment block is taken for a headerless sequence
    and emitted as a record named 'seq1': a fabricated sequence made of prose
    that then fails to clean, or worse, cleans into something. No real residue
    line can begin with either character, so dropping them is unambiguous.
    """
    if from_text:
        handle: Iterable[str] = path_or_text.splitlines()
    elif str(path_or_text).endswith(".gz"):
        handle = gzip.open(path_or_text, "rt", encoding="utf-8", errors="replace")
    else:
        handle = open(path_or_text, "r", encoding="utf-8", errors="replace")

    records: List[Tuple[str, str]] = []
    name: Optional[str] = None
    chunks: List[str] = []
    try:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line or line[0] in ";#":
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].strip() or f"seq{len(records) + 1}"
                chunks = []
            else:
                if name is None:  # headerless input
                    name = "seq1"
                chunks.append(line.strip())
        if name is not None:
            records.append((name, "".join(chunks)))
    finally:
        if hasattr(handle, "close"):
            handle.close()  # type: ignore[union-attr]
    return records


def load_sequences(
    path_or_text: str,
    from_text: bool = False,
    min_length: int = 30,
    skip_invalid: bool = True,
) -> List[CleanSequence]:
    """Parse and clean a FASTA file, logging and skipping unusable records."""
    out: List[CleanSequence] = []
    n_skipped = 0
    for name, seq in parse_fasta(path_or_text, from_text=from_text):
        try:
            out.append(clean_sequence(seq, name=name, min_length=min_length))
        except SequenceError as exc:
            n_skipped += 1
            if not skip_invalid:
                raise
            LOG.debug("skipping %s: %s", name, exc)
    if n_skipped:
        LOG.info("Skipped %d unusable record(s) of %d.", n_skipped, n_skipped + len(out))
    if not out:
        raise SequenceError("No usable sequences found.")
    return out


def composition(seq: str, pseudocount: float = 0.5) -> np.ndarray:
    """Return a length-20 frequency vector. X residues are excluded."""
    counts = np.full(20, pseudocount, dtype=float)
    for c in seq:
        idx = AA_INDEX.get(c)
        if idx is not None:
            counts[idx] += 1.0
    return counts / counts.sum()


# ===========================================================================
# 2. Generator fingerprints
# ===========================================================================


@dataclass(frozen=True)
class GeneratorFingerprint:
    """Expected residue composition of a sequence-design model.

    `freqs` is a length-20 vector over AA order 'ACDEFGHIKLMNPQRSTVWY'.
    `confidence` records how much to trust the prior: 'literature-prior' values
    are approximate and should be replaced by `fit_fingerprint()` on real output.
    """

    name: str
    freqs: np.ndarray
    confidence: str = "literature-prior"
    notes: str = ""
    n_sequences: int = 0

    def llr_per_residue(self, seq_comp: np.ndarray, reference: np.ndarray) -> float:
        """Mean per-residue log-odds of this generator vs a reference composition.

        Positive => composition looks more like this generator than like the
        reference. Scale is nats per residue, so it is length-independent and
        comparable across sequences.
        """
        p = np.clip(self.freqs, 1e-6, None)
        q = np.clip(reference, 1e-6, None)
        return float(np.sum(seq_comp * (np.log(p) - np.log(q))))


def _fp(name: str, mapping: Dict[str, float], **kw) -> GeneratorFingerprint:
    v = np.array([mapping[a] for a in AA], dtype=float)
    return GeneratorFingerprint(name=name, freqs=v / v.sum(), **kw)


#: ProteinMPNN at low sampling temperature (T~0.1), soluble monomeric designs.
#: APPROXIMATE literature-derived prior. Recalibrate on your own output.
PROTEINMPNN_PRIOR = _fp(
    "proteinmpnn",
    {
        "A": 0.105, "C": 0.002, "D": 0.055, "E": 0.118, "F": 0.033,
        "G": 0.050, "H": 0.013, "I": 0.060, "K": 0.098, "L": 0.115,
        "M": 0.012, "N": 0.035, "P": 0.035, "Q": 0.030, "R": 0.065,
        "S": 0.040, "T": 0.045, "V": 0.070, "W": 0.007, "Y": 0.025,
    },
    notes=(
        "Strong E/K/A/L/R enrichment, near-total cysteine avoidance, "
        "M/W/H depletion. Blunted at higher sampling temperature."
    ),
)

#: ProteinMPNN composition MEASURED on real output, not taken from the
#: literature. Fitted on 1080 designs from 6 natural kinase catalytic-domain
#: backbones (cdk2, cdk8, cmet, p38, syk, tyk2), vanilla and soluble weights,
#: sampling temperatures 0.1, 0.2, 0.3.
#:
#: This is a better default than PROTEINMPNN_PRIOR, which disagrees with it
#: in DIRECTION on D, G, P, W and Y - the published descriptions the prior
#: came from are of de novo helical bundles, and redesigning a natural
#: backbone is a different distribution.
#:
#: It is still conditioned on these six structures. ProteinMPNN follows the
#: burial pattern it is given, and across the six backbones the per-residue
#: frequency moves by up to 0.040 (K), 0.039 (E), 0.036 (T). Refit with
#: fit_fingerprint() on output from your own targets wherever you can.
PROTEINMPNN_MEASURED = _fp(
    "proteinmpnn_measured",
    {
        "A": 0.0582, "C": 0.0069, "D": 0.0651, "E": 0.1062, "F": 0.0339,
        "G": 0.0670, "H": 0.0184, "I": 0.0532, "K": 0.0836, "L": 0.1286,
        "M": 0.0149, "N": 0.0287, "P": 0.0739, "Q": 0.0161, "R": 0.0496,
        "S": 0.0429, "T": 0.0345, "V": 0.0590, "W": 0.0133, "Y": 0.0459,
    },
    confidence="measured",
    n_sequences=1080,
    notes=(
        "Measured on real ProteinMPNN output over six kinase backbones at "
        "T=0.1/0.2/0.3. P, W and Y come out enriched, where the "
        "literature prior has them flat or depleted."
    ),
)

#: ESM-IF1 - similar charged-surface bias, less extreme cysteine avoidance.
ESM_IF_PRIOR = _fp(
    "esm_if",
    {
        "A": 0.095, "C": 0.006, "D": 0.058, "E": 0.100, "F": 0.037,
        "G": 0.058, "H": 0.018, "I": 0.058, "K": 0.085, "L": 0.105,
        "M": 0.016, "N": 0.040, "P": 0.040, "Q": 0.034, "R": 0.058,
        "S": 0.048, "T": 0.050, "V": 0.066, "W": 0.009, "Y": 0.028,
    },
    notes="Less extreme than ProteinMPNN; retains more polar/loop residues.",
)

#: Idealised de novo scaffolds (Rosetta/hallucination helical bundles, Top7-like).
#: Dominated by the designed-helix alphabet rather than by a network's prior.
IDEALIZED_DENOVO_PRIOR = _fp(
    "idealized_denovo",
    {
        "A": 0.120, "C": 0.002, "D": 0.045, "E": 0.135, "F": 0.025,
        "G": 0.055, "H": 0.008, "I": 0.065, "K": 0.130, "L": 0.120,
        "M": 0.008, "N": 0.030, "P": 0.025, "Q": 0.025, "R": 0.045,
        "S": 0.035, "T": 0.040, "V": 0.060, "W": 0.005, "Y": 0.022,
    },
    notes="Extreme E/K/A/L; repeated heptads; frequent GS linkers.",
)

FINGERPRINTS: Dict[str, GeneratorFingerprint] = {
    fp.name: fp
    for fp in (PROTEINMPNN_PRIOR, PROTEINMPNN_MEASURED, ESM_IF_PRIOR,
               IDEALIZED_DENOVO_PRIOR)
}

#: Why the vague literature prior and not the measured one is still the default.
#:
#: PROTEINMPNN_MEASURED is a real measurement and it beats PROTEINMPNN_PRIOR on
#: the output it was measured from: +5.0 points of detection, winning 5 of 6
#: leave-one-backbone-out folds on real ProteinMPNN designs at T <= 0.3. On
#: output from the SAME tool at T >= 0.5 it loses to the prior by 7.5 points,
#: winning 1 of 6. A sharp fingerprint fitted at one temperature is confidently
#: wrong at another, where a vague one is merely uninformative - and a detector
#: shipped to someone whose sampling temperature is unknown should fail the
#: second way. So the default stays vague, and the measured one is there to be
#: chosen deliberately:
#:
#:     Detector(profile=prof, fingerprint=md.PROTEINMPNN_MEASURED)
#:
#: Better still, measure your own with fit_fingerprint() at the temperature you
#: expect to see. That beats both.
DEFAULT_FINGERPRINT = PROTEINMPNN_PRIOR


def register_fingerprint(fp: GeneratorFingerprint) -> None:
    """Add or replace a generator fingerprint in the global registry."""
    FINGERPRINTS[fp.name] = fp
    LOG.info("Registered fingerprint %r (%s, n=%d).", fp.name, fp.confidence, fp.n_sequences)


def fit_fingerprint(
    name: str, sequences: Iterable[str], pseudocount: float = 1.0
) -> GeneratorFingerprint:
    """Measure a generator's composition from real output. Strongly preferred
    over the shipped priors: 200+ sequences gives a solid estimate."""
    counts = np.full(20, pseudocount, dtype=float)
    n = 0
    for s in sequences:
        cs = clean_sequence(s, name=f"{name}_{n}", min_length=1) if isinstance(s, str) else s
        for c in cs.seq:
            idx = AA_INDEX.get(c)
            if idx is not None:
                counts[idx] += 1.0
        n += 1
    if n < 20:
        LOG.warning(
            "Fitting fingerprint %r on only %d sequences; estimates will be noisy.",
            name, n,
        )
    return GeneratorFingerprint(
        name=name,
        freqs=counts / counts.sum(),
        confidence="fitted",
        notes=f"Fitted on {n} sequences.",
        n_sequences=n,
    )


def best_generator_match(
    seq_comp: np.ndarray, reference: np.ndarray
) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Score the query composition against every registered fingerprint."""
    scores = {
        fp.name: fp.llr_per_residue(seq_comp, reference)
        for fp in FINGERPRINTS.values()
    }
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.__getitem__)
    return best, scores[best], scores


# ===========================================================================
# 3. Complexity, idealization and construct motifs
# ===========================================================================

CONSTRUCT_MOTIFS: List[Tuple[str, str, str]] = [
    ("his_tag", r"H{6,}", "polyhistidine purification tag"),
    ("gs_linker", r"(?:GGGGS){2,}|(?:GGS){3,}|GGSGGS", "flexible GS linker"),
    ("tev_site", r"ENLYFQ[GS]?", "TEV protease cleavage site"),
    ("thrombin_site", r"LVPR[G]S", "thrombin cleavage site"),
    ("strep_tag", r"WSHPQFEK", "Strep-tag II"),
    ("flag_tag", r"DYKDDDDK", "FLAG tag"),
    ("sortase_tag", r"LPETG[G]?", "sortase A recognition motif"),
    ("avi_tag", r"GLNDIFEAQKIEWHE", "AviTag biotinylation sequence"),
    ("myc_tag", r"EQKLISEEDL", "c-Myc tag"),
]
_COMPILED_MOTIFS = [(k, re.compile(p), d) for k, p, d in CONSTRUCT_MOTIFS]


def find_construct_motifs(seq: str) -> List[Tuple[str, str, int]]:
    """Return (motif_key, description, position) for each construct motif found.

    These are evidence of an expression construct, not of design per se, but a
    database-derived natural sequence essentially never carries them.
    """
    found = []
    for key, rx, desc in _COMPILED_MOTIFS:
        for m in rx.finditer(seq):
            found.append((key, desc, m.start()))
    return found


def shannon_entropy(seq: str) -> float:
    """Residue-composition entropy in bits (max log2(20) = 4.32)."""
    counts = Counter(c for c in seq if c in AA_INDEX)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=float) / total
    return float(-(p * np.log2(p)).sum())


def max_homopolymer_run(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def kmer_repetition(seq: str, k: int = 6) -> float:
    """Fraction of k-mer occurrences that are repeats of an earlier k-mer.

    Idealised repeat scaffolds (helical bundles built from a repeated heptad)
    score near 1; natural globular proteins sit well below 0.2.
    """
    if len(seq) < k + 1:
        return 0.0
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    distinct = len(set(kmers))
    return 1.0 - distinct / len(kmers)


def hydrophobic_moment(seq: str, window: int = 18, angle_deg: float = 100.0) -> float:
    """Maximum Eisenberg hydrophobic moment (per residue) over sliding windows.

    De novo amphipathic helical bundles reach values natural globular proteins
    rarely sustain, because the designed helix faces are perfectly segregated.
    """
    if len(seq) < window:
        window = len(seq)
    if window < 5:
        return 0.0
    delta = math.radians(angle_deg)
    h = np.array([EISENBERG.get(c, 0.0) for c in seq])
    best = 0.0
    for start in range(0, len(seq) - window + 1):
        w = h[start : start + window]
        idx = np.arange(window) * delta
        mu = math.hypot(float((w * np.cos(idx)).sum()), float((w * np.sin(idx)).sum()))
        best = max(best, mu / window)
    return best


def sequence_identity(a: str, b: str) -> float:
    """Ungapped-anchored identity via a banded Needleman-Wunsch.

    Used only for nearest-neighbour screening against a family set, so speed
    matters more than optimal gap handling. Returns identity over the aligned
    length of the shorter sequence.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    band = max(32, abs(m - n) + 16)
    gap = -1.0
    prev = np.full(m + 1, -np.inf)
    prev_id = np.zeros(m + 1)
    prev[0] = 0.0
    for j in range(1, min(band, m) + 1):
        prev[j] = gap * j
    for i in range(1, n + 1):
        cur = np.full(m + 1, -np.inf)
        cur_id = np.zeros(m + 1)
        lo, hi = max(1, i - band), min(m, i + band)
        if lo == 1:
            cur[0] = gap * i
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            match = 1.0 if ai == b[j - 1] else -1.0
            diag = prev[j - 1] + match
            up = prev[j] + gap
            left = cur[j - 1] + gap
            if diag >= up and diag >= left:
                cur[j] = diag
                cur_id[j] = prev_id[j - 1] + (1.0 if ai == b[j - 1] else 0.0)
            elif up >= left:
                cur[j] = up
                cur_id[j] = prev_id[j]
            else:
                cur[j] = left
                cur_id[j] = cur_id[j - 1]
        prev, prev_id = cur, cur_id
    return float(prev_id[m] / n)


def kmer_set(seq: str, k: int = 4) -> frozenset:
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def kmer_jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ===========================================================================
# 4. UniProt retrieval (robust: retries, backoff, pagination, caching, caps)
# ===========================================================================

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
USER_AGENT = f"mpnn_detect/{__version__} (protein design screening)"
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _implausible_records(records: Sequence[Tuple[str, str]]) -> Optional[str]:
    """Describe why a set of FASTA records cannot be real protein, or None.

    Used to screen cached files and freshly parsed bodies. The checks target
    what mis-decoded binary actually looks like: a handful of records, absurd
    lengths, and a residue alphabet full of characters no protein contains.
    """
    if not records:
        return "it holds no records"
    lengths = [len(s) for _name, s in records]
    if max(lengths) > 10000:
        return f"a {max(lengths)}-residue record (no natural protein is that long)"
    joined = "".join(s for _n, s in records[:50])
    if not joined:
        return "every record is empty"
    good = sum(1 for c in joined.upper() if c in AA or c in "XUOBZJ*-. \t")
    frac = good / len(joined)
    if frac < 0.9:
        return f"only {frac:.0%} of characters are amino acids (binary, not FASTA?)"
    return None


def build_uniprot_query(
    term: str,
    min_length: int = 50,
    max_length: int = 1000,
    reviewed: bool = True,
    exclude_fragments: bool = True,
    taxonomy_id: Optional[int] = None,
) -> str:
    """Build a UniProtKB query that actually returns the intended family.

    The original notebook used `family:"KINASE"`, which searches the protein
    family annotation field for a literal string that is not a family name.
    This widens the search sensibly:

      * a Pfam accession (PF00069) becomes an `xref:pfam-` lookup - the precise
        and recommended way to pull a domain family;
      * anything else is matched against both `family` and `protein_name`.
    """
    term = term.strip()
    if re.fullmatch(r"PF\d{5}", term, flags=re.IGNORECASE):
        core = f"(xref:pfam-{term.upper()})"
    elif re.fullmatch(r"IPR\d{6}", term, flags=re.IGNORECASE):
        core = f"(xref:interpro-{term.upper()})"
    else:
        esc = term.replace('"', '\\"')
        core = f'(family:"{esc}" OR protein_name:"{esc}")'

    parts = [core, f"(length:[{int(min_length)} TO {int(max_length)}])"]
    if reviewed:
        parts.append("(reviewed:true)")
    if exclude_fragments:
        parts.append("(fragment:false)")
    if taxonomy_id is not None:
        parts.append(f"(taxonomy_id:{int(taxonomy_id)})")
    return " AND ".join(parts)


def _decompress_fully(raw: bytes, max_rounds: int = 4) -> bytes:
    """Decompress until the body stops looking compressed.

    A single pass is not enough. UniProt compresses the payload and the
    transport layer compresses again, so one `gzip.decompress` leaves a body
    that is STILL gzip - and that got decoded as text, producing mojibake that
    parsed into a handful of junk records. The symptom was a report of "14
    retrieved, 14 unusable" for a query that should return thousands.

    Also handles zlib and raw-deflate bodies, which some proxies substitute for
    gzip. Each attempt is bounded, and whatever comes out is still validated as
    FASTA by the caller - so an unexpected encoding fails loudly rather than
    becoming a thin-looking query.
    """
    for _ in range(max_rounds):
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
                continue
            except (OSError, EOFError, zlib.error):
                return raw
        if raw[:1] == b"\x78":  # zlib header
            try:
                raw = zlib.decompress(raw)
                continue
            except zlib.error:
                return raw
        break
    return raw


def _http_get(url: str, timeout: float, attempts: int, backoff: float) -> Tuple[bytes, Dict[str, str]]:
    """GET with retry, exponential backoff with jitter, and Retry-After support."""
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Decide by the gzip magic bytes rather than by matching the
                # Content-Encoding header. Header casing and whitespace vary
                # between servers and proxies, and a missed match here does not
                # fail: the compressed bytes get decoded as text, yielding
                # mojibake that still parses as a handful of junk FASTA records
                # which are then silently discarded downstream. Failing to
                # notice compression must not look like "the query returned
                # almost nothing".
                raw = _decompress_fully(raw)
                return raw, dict(resp.headers)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in _RETRYABLE_STATUS or attempt == attempts:
                raise RuntimeError(
                    f"UniProt returned HTTP {exc.code} ({exc.reason}) for {url}"
                ) from exc
            wait = float(exc.headers.get("Retry-After") or backoff * 2 ** (attempt - 1))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt == attempts:
                raise RuntimeError(
                    f"Network failure after {attempts} attempts: {exc}. "
                    "Pass natural_fasta=<local file> to run offline."
                ) from exc
            wait = backoff * 2 ** (attempt - 1)
        wait *= 1.0 + random.random() * 0.3
        LOG.warning("UniProt attempt %d/%d failed (%s); retrying in %.1fs.",
                    attempt, attempts, last, wait)
        time.sleep(wait)
    raise RuntimeError(f"Unreachable: {last}")


def fetch_uniprot(
    query: str,
    max_sequences: int = 3000,
    page_size: int = 500,
    timeout: float = 90.0,
    attempts: int = 4,
    backoff: float = 2.0,
    cache_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Download sequences for a UniProt query, paginated and capped.

    Returns [(name, sequence)]. Caches by query hash so re-running a notebook
    cell does not re-download. Raises RuntimeError with a readable message on
    failure instead of silently writing an empty file.
    """
    # The cache key carries a format version. A cache written by a build with
    # the double-gzip bug holds garbage, and on Drive it survives every later
    # run - so the fix above would never have fired for anyone who had already
    # cached a bad response. Bumping the version retires those entries instead
    # of trusting them.
    key = hashlib.sha256(f"v2|{query}|{max_sequences}".encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"uniprot_{key}.fasta.gz") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        recs = parse_fasta(cache_path)
        bad = _implausible_records(recs)
        if bad:
            LOG.warning(
                "Ignoring cached %s: %s. Re-fetching.", os.path.basename(cache_path), bad
            )
            try:
                os.replace(cache_path, cache_path + ".rejected")
            except OSError:
                pass
        else:
            LOG.info("Loaded %d cached sequences for query hash %s.", len(recs), key)
            return recs

    params = {
        "query": query,
        "format": "fasta",
        "size": str(min(page_size, max_sequences)),
    }
    url: Optional[str] = f"{UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}"
    chunks: List[str] = []
    total_reported: Optional[str] = None
    n = 0
    page = 0

    while url and n < max_sequences:
        page += 1
        body, headers = _http_get(url, timeout, attempts, backoff)
        total_reported = headers.get("x-total-results", total_reported)
        text = body.decode("utf-8", errors="replace")
        stripped = text.lstrip()
        if stripped and not stripped.startswith(">"):
            raise RuntimeError(
                "UniProt returned a body that is not FASTA (first bytes: "
                f"{stripped[:40]!r}). This usually means the response was "
                "encoded in a way that was not decoded - rerun, and if it "
                "persists report the first bytes above. Continuing would "
                "silently parse the garbage into a few junk records and look "
                "like an unproductive query."
            )
        chunks.append(text)
        # Count by parsing, not by counting "\n>": the two disagreeing is the
        # symptom that a malformed body is being accepted.
        page_records = len(parse_fasta(text, from_text=True)) if stripped else 0
        n += page_records
        link = headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        LOG.info("UniProt page %d: +%d sequences (%d so far)%s.",
                 page, page_records, n,
                 f" of {total_reported} matching" if total_reported else "")

    records = parse_fasta("".join(chunks), from_text=True)[:max_sequences]

    bad = _implausible_records(records)
    if records and bad:
        raise RuntimeError(
            f"UniProt returned something that is not protein FASTA ({bad}). "
            "The response was probably encoded in a way this client did not "
            "decode. Nothing was cached. Rerun; if it persists, report this "
            f"message.\nQuery: {query}"
        )
    if not records:
        raise RuntimeError(
            f"UniProt returned 0 sequences for query:\n  {query}\n"
            "Check the family term (try a Pfam accession such as PF00069), "
            "widen the length range, or set reviewed=False."
        )
    LOG.info("Retrieved %d sequences.", len(records))
    if len(records) < 50:
        LOG.warning(
            "Only %d sequences matched. A profile needs on the order of "
            "hundreds to be meaningful. Widen the query: try reviewed=False, a "
            "broader length range, or a Pfam accession instead of a name. "
            "Query was: %s", len(records), query,
        )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
            for nm, sq in records:
                fh.write(f">{nm}\n{sq}\n")
    return records


# ===========================================================================
# 5. Dereplication
# ===========================================================================


@dataclass
class FamilyFetch:
    """What assembling a family actually did, so the caller can report it."""

    sequences: List[CleanSequence]
    source: str                      #: "UniProt (PF00069)", "local file (x.fa)"
    stages: List[Dict[str, object]] = field(default_factory=list)
    n_raw: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.sequences)

    def report(self) -> str:
        lines = []
        for st in self.stages:
            lines.append(
                f"  {st['stage']}: retrieved {st['retrieved']}, "
                f"{st['unclean']} unusable, {st['wrong_length']} outside the "
                f"length window, {st['added']} new -> {st['total']} total"
            )
        if self.sequences:
            L = [len(s) for s in self.sequences]
            lines.append(f"  {len(self.sequences)} sequences from {self.source} "
                         f"(lengths {min(L)}-{max(L)})")
        return "\n".join(lines)


def fetch_family(
    term: str,
    target: int = 3000,
    min_length: int = 150,
    max_length: int = 400,
    min_usable: int = 120,
    natural_fasta: Optional[str] = None,
    cache_dir: Optional[str] = None,
    identity_threshold: float = 0.90,
    taxonomy_id: Optional[int] = None,
    seed: int = 0,
) -> FamilyFetch:
    """Assemble a natural family: reviewed entries first, then unreviewed.

    This lived in a notebook cell twice, and broke the same way twice. The
    failure is quiet and expensive: SwissProt holds only a few hundred reviewed
    protein kinases, so stopping after the reviewed pass yields ~433 sequences
    against a ~312-position model. Meff/M then lands near 0.5, the Potts fit
    declines to mutual information, the null is too small to place alpha=0.01,
    and every downstream number is weaker - while the run still completes and
    prints a plausible-looking result. The second occurrence was a one-word
    short-circuit that skipped the top-up immediately after announcing it.

    Notebook-only control flow is untested control flow. It is a module
    function now so that a test can hold it to its contract:

      * a local `natural_fasta` short-circuits the network entirely;
      * a SUCCESSFUL reviewed pass must NOT stop the unreviewed top-up - only
        reaching `target`, or running out of stages, ends it.
    """
    if natural_fasta:
        seqs = [s for s in load_sequences(natural_fasta, min_length=min_length)
                if min_length <= len(s) <= max_length]
        if len(seqs) < min_usable:
            raise SequenceError(
                f"{natural_fasta} gave {len(seqs)} usable sequences in "
                f"{min_length}-{max_length} aa; at least {min_usable} are "
                "needed to build a profile, fit references and hold out a null."
            )
        n_raw = len(seqs)
        seqs = dereplicate(seqs, identity_threshold=identity_threshold,
                           max_keep=target, seed=seed)
        return FamilyFetch(sequences=seqs, n_raw=n_raw,
                           source=f"local file ({os.path.basename(natural_fasta)})")

    family: List[CleanSequence] = []
    seen: set = set()
    stages: List[Dict[str, object]] = []
    source = ""
    for reviewed in (True, False):
        # Only the target, or exhausting the stages, ends this loop. In
        # particular NOT "we already have a usable family" - that is what
        # reduced a 3,544-sequence family to 433.
        if len(family) >= target:
            break
        query = build_uniprot_query(term, min_length=min_length,
                                    max_length=max_length, reviewed=reviewed,
                                    taxonomy_id=taxonomy_id)
        try:
            records = fetch_uniprot(query, max_sequences=target * 2,
                                    cache_dir=cache_dir)
        except Exception as exc:
            LOG.warning("UniProt request failed (%s: %s); stopping with %d "
                        "sequences.", type(exc).__name__, exc, len(family))
            stages.append({"stage": f"reviewed={reviewed}", "retrieved": 0,
                           "unclean": 0, "wrong_length": 0, "added": 0,
                           "total": len(family), "error": str(exc)})
            break
        unclean = wrong_length = added = 0
        for name, seq in records:
            try:
                cs = clean_sequence(seq, name=name.split()[0],
                                    min_length=min_length)
            except SequenceError:
                unclean += 1
                continue
            if not (min_length <= len(cs) <= max_length):
                wrong_length += 1
                continue
            if cs.name in seen:
                continue
            seen.add(cs.name)
            family.append(cs)
            added += 1
        stages.append({"stage": f"reviewed={reviewed}", "retrieved": len(records),
                       "unclean": unclean, "wrong_length": wrong_length,
                       "added": added, "total": len(family)})
        if len(family) >= min_usable:
            source = f"UniProt ({term})"
        if reviewed and len(family) < target:
            LOG.info("Reviewed entries gave %d of the %d target; topping up "
                     "with unreviewed entries.", len(family), target)

    if len(family) < min_usable:
        return FamilyFetch(sequences=[], source="", stages=stages,
                           n_raw=len(family))
    n_raw = len(family)
    family = dereplicate(family, identity_threshold=identity_threshold,
                         max_keep=target, seed=seed)
    return FamilyFetch(sequences=family, source=source, stages=stages,
                       n_raw=n_raw)


def dereplicate(
    sequences: Sequence[CleanSequence],
    identity_threshold: float = 0.90,
    max_keep: int = 1500,
    k: int = 4,
    seed: int = 0,
) -> List[CleanSequence]:
    """Greedy redundancy filter so the profile is not dominated by near-duplicates.

    Uses k-mer Jaccard as a cheap proxy for identity (Jaccard ~0.45 corresponds
    roughly to 90% identity for k=4). This matters: UniProt families are full of
    near-identical orthologues, and an unfiltered profile encodes the most
    sequenced organism rather than the family.

    `max_keep` subsamples at random AFTER dereplication, not by stopping the
    traversal early. Stopping early is the obvious implementation and it is
    wrong: the traversal goes longest-first, which is the right way to pick a
    cluster representative but means an early stop returns the longest
    `max_keep` sequences. Measured on 3,544 real PF00069 kinases spanning
    150-400 aa, asking for 1,400 returned sequences of 361-400 aa only. Every
    downstream comparison then has a length confound, and against design
    sequences - which are a fixed backbone length - that confound reads as
    near-perfect detection: 99.9% with half the design backbones' own NATIVE
    sequences also called designed.
    """
    if not sequences:
        raise SequenceError(
            "dereplicate() received no sequences. Returning an empty list here "
            "would push the problem downstream, where it surfaces as an "
            "unrelated error such as min() on an empty iterable. Check how many "
            "sequences survived retrieval and length filtering."
        )
    jaccard_cut = max(0.05, identity_threshold ** (k * 1.6))
    # Longest-first is the right order for choosing representatives, but the
    # RESULT is returned in the caller's original order. Returning it sorted by
    # length is a quiet trap: a caller who then splits the list positionally
    # into train/reference/null gets sets stratified by sequence length rather
    # than random ones, and every downstream statistic is biased by it.
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i].seq))
    kept_idx: List[int] = []

    # An inverted k-mer index, so each sequence is only ever compared with the
    # ones it shares k-mers with. Writing out the Jaccard condition,
    #
    #     J = c / (a + b - c) >= t    <=>    c >= t*(a + b) / (1 + t)
    #
    # where c = |A & B| and a, b are the set sizes. Counting c directly off the
    # postings makes this EXACT - no set intersections, and the kept set is
    # identical to the naive all-pairs scan's, which a test checks.
    #
    # This matters because the length-bias fix below had to drop the early stop
    # at max_keep, turning an O(n x max_keep) pass into O(n x clusters). On
    # 11,993 real UniProt kinases the all-pairs version took 115s, and on a
    # family with little redundancy it is quadratic. Bounding by set size
    # alone does not help: kinase domains in a 150-400 aa window all have
    # similar k-mer counts, so that band admits nearly everything.
    postings: Dict[str, List[int]] = {}
    sizes: List[int] = []
    ratio = jaccard_cut / (1.0 + jaccard_cut)
    for i in order:
        ks = kmer_set(sequences[i].seq, k)
        a = len(ks)
        if not a:
            continue
        shared: Dict[int, int] = {}
        for km in ks:
            for j in postings.get(km, ()):
                shared[j] = shared.get(j, 0) + 1
        if any(c >= ratio * (a + sizes[j]) for j, c in shared.items()):
            continue
        kept_idx.append(i)
        j = len(sizes)
        sizes.append(a)
        for km in ks:
            postings.setdefault(km, []).append(j)
    n_clusters = len(kept_idx)
    if n_clusters > max_keep:
        kept_idx = random.Random(seed).sample(kept_idx, max_keep)
    kept = [sequences[i] for i in sorted(kept_idx)]
    lengths = [len(s.seq) for s in kept]
    LOG.info(
        "Dereplication: %d -> %d sequences (<%.0f%% identity)%s; "
        "lengths %d-%d, median %d.",
        len(sequences), len(kept), identity_threshold * 100,
        f", subsampled at random from {n_clusters} clusters"
        if n_clusters > max_keep else "",
        min(lengths), max(lengths), int(np.median(lengths)),
    )
    return kept


# ===========================================================================
# 6. Family profile HMM
# ===========================================================================


@dataclass
class ProfileHit:
    """Result of scoring one sequence against the family profile."""

    bits: float
    evalue: float
    bits_per_position: float
    coverage: float
    #: (model_position, target_index_0based, residue) for every aligned residue.
    aligned_positions: List[Tuple[int, int, str]] = field(default_factory=list)
    matched: bool = True


#: Seven chemical classes, used to screen column pairs for covariation cheaply
#: before computing exact 20x20 joint distributions on the survivors.
REDUCED_CLASSES = {
    **{a: 0 for a in "AVLIMC"},   # hydrophobic
    **{a: 1 for a in "FWY"},      # aromatic
    **{a: 2 for a in "STNQ"},     # polar
    **{a: 3 for a in "KRH"},      # positive
    **{a: 4 for a in "DE"},       # negative
    "G": 5,
    "P": 6,
}
N_REDUCED = 7
REDUCED_INDEX = np.array([REDUCED_CLASSES[a] for a in AA], dtype=np.int8)


class FamilyProfile:
    """A profile HMM for one natural protein family, plus residual analysis.

    Built without MAFFT: a seed HMM from the most central sequence, then
    iterative `hmmalign` + rebuild. This removes the apt-get dependency entirely
    and is far more robust than aligning thousands of full-length sequences with
    `mafft --auto`.
    """

    def __init__(self, name: str = "family", random_state: int = 0):
        if not HAVE_PYHMMER:
            raise ModelNotBuiltError(
                "pyhmmer is required for profile signals. pip install pyhmmer"
            )
        self.name = name
        # Everything stochastic in this class draws from here. Using the global
        # random module made two builds on identical input produce different
        # profiles, which is unacceptable for a screening tool: a borderline
        # call could change between runs for no reason the user could see.
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self.alphabet = Alphabet.amino()
        self.background = Background(self.alphabet)
        self.hmm = None
        self.n_sequences = 0
        self.information_content: Optional[np.ndarray] = None  # (M,) bits
        self.emissions: Optional[np.ndarray] = None  # (M, 20)
        self.variable_composition: Optional[np.ndarray] = None  # (20,)
        self.family_composition: Optional[np.ndarray] = None  # (20,)
        self._members: List[CleanSequence] = []
        self._member_kmers: List[frozenset] = []
        self._member_matrix: Optional[np.ndarray] = None
        self.coupling: Optional[dict] = None
        self.evo: Optional[dict] = None
        self.placement: Optional[dict] = None
        self.meff: Optional[float] = None
        self._member_inserts: Optional[np.ndarray] = None
        self.burial: Optional[np.ndarray] = None
        self.exposed_mask: Optional[np.ndarray] = None
        self.buried_mask: Optional[np.ndarray] = None
        self.exposed_composition: Optional[np.ndarray] = None
        self.buried_composition: Optional[np.ndarray] = None
        self.feature_reference: Dict[str, Tuple[float, float]] = {}
        self.feature_reference_n = 0
        self.feature_reference_holdout = False
        self._reference_ids: set = set()
        self._score_cache: Dict[str, "ProfileHit"] = {}
        self._cache_limit = 4096
        self._bg = np.array(self.background.residue_frequencies)[:20]

    # -- construction -------------------------------------------------------

    def _digital(self, seqs: Sequence[CleanSequence]):
        return [
            TextSequence(name=s.name.encode()[:200], sequence=s.seq).digitize(self.alphabet)
            for s in seqs
        ]

    def _pick_seed(self, seqs: Sequence[CleanSequence], sample: int = 60) -> int:
        """Choose the most central sequence, not an arbitrary first one."""
        idx = list(range(len(seqs)))
        probe = idx if len(idx) <= sample else self._rng.sample(idx, sample)
        kms = {i: kmer_set(seqs[i].seq, 4) for i in probe}
        best_i, best_score = probe[0], -1.0
        for i in probe:
            score = sum(kmer_jaccard(kms[i], kms[j]) for j in probe if j != i)
            if score > best_score:
                best_i, best_score = i, score
        return best_i

    def build(
        self,
        sequences: Sequence[CleanSequence],
        iterations: int = 2,
        min_sequences: int = 5,
        min_homologous_fraction: float = 0.5,
        check_homogeneity: bool = True,
        drop_outliers: bool = True,
        coupling_method: str = "dca",
        symfrac: float = 0.6,
    ) -> "FamilyProfile":
        """Build the profile, refusing to build nonsense.

        `check_homogeneity` guards against the failure mode in the original
        notebook: aligning three structurally unrelated proteins and building an
        HMM from the result. A profile over unrelated sequences has no
        information content, and every downstream number computed from it is
        noise - including the bit scores that looked so convincing.

        The test is done with HMMER itself rather than a k-mer heuristic: build
        a seed model from the most central sequence and see how many of the
        others it actually recruits. Genuine homologues down to ~25% identity
        are recruited; unrelated folds are not.
        """
        if len(sequences) < min_sequences:
            raise HeterogeneousSetError(
                f"Need at least {min_sequences} sequences to build a profile; "
                f"got {len(sequences)}. A profile HMM over a handful of "
                "sequences just memorises them."
            )

        builder = Builder(self.alphabet, seed=42)
        digital = self._digital(sequences)
        seed_i = self._pick_seed(sequences)
        LOG.info("Seeding profile from %r (%d aa).",
                 sequences[seed_i].name, len(sequences[seed_i].seq))
        hmm, _, _ = builder.build(digital[seed_i], self.background)

        keep = list(range(len(sequences)))
        if check_homogeneity or drop_outliers:
            recruited, evalues = self._recruitment(hmm, digital)
            fraction = len(recruited) / len(sequences)
            LOG.info("Seed model recruits %d/%d sequences (%.0f%%).",
                     len(recruited), len(sequences), fraction * 100)
            if check_homogeneity and fraction < min_homologous_fraction:
                raise HeterogeneousSetError(
                    f"Only {fraction:.0%} of these {len(sequences)} sequences are "
                    f"detectably homologous to the most central one "
                    f"(threshold {min_homologous_fraction:.0%}). They are not one "
                    "family. A profile HMM built over unrelated folds has no "
                    "information content, so every score derived from it is "
                    "meaningless - including impressive-looking bit scores, "
                    "which in that situation only measure similarity to whichever "
                    "training sequence the query happens to resemble. Build one "
                    "profile per family, or pass check_homogeneity=False if you "
                    "have a reason to."
                )
            if drop_outliers and recruited and len(recruited) < len(sequences):
                LOG.info("Dropping %d non-homologous outlier(s) before building.",
                         len(sequences) - len(recruited))
                keep = recruited

        sequences = [sequences[i] for i in keep]
        digital = [digital[i] for i in keep]

        for it in range(1, iterations + 1):
            msa = pyhmmer.hmmer.hmmalign(hmm, digital)
            msa.name = self.name.encode()[:200]
            if symfrac > 0:
                msa = self._trim_alignment(msa, symfrac)
            hmm, _, _ = builder.build_msa(msa.digitize(self.alphabet), self.background)
            LOG.info("Iteration %d: M=%d positions, mean info %.2f bits.",
                     it, hmm.M, hmm.mean_match_information(self.background))

        # pyhmmer leaves command_line empty, which writes a malformed "COM"
        # record that HMMFile then refuses to read back. Set it explicitly.
        hmm.command_line = f"mpnn_detect {__version__} build (n={len(sequences)})"
        hmm.accession = None
        self.hmm = hmm
        self.n_sequences = len(sequences)
        self._members = list(sequences)
        self._member_kmers = [kmer_set(s.seq, 4) for s in sequences]
        self._compute_position_statistics()
        self._index_members()
        meff = float(self._sequence_weights(self._member_matrix).sum())
        self.meff = meff
        LOG.info(
            "Model: %d positions, %d members, Meff=%.0f (Meff/M=%.2f).",
            int(self.hmm.M), len(self._members), meff, meff / max(self.hmm.M, 1),
        )
        if self.hmm.M > 1.6 * np.median([len(s) for s in self._members]):
            LOG.warning(
                "The model is %d positions but the median member is only %d "
                "residues - the alignment is padded with sparsely occupied "
                "columns. Raise symfrac (currently %.2f) to trim them.",
                int(self.hmm.M), int(np.median([len(s) for s in self._members])),
                symfrac,
            )
        self.fit_covariation(method=coupling_method)
        self.fit_evolutionary_model()
        self.fit_placement_null()
        return self

    def _trim_alignment(self, msa, symfrac: float):
        """Drop alignment columns that few sequences actually occupy.

        This is what `hmmbuild --symfrac` does, and without it a family of
        full-length proteins produces a model far longer than the domain it is
        supposed to describe. Measured on 238 real PF00069 kinases: the
        untrimmed alignment was 1301 columns wide and produced a 643-position
        model for a domain that is about 260 positions long, because sparsely
        occupied flanking columns were promoted to match states.

        The cost of that is not cosmetic. Every per-position statistic - the
        emissions, the burial split, the site rates, the coupling model - is
        then estimated from a fraction of the data it appears to have, and the
        effective-sequences-per-position ratio that DCA depends on falls by the
        same factor.
        """
        aln = msa.alignment
        if not aln:
            return msa
        width = len(aln[0])
        occupied = np.zeros(width, dtype=np.int32)
        for row in aln:
            occupied += np.fromiter(
                (0 if c in "-." else 1 for c in row), dtype=np.int32, count=width
            )
        keep = np.flatnonzero(occupied >= symfrac * len(aln))
        if len(keep) < 20 or len(keep) == width:
            return msa
        names = [msa.sequences[i].name for i in range(len(aln))]
        trimmed = TextMSA(sequences=[
            TextSequence(
                name=names[i],
                sequence="".join(row[j] for j in keep).upper().replace(".", "-"),
            )
            for i, row in enumerate(aln)
        ])
        trimmed.name = self.name.encode()[:200]
        LOG.info("Trimmed alignment %d -> %d columns at symfrac=%.2f.",
                 width, len(keep), symfrac)
        return trimmed

    def _recruitment(self, hmm, digital, evalue_cut: float = 1e-3):
        """Indices of sequences the given model detects, plus their E-values."""
        pli = Pipeline(
            self.alphabet, background=self.background,
            E=1e6, domE=1e6, incE=1e6, incdomE=1e6, bias_filter=False,
        )
        block = DigitalSequenceBlock(self.alphabet, digital)
        hits = pli.search_hmm(hmm, block)
        by_name = {}
        for h in hits:
            nm = h.name.decode() if isinstance(h.name, bytes) else str(h.name)
            by_name[nm] = min(by_name.get(nm, float("inf")), float(h.evalue))
        recruited, evalues = [], []
        for i, d in enumerate(digital):
            nm = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
            e = by_name.get(nm, float("inf"))
            evalues.append(e)
            if e <= evalue_cut:
                recruited.append(i)
        return recruited, evalues

    def _compute_position_statistics(self) -> None:
        em = np.array(self.hmm.match_emissions)[1:, :20]
        em = em / np.clip(em.sum(axis=1, keepdims=True), 1e-12, None)
        self.emissions = em
        bg = np.clip(self._bg, 1e-12, None)
        self.information_content = (em * np.log2(np.clip(em, 1e-12, None) / bg)).sum(axis=1)

        ic = self.information_content
        lo = float(np.quantile(ic, 0.33))
        var_mask = ic <= lo
        if var_mask.sum() < 3:
            var_mask = ic <= float(np.median(ic))
        self.variable_composition = em[var_mask].mean(axis=0)
        self.family_composition = em.mean(axis=0)

        # Burial proxy: hydrophobicity expected under each position's own
        # emission distribution. Buried positions sit in the core and are
        # structurally determined, so an inverse-folding model reproduces them;
        # exposed positions are where it substitutes its own preferences. The
        # original single "variable position" bucket mixed the two, which blunts
        # the signal, because a variable buried position behaves nothing like a
        # variable surface position.
        hyd = np.array([EISENBERG[a] for a in AA])
        self.burial = em @ hyd
        b_lo = float(np.quantile(self.burial, 0.50))
        self.exposed_mask = (self.burial <= b_lo) & (ic <= float(np.quantile(ic, 0.50)))
        self.buried_mask = self.burial > float(np.quantile(self.burial, 0.67))
        if self.exposed_mask.sum() < 5:
            self.exposed_mask = self.burial <= b_lo
        self.exposed_composition = (
            em[self.exposed_mask].mean(axis=0)
            if self.exposed_mask.any() else self.variable_composition
        )
        self.buried_composition = (
            em[self.buried_mask].mean(axis=0)
            if self.buried_mask.any() else self.family_composition
        )
        LOG.info(
            "Profile statistics: %d positions, %d variable (IC<=%.2f bits), "
            "%d exposed-variable, %d buried.",
            len(ic), int(var_mask.sum()), lo,
            int(self.exposed_mask.sum()), int(self.buried_mask.sum()),
        )

    # -- scoring ------------------------------------------------------------

    def _pipeline(self) -> "Pipeline":
        # Permissive thresholds: we always want a score back, not a filtered
        # empty result that the caller mistakes for "score 0".
        return Pipeline(
            self.alphabet,
            background=self.background,
            E=1e9, domE=1e9, incE=1e9, incdomE=1e9,
            bias_filter=False, F1=1.0, F2=1.0, F3=1.0,
        )

    def score(self, cs: CleanSequence) -> ProfileHit:
        """Score one sequence, returning alignment detail for residual analysis.

        Memoised: a single prediction needs the alignment for the residual, the
        nearest-neighbour projection and the coupling check, and re-running
        hmmsearch three times for the same sequence is pure waste.
        """
        if self.hmm is None:
            raise ModelNotBuiltError("Profile not built. Call build() first.")
        cached = self._score_cache.get(cs.seq)
        if cached is not None:
            return cached
        q = TextSequence(name=b"query", sequence=cs.seq).digitize(self.alphabet)
        hits = self._pipeline().search_hmm(self.hmm, DigitalSequenceBlock(self.alphabet, [q]))
        if len(hits) == 0:
            miss = ProfileHit(
                bits=0.0, evalue=float("inf"), bits_per_position=0.0,
                coverage=0.0, matched=False,
            )
            self._remember(cs.seq, miss)
            return miss
        hit = max(hits, key=lambda h: h.score)
        dom = max(hit.domains, key=lambda d: d.score)
        aln = dom.alignment

        # Walk the alignment, tracking both coordinate systems.
        #   '.' in the model row  -> insertion: target consumes a residue,
        #                            the model does not advance.
        #   '-' in the target row -> deletion: the model advances, no residue.
        aligned: List[Tuple[int, int, str]] = []
        hmm_pos = aln.hmm_from
        tgt_pos = aln.target_from - 1  # pyhmmer reports 1-based target coords
        for hc, tc in zip(aln.hmm_sequence, aln.target_sequence):
            if hc == ".":
                tgt_pos += 1
                continue
            if tc != "-":
                aligned.append((hmm_pos, tgt_pos, tc.upper()))
                tgt_pos += 1
            hmm_pos += 1

        span = max(1, aln.hmm_to - aln.hmm_from + 1)
        result = ProfileHit(
            bits=float(hit.score),
            evalue=float(hit.evalue),
            bits_per_position=float(hit.score) / span,
            coverage=span / float(self.hmm.M),
            aligned_positions=aligned,
            matched=True,
        )
        self._remember(cs.seq, result)
        return result

    def _remember(self, key: str, hit: "ProfileHit") -> None:
        if len(self._score_cache) >= self._cache_limit:
            self._score_cache.clear()
        self._score_cache[key] = hit

    def null_z(self, cs: CleanSequence, n_shuffles: int = 30, seed: int = 0) -> float:
        """Z-score of the profile bit score against composition-matched shuffles.

        Shuffling preserves composition exactly, so this isolates the
        order/structure signal from the composition signal. It is also what makes
        scores comparable across profiles of different length - raw HMMER bits
        are not, which is why differencing two HMMs' bit scores (as the original
        notebook did) is not a log-odds ratio of anything.
        """
        observed = self.score(cs).bits
        rng = random.Random(seed)
        chars = list(cs.seq)
        null = []
        for _ in range(n_shuffles):
            rng.shuffle(chars)
            shuffled = CleanSequence(seq="".join(chars), name="shuffle")
            null.append(self.score(shuffled).bits)
        mu, sd = float(np.mean(null)), float(np.std(null))
        return (observed - mu) / sd if sd > 1e-9 else 0.0

    def residual_analysis(
        self, hit: ProfileHit, fingerprint: GeneratorFingerprint
    ) -> Dict[str, float]:
        """The core discriminative measurement.

        Splits the query's aligned residues into conserved (high information
        content) and variable (low IC) model positions, then asks two separate
        questions:

          core_agreement  - does the query match the family at the positions the
                            family cares about? High for naturals AND for
                            redesigns, because those positions are structurally
                            determined. On its own it says "this is the fold".

          variable_llr    - at the positions the family lets vary, do the
                            residue choices look like the generator's
                            preferences or like the family's own usage? This is
                            where an inverse-folding model leaves its signature,
                            because it fills the surface with its favourite
                            residues instead of family-typical variation.

        The conjunction (fold preserved + surface rewritten) is what separates a
        redesign from both a natural homologue and an unrelated sequence.
        """
        out = {"core_agreement": 0.0, "variable_llr": 0.0, "exposed_llr": 0.0,
               "buried_llr": 0.0, "stratum_contrast": 0.0,
               "n_core": 0.0, "n_variable": 0.0, "n_exposed": 0.0, "n_buried": 0.0}
        if not hit.matched or not hit.aligned_positions or self.information_content is None:
            return out

        ic = self.information_content
        hi = float(np.quantile(ic, 0.67))
        lo = float(np.quantile(ic, 0.33))
        bg = np.clip(self._bg, 1e-12, None)
        gen = np.clip(fingerprint.freqs, 1e-6, None)
        var_ref = np.clip(self.variable_composition, 1e-6, None)
        exp_ref = np.clip(
            self.exposed_composition
            if self.exposed_composition is not None else self.variable_composition,
            1e-6, None,
        )
        bur_ref = np.clip(
            self.buried_composition
            if self.buried_composition is not None else self.family_composition,
            1e-6, None,
        )
        exposed = self.exposed_mask if self.exposed_mask is not None else (ic <= lo)
        buried = self.buried_mask if self.buried_mask is not None else (ic >= hi)

        core_terms, var_terms, exp_terms, bur_terms = [], [], [], []
        for pos, _tgt, res in hit.aligned_positions:
            m = pos - 1
            if not (0 <= m < len(ic)):
                continue
            ridx = AA_INDEX.get(res)
            if ridx is None:
                continue
            if ic[m] >= hi:
                core_terms.append(math.log2(max(self.emissions[m, ridx], 1e-6) / bg[ridx]))
            if ic[m] <= lo:
                var_terms.append(math.log(gen[ridx]) - math.log(var_ref[ridx]))
            if exposed[m]:
                exp_terms.append(math.log(gen[ridx]) - math.log(exp_ref[ridx]))
            if buried[m]:
                bur_terms.append(math.log(gen[ridx]) - math.log(bur_ref[ridx]))

        if core_terms:
            out["core_agreement"] = float(np.mean(core_terms))
            out["n_core"] = float(len(core_terms))
        if var_terms:
            out["variable_llr"] = float(np.mean(var_terms))
            out["n_variable"] = float(len(var_terms))
        if exp_terms:
            out["exposed_llr"] = float(np.mean(exp_terms))
            out["n_exposed"] = float(len(exp_terms))
        if bur_terms:
            out["buried_llr"] = float(np.mean(bur_terms))
            out["n_buried"] = float(len(bur_terms))
        # The signature is differential, not absolute: a redesign rewrites the
        # surface while leaving the core alone, so the gap between the two
        # strata is larger than for a natural homologue. Taking the difference
        # also cancels any whole-sequence composition offset, which makes this
        # the one composition-derived signal that does not move when someone
        # simply shifts the overall amino-acid distribution.
        if exp_terms and bur_terms:
            out["stratum_contrast"] = out["exposed_llr"] - out["buried_llr"]
        return out

    def _index_members(self) -> None:
        """Project every member onto model coordinates once, as an int8 matrix.

        Nearest-neighbour identity then becomes a vectorised comparison over
        shared model positions instead of thousands of pairwise alignments.
        Family members are by definition alignable to the profile, so this is
        both much faster and more consistent than independent pairwise
        alignment.
        """
        if self.hmm is None or not self._members:
            self._member_matrix = None
            return
        M = int(self.hmm.M)
        mat = np.full((len(self._members), M), -1, dtype=np.int8)
        ins = np.zeros(len(self._members), dtype=np.int32)
        for i, s in enumerate(self._members):
            aligned = self.score(s).aligned_positions
            for pos, _tgt, res in aligned:
                idx = AA_INDEX.get(res)
                if idx is not None and 1 <= pos <= M:
                    mat[i, pos - 1] = idx
            if aligned:
                tg = [t for _m, t, _r in aligned]
                ins[i] = (max(tg) - min(tg) + 1) - len(tg)
        self._member_matrix = mat
        self._member_inserts = ins
        LOG.info("Indexed %d members onto %d model positions.", len(self._members), M)

    def _query_vector(self, cs: CleanSequence) -> np.ndarray:
        M = int(self.hmm.M)
        vec = np.full(M, -1, dtype=np.int8)
        for pos, _tgt, res in self.score(cs).aligned_positions:
            idx = AA_INDEX.get(res)
            if idx is not None and 1 <= pos <= M:
                vec[pos - 1] = idx
        return vec

    # -- covariation --------------------------------------------------------

    @staticmethod
    def _sequence_weights(mat: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Standard 1/n_neighbours weighting to damp phylogenetic redundancy.

        Without this, covariation statistics mostly describe whichever clade was
        sequenced most, not the family's structural constraints.
        """
        n, M = mat.shape
        valid = mat >= 0
        counts = np.ones(n, dtype=float)
        chunk = max(4, int(3.0e7 / max(n * M, 1)))
        for start in range(0, n, chunk):
            block = mat[start : start + chunk]
            bvalid = valid[start : start + chunk]
            both = bvalid[:, None, :] & valid[None, :, :]
            shared = both.sum(-1)
            same = ((block[:, None, :] == mat[None, :, :]) & both).sum(-1)
            ident = np.where(shared > 0, same / np.maximum(shared, 1), 0.0)
            counts[start : start + chunk] = (ident >= threshold).sum(1)
        return 1.0 / np.maximum(counts, 1.0)

    # -- covariation: mean-field DCA ----------------------------------------

    @staticmethod
    def _reduce_matrix(mat: np.ndarray) -> np.ndarray:
        """Map the 20 residues onto 7 chemical classes, gaps to the last state.

        A Potts model over 21 states needs (20M)^2 parameters; over 8 it needs
        (7M)^2, about eight times fewer, so the same data supports it at a
        correspondingly lower effective sequence count. The couplings are
        coarser - it can see that a position pair swaps charge but not which
        residue - and that is the trade being made when the full alphabet is
        out of reach.
        """
        out = np.where(mat >= 0, REDUCED_INDEX[np.clip(mat, 0, 19)], N_REDUCED)
        return out.astype(np.int8)

    def _onehot_frequencies(
        self, mat: np.ndarray, w: np.ndarray, q: int, lam: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reweighted single- and pair-site frequencies with a flat pseudocount.

        Gaps are carried as state q-1 rather than dropped. A gap is information
        about the alignment, and discarding it biases the pair statistics toward
        whichever positions happen to be well covered.
        """
        n, M = mat.shape
        states = np.where(mat >= 0, mat, q - 1).astype(np.int64)
        meff = float(w.sum())

        f1 = np.zeros((M, q))
        for a in range(q):
            f1[:, a] = (w[:, None] * (states == a)).sum(0)
        f1 = (lam / q + f1) / (lam + meff)

        # Pair frequencies as a single weighted outer product over one-hot
        # columns: X is (n, M*q), so X^T W X gives every f_ij(a,b) at once.
        sw = np.sqrt(w)
        X = np.zeros((n, M * q), dtype=np.float64)
        rows = np.arange(n)
        for i in range(M):
            X[rows, i * q + states[:, i]] = sw
        f2 = (X.T @ X) / (lam + meff)
        f2 += lam / (q * q) / (lam + meff)
        # The diagonal blocks must hold the single-site marginals, not the
        # pseudocounted outer product, or the covariance is wrong on the block
        # diagonal and the inverse picks up spurious self-coupling.
        for i in range(M):
            blk = np.zeros((q, q))
            np.fill_diagonal(blk, f1[i])
            f2[i * q : (i + 1) * q, i * q : (i + 1) * q] = blk
        return f1, f2

    def _mean_field_couplings(
        self, f1: np.ndarray, f2: np.ndarray, M: int, q: int
    ) -> np.ndarray:
        """Direct couplings e_ij = -(C^-1)_ij from the mean-field inverse.

        This is the step that distinguishes DCA from mutual information. MI
        counts every correlation, including the transitive ones: if i couples to
        k and k couples to j, then i and j correlate without touching. Inverting
        the covariance matrix conditions each pair on all other positions, which
        is what leaves direct couplings behind.

        The last state is dropped before inversion because the one-hot encoding
        is linearly dependent (frequencies sum to one), so the full covariance
        is singular by construction.
        """
        qr = q - 1  # reduced alphabet, last state as reference
        idx = np.concatenate([np.arange(i * q, i * q + qr) for i in range(M)])
        C = f2[np.ix_(idx, idx)] - np.outer(
            f1[:, :qr].ravel(), f1[:, :qr].ravel()
        )
        # Ridge for conditioning: C is estimated from finite data and is often
        # near-singular even after dropping a state.
        C.flat[:: C.shape[0] + 1] += 1e-4
        try:
            inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            LOG.warning("Covariance inversion failed; falling back to pseudo-inverse.")
            inv = np.linalg.pinv(C)
        return -inv

    @staticmethod
    def _direct_information_batch(
        e_blocks: np.ndarray, fi: np.ndarray, fj: np.ndarray, iters: int = 150
    ) -> np.ndarray:
        """Direct information for many pairs at once.

        Fields are fitted so each two-site direct model reproduces the observed
        single-site marginals - that is what makes DI comparable across pairs
        whose positions differ wildly in conservation. The fit is a Sinkhorn
        iteration, batched over pairs: a Python loop over tens of thousands of
        pairs, each with its own iteration, is the difference between seconds
        and many minutes.

        e_blocks: (K, q, q), fi/fj: (K, q). Returns (K,).
        """
        K, q, _ = e_blocks.shape
        W = np.exp(np.clip(e_blocks, -30, 30))
        mu_i = np.full((K, q), 1.0 / q)
        mu_j = np.full((K, q), 1.0 / q)
        for _ in range(iters):
            denom_i = np.einsum("kab,kb->ka", W, mu_j)
            new_i = fi / np.maximum(denom_i, 1e-12)
            new_i /= np.maximum(new_i.sum(1, keepdims=True), 1e-12)
            denom_j = np.einsum("kab,ka->kb", W, new_i)
            new_j = fj / np.maximum(denom_j, 1e-12)
            new_j /= np.maximum(new_j.sum(1, keepdims=True), 1e-12)
            shift = max(np.abs(new_i - mu_i).max(), np.abs(new_j - mu_j).max())
            mu_i, mu_j = new_i, new_j
            if shift < 1e-10:
                break
        P = W * mu_i[:, :, None] * mu_j[:, None, :]
        P /= np.maximum(P.sum((1, 2), keepdims=True), 1e-12)
        indep = fi[:, :, None] * fj[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = P * np.log(np.maximum(P, 1e-12) / np.maximum(indep, 1e-12))
        return np.nan_to_num(terms).sum((1, 2))

    def fit_covariation(
        self,
        method: str = "dca",
        max_pairs: int = 400,
        min_pairs: int = 15,
        min_z: float = 4.0,
        min_separation: int = 5,
        min_neff: float = 50.0,
        max_positions: int = 800,
        pseudocount: float = 0.5,
        oof_folds: int = 3,
        dca_pseudocount_weight: float = 1.0,
    ) -> None:
        """Learn which column pairs are directly coupled in the natural family.

        Natural families carry couplings between positions that are not
        individually constrained: a salt bridge, a packing contact, a
        charge-compensating pair. Evolution preserves the *combination* while
        letting either position vary alone. A model that samples each position
        conditioned on structure reproduces the single-position statistics a
        profile HMM measures, but has no reason to reproduce family-specific
        couplings the backbone does not dictate.

        `method="dca"` fits a mean-field Potts model (Morcos et al. 2011) and
        scores a query by its statistical energy under the direct couplings.
        `method="mi"` keeps the cheaper mutual-information proxy, which is what
        to use when memory is tight or the family has too few effective
        sequences for a Potts fit to mean anything.

        DCA is the better model but it is not free: it needs the effective
        sequence count to exceed the number of positions, and it inverts an
        (M x (q-1)) square matrix, which is where the run time goes. When Meff
        is too small for a reliable fit this falls back to MI and says so.

        `dca_pseudocount_weight` is a multiple of Meff, not an absolute count.
        That matters more than it looks: with a small absolute pseudocount the
        covariance is near-singular, its inverse is numerical noise, and the
        fit silently returns nothing useful - measured here, the known coupled
        pairs scored 1.6 sd BELOW the background. The standard mean-field
        choice of one Meff (Morcos et al. 2011) is a deliberately heavy prior
        and is what makes the inverse meaningful.
        """
        self.coupling = None
        mat = getattr(self, "_member_matrix", None)
        if mat is None or mat.shape[0] < 20:
            LOG.info("Too few members for covariation; skipping.")
            return
        n, M = mat.shape
        if M > max_positions:
            LOG.warning(
                "Profile has %d positions (>%d); skipping covariation to bound "
                "memory. Raise max_positions if you have the RAM.", M, max_positions,
            )
            return

        w = self._sequence_weights(mat)
        neff = float(w.sum())
        if neff < min_neff:
            LOG.info(
                "Effective sequence count %.1f < %.0f; covariation would be "
                "noise. Skipping.", neff, min_neff,
            )
            return

        # A Potts fit over q states needs roughly 2*M*(q-1)/20 effective
        # sequences. When the full alphabet is out of reach, a reduced one may
        # not be - direct couplings over seven chemical classes still condition
        # away the transitive correlations that mutual information counts, and
        # that is the property worth keeping.
        dca_q = None
        if method == "dca":
            for q_try in (21, N_REDUCED + 1):
                if neff >= 2.0 * M * (q_try - 1) / 20.0:
                    dca_q = q_try
                    break
            if dca_q == N_REDUCED + 1:
                LOG.info(
                    "Meff=%.0f supports a reduced-alphabet Potts model (7 "
                    "chemical classes) but not the full 20-residue one, which "
                    "would need %.0f. Using reduced-alphabet DCA.",
                    neff, 2.0 * M,
                )
        if method == "dca" and dca_q is None:
            # Say what would actually fix it. Meff/n runs about 0.7 on a
            # dereplicated natural family, and the build split is a little over
            # half, so the sequence count needed is roughly 2*M/0.7/0.55.
            ratio = neff / max(len(mat), 1)
            needed = int(2.0 * M / max(ratio, 0.3) / 0.55)
            LOG.warning(
                "Meff=%.0f is only %.2fx the %d model positions; a Potts model "
                "needs at least 2x, so using mutual information instead. "
                "Reaching DCA on this family would take roughly %d sequences "
                "before splitting (you have %d) - include unreviewed UniProt "
                "entries, or accept MI.",
                neff, neff / M, M, needed, len(mat),
            )
            method = "mi"
        if method == "dca":
            budget_gb = (M * (dca_q - 1)) ** 2 * 8 / 1e9
            if budget_gb > 3.0:
                LOG.warning(
                    "A DCA fit here needs ~%.1f GB for the covariance matrix; "
                    "using mutual information instead. Reduce max_positions or "
                    "run with more memory.", budget_gb,
                )
                method = "mi"

        valid = mat >= 0
        if method == "dca":
            self._fit_covariation_dca(
                mat, valid, w, neff, M, max_pairs, min_pairs, min_z,
                min_separation, dca_pseudocount_weight * neff, oof_folds, dca_q,
            )
        else:
            self._fit_covariation_mi(
                mat, valid, w, neff, M, max_pairs, min_pairs, min_z,
                min_separation, pseudocount, oof_folds,
            )

    def _fit_covariation_dca(
        self, mat, valid, w, neff, M, max_pairs, min_pairs, min_z,
        min_separation, lam, oof_folds, q=21,
    ) -> None:
        if q != 21:
            mat = self._reduce_matrix(mat)
        t0 = time.time()
        f1, f2 = self._onehot_frequencies(mat, w, q, lam)
        e = self._mean_field_couplings(f1, f2, M, q)
        LOG.info(
            "DCA: inverted a %dx%d covariance in %.1fs (Meff=%.0f, Meff/M=%.1f).",
            M * (q - 1), M * (q - 1), time.time() - t0, neff, neff / M,
        )

        # Direct information for every pair, then the average product
        # correction. DI is computed for ALL pairs, not only the well-separated
        # ones: APC subtracts each position's mean coupling, and leaving the
        # near-diagonal pairs out of those means biases the correction.
        qr = q - 1
        iu = np.triu_indices(M, k=1)
        blocks = np.zeros((len(iu[0]), q, q))
        e_resh = e.reshape(M, qr, M, qr)
        blocks[:, :qr, :qr] = e_resh[iu[0], :, iu[1], :]
        di_vals = self._direct_information_batch(blocks, f1[iu[0]], f1[iu[1]])
        di = np.zeros((M, M))
        di[iu] = di_vals
        di = di + di.T
        row_mean = di.mean(axis=1)
        total = max(di.mean(), 1e-12)
        di_apc = di - np.outer(row_mean, row_mean) / total

        sep = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :])
        cand = np.argwhere(np.triu(sep >= min_separation, k=1))
        scores = di_apc[cand[:, 0], cand[:, 1]]
        cut = float(scores.mean() + min_z * scores.std())
        keep = scores >= cut
        if keep.sum() < min_pairs:
            LOG.info(
                "Only %d pairs exceed %.1f sd of the DI background; no reliable "
                "direct couplings in this family, skipping the signal.",
                int(keep.sum()), min_z,
            )
            return
        idx = np.flatnonzero(keep)
        order = idx[np.argsort(-scores[idx])][:max_pairs]
        pairs = cand[order]
        LOG.info("DCA: %d pairs above %.1f sd of the DI background (of %d candidates).",
                 len(pairs), min_z, len(scores))

        e_pairs = np.zeros((len(pairs), q, q))
        for k, (i, j) in enumerate(pairs):
            e_pairs[k, :qr, :qr] = e[i * qr : (i + 1) * qr, j * qr : (j + 1) * qr]

        self.coupling = {
            "method": "dca" if q == 21 else "dca7",
            "pairs": pairs.astype(np.int32), "e": e_pairs,
            "f1": f1, "neff": neff, "di_apc": scores[order],
            "member_mean": 0.0, "member_sd": 1.0, "q": q,
        }

        # Out-of-fold calibration. Scoring the sequences the couplings were fit
        # from makes members look far better than any unseen homologue, which
        # would turn an honest relative into an outlier. Each fold is refit
        # without its own members - for DCA that means a fresh inversion per
        # fold, which is most of the cost here and the reason it is worth it.
        rng = np.random.default_rng(self.random_state)
        folds = max(2, min(oof_folds, 5)) if len(mat) >= 60 else 2
        assign = rng.permutation(len(mat)) % folds
        oof: List[float] = []
        for f in range(folds):
            tr = assign != f
            if tr.sum() < 20:
                continue
            f1_tr, f2_tr = self._onehot_frequencies(mat[tr], w[tr], q, lam)
            e_tr = self._mean_field_couplings(f1_tr, f2_tr, M, q)
            e_fold = np.zeros_like(e_pairs)
            for k, (i, j) in enumerate(pairs):
                e_fold[k, :qr, :qr] = e_tr[i * qr : (i + 1) * qr, j * qr : (j + 1) * qr]
            saved = self.coupling["e"]
            self.coupling["e"] = e_fold
            for i in np.flatnonzero(assign == f):
                s, _ = self._coupling_pmi(mat[i])
                if s is not None:
                    oof.append(s)
            self.coupling["e"] = saved
        if len(oof) > 2:
            self.coupling["member_mean"] = float(np.mean(oof))
            self.coupling["member_sd"] = float(np.std(oof)) or 1.0
        LOG.info(
            "%s: %d directly coupled pairs; out-of-fold members score "
            "%.3f +- %.3f mean coupling energy (%d folds, %.0fs total).",
            "DCA" if q == 21 else "Reduced-alphabet DCA",
            len(pairs), self.coupling["member_mean"], self.coupling["member_sd"],
            folds, time.time() - t0,
        )

    def _fit_covariation_mi(
        self, mat, valid, w, neff, M, max_pairs, min_pairs, min_z,
        min_separation, pseudocount, oof_folds,
    ) -> None:
        n = len(mat)
        f1 = np.empty((M, 20))
        for a in range(20):
            f1[:, a] = (w[:, None] * (mat == a)).sum(0)
        wvalid = (w[:, None] * valid).sum(0)
        f1 = (f1 + pseudocount) / (wvalid[:, None] + 20 * pseudocount)

        red = np.where(valid, REDUCED_INDEX[np.clip(mat, 0, 19)], -1)
        sw = np.sqrt(w).astype(np.float32)
        A = np.zeros((n, M * N_REDUCED), dtype=np.float32)
        rows = np.arange(n)
        for i in range(M):
            ok = red[:, i] >= 0
            A[rows[ok], red[ok, i].astype(np.int64) + i * N_REDUCED] = sw[ok]
        C = (A.T @ A).reshape(M, N_REDUCED, M, N_REDUCED)
        Z = C.sum(axis=(1, 3))
        np.fill_diagonal(Z, np.inf)
        P = C / np.maximum(Z[:, None, :, None], 1e-9)
        Pi = P.sum(axis=3)[:, :, :, None]
        Pj = P.sum(axis=1)[:, None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = P * np.log(np.maximum(P, 1e-12) / np.maximum(Pi * Pj, 1e-12))
        mi = np.nan_to_num(terms.sum(axis=(1, 3)))
        np.fill_diagonal(mi, 0.0)
        del A, C, P, Pi, Pj, terms

        row_mean = mi.mean(axis=1)
        total = max(mi.mean(), 1e-12)
        mi_apc = mi - np.outer(row_mean, row_mean) / total

        sep = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :])
        cand = np.argwhere(np.triu(sep >= min_separation, k=1))
        if len(cand) == 0:
            LOG.info("No candidate pairs for covariation.")
            return
        scores = mi_apc[cand[:, 0], cand[:, 1]]
        cut = float(scores.mean() + min_z * scores.std())
        keep = scores >= cut
        if keep.sum() < min_pairs:
            LOG.info(
                "Only %d column pairs exceed %.1f sd of the MI background; "
                "no reliable couplings in this family, skipping the signal.",
                int(keep.sum()), min_z,
            )
            return
        idx = np.flatnonzero(keep)
        order = idx[np.argsort(-scores[idx])][:max_pairs]
        pairs = cand[order]
        LOG.info("Selected %d coupled pairs above %.1f sd (of %d candidates).",
                 len(pairs), min_z, len(scores))

        lam = 10.0
        f2 = self._fit_joints(pairs, mat, valid, w, f1, lam)
        self.coupling = {
            "method": "mi", "pairs": pairs.astype(np.int32), "f2": f2, "f1": f1,
            "neff": neff, "mi_apc": scores[order],
            "member_mean": 0.0, "member_sd": 1.0, "q": 20,
        }

        rng = np.random.default_rng(self.random_state)
        folds = 5 if n >= 50 else 2
        assign = rng.permutation(n) % folds
        oof: List[float] = []
        for f in range(folds):
            tr = assign != f
            if tr.sum() < 10:
                continue
            f2_tr = self._fit_joints(pairs, mat[tr], valid[tr], w[tr], f1, lam)
            saved = self.coupling["f2"]
            self.coupling["f2"] = f2_tr
            for i in np.flatnonzero(assign == f):
                s, _ = self._coupling_pmi(mat[i])
                if s is not None:
                    oof.append(s)
            self.coupling["f2"] = saved
        if len(oof) > 2:
            self.coupling["member_mean"] = float(np.mean(oof))
            self.coupling["member_sd"] = float(np.std(oof)) or 1.0
        LOG.info(
            "Covariation (MI): %d coupled pairs from n_eff=%.0f; out-of-fold "
            "members score %.3f +- %.3f.", len(pairs), neff,
            self.coupling["member_mean"], self.coupling["member_sd"],
        )

    @staticmethod
    def _fit_joints(pairs, mat, valid, w, f1, lam) -> np.ndarray:
        """Pairwise joint distributions, with the product of marginals as prior."""
        f2 = np.empty((len(pairs), 20, 20))
        for k, (i, j) in enumerate(pairs):
            ok = valid[:, i] & valid[:, j]
            joint = np.zeros((20, 20))
            if ok.any():
                np.add.at(joint, (mat[ok, i], mat[ok, j]), w[ok])
            f2[k] = (joint + lam * np.outer(f1[i], f1[j])) / (joint.sum() + lam)
        return f2

    def _coupling_pmi(self, qvec: np.ndarray) -> Tuple[Optional[float], int]:
        """Score a query over the coupled pairs.

        Under DCA this is the mean direct-coupling energy: how favourable the
        query's residue combinations are under the fitted Potts model. Under MI
        it is mean pointwise mutual information against the empirical joints.
        Both are oriented so that higher means more family-like, and both are
        reported as a z-score against the family's own out-of-fold spread.
        """
        if not self.coupling:
            return None, 0
        pairs = self.coupling["pairs"]
        a = qvec[pairs[:, 0]]
        b = qvec[pairs[:, 1]]
        if self.coupling["method"] in ("dca", "dca7"):
            q = self.coupling["q"]
            if self.coupling["method"] == "dca7":
                qvec = self._reduce_matrix(qvec.reshape(1, -1))[0]
                gap = N_REDUCED
            else:
                gap = q - 1
            a = qvec[pairs[:, 0]]
            b = qvec[pairs[:, 1]]
            ai = np.where(a >= 0, a, gap)
            bj = np.where(b >= 0, b, gap)
            ok = (a >= 0) | (b >= 0)
            if ok.sum() < 10:
                return None, int(ok.sum())
            k = np.arange(len(pairs))[ok]
            energy = self.coupling["e"][k, ai[ok], bj[ok]]
            return float(energy.mean()), int(ok.sum())
        f1, f2 = self.coupling["f1"], self.coupling["f2"]
        ok = (a >= 0) & (b >= 0)
        if ok.sum() < 10:
            return None, int(ok.sum())
        k = np.arange(len(pairs))[ok]
        joint = f2[k, a[ok], b[ok]]
        marg = f1[pairs[ok, 0], a[ok]] * f1[pairs[ok, 1], b[ok]]
        pmi = np.log(np.maximum(joint, 1e-12) / np.maximum(marg, 1e-12))
        return float(pmi.mean()), int(ok.sum())

    def coupling_violation(self, cs: CleanSequence) -> Tuple[float, int]:
        """How far the query falls below family members in coupling agreement.

        Returned as a z-score against the family's own distribution, so it needs
        no hand-set constant and stays comparable across families. Higher means
        the query satisfies the family's couplings less well than a real member.
        """
        if not self.coupling:
            return 0.0, 0
        pmi, n_pairs = self._coupling_pmi(self._query_vector(cs))
        if pmi is None:
            return 0.0, n_pairs
        mu = self.coupling["member_mean"]
        sd = max(self.coupling["member_sd"], 1e-6)
        return float((mu - pmi) / sd), n_pairs

    # -- evolutionary model -------------------------------------------------

    def fit_evolutionary_model(
        self,
        max_pairs: int = 4000,
        dist_lo: float = 0.12,
        dist_hi: float = 0.75,
        pseudocount: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Learn how this family actually diverges, as a process rather than a set.

        Everything measured so far treats the family as a bag of sequences and
        asks whether the query's residue frequencies fit. But natural homologues
        are not independent draws from a distribution - they are the tips of a
        tree, related by a substitution process with its own regularities:

          * substitutions are exchangeable, not arbitrary. A family tolerates
            I<->V far more readily than I<->D, and the pattern is family
            specific;
          * the process is approximately time reversible, so over a tree there
            is no systematic drift in one direction;
          * substitutions concentrate at fast sites, and which sites are fast is
            a property of the family;
          * divergence is accompanied by indels, at a rate that scales with it;
          * distances between tips satisfy the four-point condition, because
            they came from a tree.

        An inverse-folding model reproduces none of this. It samples residues
        conditioned on a backbone, which gives a sequence that sits at the right
        place in composition space but did not get there by descent. That makes
        these five properties a different kind of evidence from everything else
        here: they test the *history* implied by the sequence rather than its
        statistics.

        All five are measured against the family's own distribution, so no
        external substitution matrix is assumed - the exchangeability model is
        estimated from the family in the same way BLOSUM is estimated from
        aligned blocks.
        """
        self.evo = None
        mat = getattr(self, "_member_matrix", None)
        if mat is None or mat.shape[0] < 30:
            LOG.info("Too few members for an evolutionary model; skipping.")
            return
        n, M = mat.shape
        rng = np.random.default_rng(self.random_state + seed)
        valid = mat >= 0

        # --- per-site evolutionary rate ------------------------------------
        # Simpson diversity per column: the chance two random members differ
        # there. This is the family's own rate profile, which is what the query
        # is supposed to be congruent with.
        rate = np.zeros(M)
        for m in range(M):
            col = mat[valid[:, m], m]
            if len(col) < 5:
                rate[m] = 0.0
                continue
            counts = np.bincount(col, minlength=20).astype(float)
            p = counts / counts.sum()
            rate[m] = 1.0 - float((p ** 2).sum())

        # --- member-member distance matrix in model coordinates -------------
        dist = np.ones((n, n), dtype=np.float32)
        chunk = max(4, int(3.0e7 / max(n * M, 1)))
        for start in range(0, n, chunk):
            blk, bval = mat[start : start + chunk], valid[start : start + chunk]
            both = bval[:, None, :] & valid[None, :, :]
            shared = both.sum(-1)
            diff = ((blk[:, None, :] != mat[None, :, :]) & both).sum(-1)
            dist[start : start + chunk] = np.where(
                shared >= 20, diff / np.maximum(shared, 1), 1.0
            )
        np.fill_diagonal(dist, 0.0)

        # --- empirical exchangeability, BLOSUM-style ------------------------
        # Counted only over positions that differ: given that a substitution
        # happened, which residue pairs does this family actually interchange?
        iu = np.triu_indices(n, k=1)
        band = (dist[iu] >= dist_lo) & (dist[iu] <= dist_hi)
        pi, pj = iu[0][band], iu[1][band]
        if len(pi) > max_pairs:
            sel = rng.choice(len(pi), max_pairs, replace=False)
            pi, pj = pi[sel], pj[sel]
        counts = np.full((20, 20), pseudocount, dtype=float)
        for a, b in zip(pi, pj):
            both = valid[a] & valid[b]
            x, y = mat[a][both], mat[b][both]
            d = x != y
            if d.any():
                np.add.at(counts, (x[d], y[d]), 1.0)
                np.add.at(counts, (y[d], x[d]), 1.0)
        np.fill_diagonal(counts, 0.0)
        total = counts.sum()
        if total < 200:
            LOG.info("Too few observed substitutions for an exchangeability model.")
            return
        q = counts / total
        marg = q.sum(axis=1)
        expected = 2.0 * np.outer(marg, marg)
        np.fill_diagonal(expected, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            exch = np.log2(np.maximum(q, 1e-9) / np.maximum(expected, 1e-9))
        np.fill_diagonal(exch, 0.0)
        exch = np.nan_to_num(exch, neginf=-6.0, posinf=6.0)

        # --- indel rate as a function of divergence -------------------------
        ins = getattr(self, "_member_inserts", np.zeros(n))
        gaps = ~valid
        indel_obs, indel_dist = [], []
        for a, b in zip(pi[:600], pj[:600]):
            lo = max(np.argmax(valid[a]), np.argmax(valid[b]))
            hi = min(M - 1 - np.argmax(valid[a][::-1]), M - 1 - np.argmax(valid[b][::-1]))
            if hi - lo < 30:
                continue
            only_one = gaps[a, lo:hi + 1] ^ gaps[b, lo:hi + 1]
            events = int(np.count_nonzero(np.diff(np.concatenate(([0], only_one.astype(np.int8)))) == 1))
            events += abs(int(ins[a]) - int(ins[b]))
            indel_obs.append(events)
            indel_dist.append(float(dist[a, b]))
        indel_bins = np.linspace(0.0, 1.0, 9)
        indel_curve = np.zeros(len(indel_bins))
        if len(indel_obs) >= 30:
            obs = np.asarray(indel_obs, dtype=float)
            dd = np.asarray(indel_dist, dtype=float)
            which = np.clip(np.digitize(dd, indel_bins) - 1, 0, len(indel_bins) - 1)
            for b in range(len(indel_bins)):
                sel = which == b
                if sel.sum() >= 5:
                    indel_curve[b] = float(obs[sel].mean())
            # fill empty bins by carrying the nearest filled one forward
            filled = np.flatnonzero(indel_curve > 0)
            if len(filled):
                for b in range(len(indel_bins)):
                    if indel_curve[b] == 0:
                        indel_curve[b] = indel_curve[filled[np.argmin(np.abs(filled - b))]]
            indel_slope = float(obs.mean() / max(dd.mean(), 1e-6))
        else:
            indel_slope = 0.0

        self.evo = {
            "rate": rate,
            "dist_corrected": self._kimura(dist).astype(np.float32),
            "exchangeability": exch,
            "dist": dist,
            "indel_slope": indel_slope,
            "indel_bins": indel_bins,
            "indel_curve": indel_curve,
            "n_subst_pairs": int(len(pi)),
        }
        LOG.info(
            "Evolutionary model: %d substitution pairs, %.0f indel events per "
            "unit divergence, mean site rate %.2f.",
            len(pi), indel_slope, float(rate.mean()),
        )

    def nearest_neighbour(self, cs: CleanSequence, min_shared: int = 20,
                          exclude_self: bool = True) -> Tuple[float, int]:
        """Highest identity to any family member, with that member's index."""
        if getattr(self, "_member_matrix", None) is None or not self._members:
            return 0.0, -1
        q = self._query_vector(cs)
        if (q >= 0).sum() < min_shared:
            return 0.0, -1
        mat = self._member_matrix
        shared = (mat >= 0) & (q >= 0)[None, :]
        n_shared = shared.sum(axis=1)
        matches = ((mat == q[None, :]) & shared).sum(axis=1)
        valid = n_shared >= min_shared
        if not valid.any():
            return 0.0, -1
        ident = np.zeros(len(mat))
        ident[valid] = matches[valid] / n_shared[valid]
        if exclude_self:
            ident[ident >= 0.999] = 0.0
        best = int(np.argmax(ident))
        return float(ident[best]), best

    def nearest_neighbour_identity(self, cs: CleanSequence, min_shared: int = 20,
                                   exclude_self: bool = True) -> float:
        return self.nearest_neighbour(cs, min_shared, exclude_self)[0]

    def evolutionary_signals(
        self, cs: CleanSequence, fingerprint: GeneratorFingerprint,
        n_quartets: int = 300, seed: int = 0,
    ) -> Dict[str, float]:
        """Test the history the query implies against the family's own process."""
        out = {
            "subst_implausibility": 0.0, "directional_drift": 0.0,
            "equilibrium_drift": 0.0,
            "indel_deficit": 0.0, "rate_incongruence": 0.0,
            "tree_violation": 0.0, "n_substitutions": 0.0, "n_quartets": 0.0,
        }
        evo = getattr(self, "evo", None)
        if not evo:
            return out
        ident, nn = self.nearest_neighbour(cs)
        if nn < 0:
            return out

        q = self._query_vector(cs)
        ref = self._member_matrix[nn]
        both = (q >= 0) & (ref >= 0)
        if both.sum() < 20:
            return out
        diff = both & (q != ref)
        n_sub = int(diff.sum())
        out["n_substitutions"] = float(n_sub)

        rate, exch, dist = evo["rate"], evo["exchangeability"], evo["dist"]

        if n_sub >= 5:
            a, b = ref[diff], q[diff]
            # How exchangeable are these substitutions by this family's own
            # standards? Evolution moves between residues the family already
            # interchanges; structure-conditioned sampling has no such prior.
            out["subst_implausibility"] = -float(exch[a, b].mean())

            # Detailed balance: over a tree, substitutions in one direction are
            # balanced by the reverse, so there is no net drift. A generator
            # pulls every sequence toward its own equilibrium instead, which
            # shows up as directional flow in the substitution counts.
            fam = np.clip(self.family_composition, 1e-6, None)
            gen = np.clip(fingerprint.freqs, 1e-6, None)
            pull = np.log(gen) - np.log(fam)
            out["directional_drift"] = float(np.mean(pull[b] - pull[a]))

            # The same test without naming a generator. Over a tree the process
            # is reversible with respect to the family's own equilibrium, so
            # flow along the family's frequency gradient averages to zero in
            # either direction. A sequence pulled toward some other equilibrium
            # shows net flow - and which equilibrium that is does not matter, so
            # this fires for generators that have never been fingerprinted.
            # Taken two-sided for exactly that reason.
            fam_pull = np.log(fam)
            out["equilibrium_drift"] = abs(float(np.mean(fam_pull[b] - fam_pull[a])))

            # Do the changes sit where this family actually varies? A natural
            # homologue differs at fast sites; a redesign differs where the
            # backbone tolerates change, which is a different set.
            #
            # Scored as a likelihood ratio rather than a correlation. With a
            # close relative there are few differences, and a correlation over
            # a handful of points is mostly noise; the likelihood uses every
            # site, including the ones that did NOT change, which is where most
            # of the information about rate congruence actually sits.
            d_ind = diff[both].astype(float)
            r = rate[both]
            D = float(d_ind.mean())
            if 0.0 < D < 1.0 and r.mean() > 1e-9:
                p_fam = np.clip(r * (D / r.mean()), 1e-3, 1 - 1e-3)
                ll_fam = float(np.mean(
                    d_ind * np.log(p_fam) + (1 - d_ind) * np.log(1 - p_fam)))
                ll_flat = float(D * math.log(D) + (1 - D) * math.log(1 - D))
                out["rate_incongruence"] = ll_flat - ll_fam

        # Divergence normally comes with indels. A fixed backbone cannot produce
        # them, so a sequence far from its nearest relative with none at all has
        # diverged by a route evolution does not take.
        if evo["indel_slope"] > 0:
            hit = self.score(cs)
            if hit.aligned_positions:
                tg = [t for _m, t, _r in hit.aligned_positions]
                q_ins = (max(tg) - min(tg) + 1) - len(tg)
            else:
                q_ins = 0
            member_ins = getattr(self, "_member_inserts", None)
            # Indel counts are Poisson with a small mean, so one comparison is
            # dominated by sampling noise. Pooling the k closest relatives
            # averages that down without reaching into unrelated parts of the
            # tree.
            dq_raw = self._distances_to_members(q, corrected=False)
            order = np.argsort(dq_raw)[:8]
            deficits = []
            for j in order:
                rj = self._member_matrix[j]
                bj = (q >= 0) & (rj >= 0)
                if bj.sum() < 30:
                    continue
                idx = np.flatnonzero(bj)
                lo, hi = idx[0], idx[-1]
                only_one = (~(q >= 0))[lo : hi + 1] ^ (~(rj >= 0))[lo : hi + 1]
                ev = int(np.count_nonzero(
                    np.diff(np.concatenate(([0], only_one.astype(np.int8)))) == 1))
                if member_ins is not None:
                    ev += abs(q_ins - int(member_ins[j]))
                curve = evo.get("indel_curve")
                if curve is not None and curve.any():
                    bins = evo["indel_bins"]
                    expected = float(curve[int(np.clip(
                        np.digitize(dq_raw[j], bins) - 1, 0, len(bins) - 1))])
                else:
                    expected = evo["indel_slope"] * float(dq_raw[j])
                deficits.append((expected - ev) / math.sqrt(expected + 1.0))
            if deficits:
                out["indel_deficit"] = float(np.median(deficits))

        # Four-point condition. Distances among tips of a tree satisfy it; a
        # sequence that did not descend from the tree distorts the quartets it
        # takes part in. This is the delta-plot statistic, restricted to
        # quartets containing the query.
        n = len(self._member_matrix)
        if n >= 8:
            dq = self._distances_to_members(q, corrected=True)
            dmat = evo.get("dist_corrected", dist)
            ok = np.flatnonzero(np.isfinite(dq) & (dq > 0))
            if len(ok) >= 4:
                rng = np.random.default_rng(self.random_state + seed)
                tri = rng.choice(ok, size=(min(n_quartets, 600), 3))
                a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
                keep = (a != b) & (b != c) & (a != c)
                a, b, c = a[keep], b[keep], c[keep]
                if len(a) >= 10:
                    s = np.stack([
                        dq[a] + dmat[b, c], dq[b] + dmat[a, c], dq[c] + dmat[a, b],
                    ], axis=1)
                    s.sort(axis=1)
                    smin, smid, smax = s[:, 0], s[:, 1], s[:, 2]
                    span = smax - smin
                    # Quartets whose three distance sums are nearly equal carry
                    # no topological information, and delta is numerically
                    # unstable there. Standard practice is to drop them rather
                    # than average noise into the statistic.
                    good = span > 0.05 * np.maximum(smax, 1e-9)
                    if good.sum() >= 10:
                        delta = (smax[good] - smid[good]) / span[good]
                        out["tree_violation"] = float(delta.mean())
                        out["n_quartets"] = float(int(good.sum()))
        return out

    @staticmethod
    def _kimura(p: np.ndarray) -> np.ndarray:
        """Kimura's protein distance correction, d = -ln(1 - p - 0.2 p^2).

        Observed p-distance saturates as divergence grows: two sequences 60%
        and 80% diverged look far more similar than they are, because of
        multiple hits at the same site. The four-point condition is a statement
        about additive distances, so testing it on raw p-distances mostly
        measures saturation. The correction restores approximate additivity.
        """
        p = np.clip(p, 0.0, 0.85)
        return -np.log(np.maximum(1.0 - p - 0.2 * p * p, 1e-6))

    def _distances_to_members(self, q: np.ndarray, corrected: bool = True) -> np.ndarray:
        mat = self._member_matrix
        shared = (mat >= 0) & (q >= 0)[None, :]
        n_shared = shared.sum(axis=1)
        diff = ((mat != q[None, :]) & shared).sum(axis=1)
        p = np.where(n_shared >= 20, diff / np.maximum(n_shared, 1), 1.0)
        return self._kimura(p) if corrected else p

    # -- phylogenetic placement ---------------------------------------------

    @staticmethod
    def _neighbour_join(d: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """Neighbour-joining on a small distance matrix.

        Returns the merge list [(a, b, branch_a, branch_b), ...] where indices
        >= n refer to internal nodes created by earlier merges. Saitou-Nei, with
        the standard Q criterion.
        """
        n = d.shape[0]
        active = list(range(n))
        D = {(i, j): float(d[i, j]) for i in range(n) for j in range(n) if i != j}
        merges: List[Tuple[int, int, float, float]] = []
        next_node = n
        while len(active) > 2:
            m = len(active)
            r = {i: sum(D[(i, k)] for k in active if k != i) for i in active}
            best, best_q = None, np.inf
            for ii, i in enumerate(active):
                for j in active[ii + 1:]:
                    q = (m - 2) * D[(i, j)] - r[i] - r[j]
                    if q < best_q:
                        best_q, best = q, (i, j)

            i, j = best
            dij = D[(i, j)]
            li = 0.5 * dij + (r[i] - r[j]) / (2 * (m - 2)) if m > 2 else 0.5 * dij
            lj = dij - li
            li, lj = max(li, 0.0), max(lj, 0.0)
            merges.append((i, j, li, lj))
            u = next_node
            next_node += 1
            for k in active:
                if k in (i, j):
                    continue
                duk = 0.5 * (D[(i, k)] + D[(j, k)] - dij)
                D[(u, k)] = D[(k, u)] = max(duk, 0.0)
            active = [k for k in active if k not in (i, j)] + [u]
        if len(active) == 2:
            i, j = active
            merges.append((i, j, D[(i, j)] / 2.0, D[(i, j)] / 2.0))
        return merges

    @staticmethod
    def _fitch_root_states(
        merges: Sequence[Tuple[int, int, float, float]],
        leaf_states: np.ndarray,
        n_leaves: int,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Fitch parsimony ancestral states at the root of a small tree.

        Vectorised over alignment positions. Ambiguity at a node is carried as a
        state set, and ties at the root are broken by the family's own emission
        probabilities rather than arbitrarily - a parsimony tie is exactly where
        the profile's opinion should decide.

        leaf_states: (n_leaves, M) int8, -1 for gap. prior: (M, 20).
        Returns (M,) int8 with -1 where the root is a gap in every leaf.
        """
        M = leaf_states.shape[1]
        n_nodes = n_leaves + len(merges)
        sets = np.zeros((n_nodes, M, 20), dtype=bool)
        for i in range(n_leaves):
            ok = leaf_states[i] >= 0
            sets[i, np.flatnonzero(ok), leaf_states[i][ok]] = True
            # A gap is uninformative, not a 21st state, so it votes for nothing
            # and lets the other leaves decide.
            sets[i, ~ok, :] = True
        node_id = n_leaves
        for a, b, _la, _lb in merges:
            inter = sets[a] & sets[b]
            has = inter.any(axis=1)
            sets[node_id] = np.where(has[:, None], inter, sets[a] | sets[b])
            node_id += 1
        root = sets[n_nodes - 1]
        scored = np.where(root, prior, -np.inf)
        best = scored.argmax(axis=1).astype(np.int8)
        best[~root.any(axis=1)] = -1
        # A column that every leaf gaps carries no information about the
        # ancestor. Imputing the profile's favourite residue there would make
        # whatever the query has look like a substitution from a residue nobody
        # observed, inflating the substitution count at exactly the positions
        # where the local subtree knows nothing.
        covered = (leaf_states >= 0).any(axis=0)
        best[~covered] = -1
        return best

    def fit_placement_null(self, k_neighbours: int = 16, max_members: int = 250,
                           seed: int = 0) -> None:
        """Leave-one-out null for phylogenetic placement.

        Every signal that compares a query to its nearest relative inherits a
        problem: the nearest relative is itself a diverged tip, so the
        comparison measures the distance between two cousins rather than
        descent from a common ancestor. This builds the honest version - for
        each member, take its nearest relatives EXCLUDING itself, infer the
        ancestor at the root of their local tree, and record how far the member
        sits from that ancestor.

        A real member sits a short pendant edge from an ancestor it actually
        descends from. A redesign is pulled off the tree: it keeps the fold, so
        it has neighbours, but no ancestor explains it. That distribution is
        what the query is then compared against.
        """
        self.placement = None
        mat = getattr(self, "_member_matrix", None)
        if mat is None or len(mat) < max(30, k_neighbours + 2):
            LOG.info("Too few members for a placement null; skipping.")
            return
        if self.emissions is None:
            return
        evo = getattr(self, "evo", None)
        exch = evo["exchangeability"] if evo else None
        prior = np.log(np.clip(self.emissions, 1e-9, None))

        rng = np.random.default_rng(self.random_state + seed)
        idx = np.arange(len(mat))
        probe = idx if len(idx) <= max_members else rng.choice(idx, max_members, replace=False)

        pendant, implaus = [], []
        t0 = time.time()
        for i in probe:
            res = self._place(mat[i], exclude=int(i), k=k_neighbours,
                              exch=exch, prior=prior)
            if res is None:
                continue
            pendant.append(res["pendant"])
            if res["implausibility"] is not None:
                implaus.append(res["implausibility"])
        if len(pendant) < 10:
            LOG.info("Placement null too sparse; skipping.")
            return
        self.placement = {
            "k": k_neighbours,
            "pendant_mean": float(np.mean(pendant)),
            "pendant_sd": float(np.std(pendant)) or 1.0,
            "implaus_mean": float(np.mean(implaus)) if implaus else 0.0,
            "implaus_sd": (float(np.std(implaus)) or 1.0) if implaus else 1.0,
            "n": len(pendant),
        }
        LOG.info(
            "Placement null on %d members in %.0fs: pendant %.3f +- %.3f "
            "substitutions/site from the inferred ancestor.",
            len(pendant), time.time() - t0,
            self.placement["pendant_mean"], self.placement["pendant_sd"],
        )

    def _place(
        self, qvec: np.ndarray, exclude: Optional[int], k: int,
        exch: Optional[np.ndarray], prior: np.ndarray,
    ) -> Optional[Dict[str, Optional[float]]]:
        """Place one sequence: find neighbours, infer their ancestor, measure.

        Returns the pendant distance to the inferred ancestor and, when an
        exchangeability model is available, how implausible the substitutions
        from that ancestor are by this family's own standards.
        """
        mat = self._member_matrix
        dq = self._distances_to_members(qvec, corrected=False)
        if exclude is not None and 0 <= exclude < len(dq):
            dq = dq.copy()
            dq[exclude] = np.inf
        order = np.argsort(dq)[:k]
        order = order[np.isfinite(dq[order]) & (dq[order] < 1.0)]
        if len(order) < 4:
            return None

        sub = mat[order]
        # Pairwise distances among the neighbours, in model coordinates.
        shared = (sub >= 0)[:, None, :] & (sub >= 0)[None, :, :]
        n_shared = shared.sum(-1)
        diff = ((sub[:, None, :] != sub[None, :, :]) & shared).sum(-1)
        d = np.where(n_shared >= 20, diff / np.maximum(n_shared, 1), 1.0)
        np.fill_diagonal(d, 0.0)
        d = self._kimura(d)

        merges = self._neighbour_join(d)
        anc = self._fitch_root_states(merges, sub, len(order), prior)

        both = (qvec >= 0) & (anc >= 0)
        if both.sum() < 20:
            return None
        diffs = both & (qvec != anc)
        pendant = float(self._kimura(np.array([diffs.sum() / both.sum()]))[0])

        implaus = None
        if exch is not None and diffs.sum() >= 5:
            a, b = anc[diffs], qvec[diffs]
            implaus = -float(exch[a, b].mean())
        return {"pendant": pendant, "implausibility": implaus,
                "n_substitutions": float(diffs.sum()), "n_neighbours": float(len(order))}

    def placement_signals(self, cs: CleanSequence) -> Dict[str, float]:
        """Compare the query's placement against the family's own distribution."""
        out = {"pendant_excess": 0.0, "ancestral_implausibility": 0.0,
               "n_placement_substitutions": 0.0}
        p = getattr(self, "placement", None)
        if not p or self.emissions is None:
            return out
        evo = getattr(self, "evo", None)
        res = self._place(
            self._query_vector(cs), exclude=None, k=p["k"],
            exch=evo["exchangeability"] if evo else None,
            prior=np.log(np.clip(self.emissions, 1e-9, None)),
        )
        if res is None:
            return out
        out["pendant_excess"] = (res["pendant"] - p["pendant_mean"]) / p["pendant_sd"]
        out["n_placement_substitutions"] = res["n_substitutions"] or 0.0
        if res["implausibility"] is not None:
            out["ancestral_implausibility"] = (
                res["implausibility"] - p["implaus_mean"]) / p["implaus_sd"]
        return out

    # -- self-calibration ---------------------------------------------------

    def fit_feature_reference(
        self,
        extractor,
        fingerprint,
        reference_sequences: Optional[Sequence[CleanSequence]] = None,
        max_sequences: int = 300,
        seed: int = 0,
    ) -> Dict[str, Tuple[float, float]]:
        """Record how real members of THIS family score on every signal.

        This is what makes the detector survive unusual families. Fixed cutoffs
        tuned on an average protein invert on a family that is genuinely
        charge-rich, cysteine-free, or low-complexity - and such families are
        common (ribosomal proteins, halophilic enzymes, coiled coils). Measuring
        each signal against the family's own distribution asks the question that
        actually matters: *is this sequence unusual for this family*, rather
        than *is this sequence unusual for proteins in general*.

        Pass `reference_sequences` with held-out natural homologues if you have
        them; otherwise the profile's own members are used, which slightly
        understates the spread.
        """
        pool = list(reference_sequences) if reference_sequences else list(self._members)
        if len(pool) < 10:
            LOG.info("Too few reference sequences for self-calibration; skipping.")
            self.feature_reference = {}
            return self.feature_reference
        if len(pool) > max_sequences:
            pool = random.Random(self.random_state + seed).sample(pool, max_sequences)

        rows: Dict[str, List[float]] = {}
        for cs in pool:
            feats, _ = extractor(cs, fingerprint, self)
            for k, v in feats.items():
                rows.setdefault(k, []).append(v)

        ref: Dict[str, Tuple[float, float]] = {}
        skip = {s.key for s in FEATURE_SPECS if s.self_calibrated}
        for k, vals in rows.items():
            if k in skip:
                continue
            arr = np.asarray(vals, dtype=float)
            ref[k] = (float(np.mean(arr)), float(np.std(arr)))
        self.feature_reference = ref
        self.feature_reference_n = len(pool)
        self.feature_reference_holdout = reference_sequences is not None
        self._reference_ids = {hash(cs.seq) for cs in pool}
        LOG.info(
            "Self-calibrated %d signals on %d %s sequences.",
            len(ref), len(pool),
            "held-out" if reference_sequences is not None else "member",
        )
        return ref

    # -- persistence --------------------------------------------------------

    def save(self, directory: str) -> None:
        if self.hmm is None:
            raise ModelNotBuiltError("Nothing to save; build the profile first.")
        os.makedirs(directory, exist_ok=True)
        if not self.hmm.command_line:
            self.hmm.command_line = f"mpnn_detect {__version__}"
        with open(os.path.join(directory, "family.hmm"), "wb") as fh:
            self.hmm.write(fh)
        with gzip.open(os.path.join(directory, "members.fasta.gz"), "wt") as fh:
            for s in self._members:
                fh.write(f">{s.name}\n{s.seq}\n")
        if self._member_matrix is not None:
            np.savez_compressed(
                os.path.join(directory, "members.npz"), matrix=self._member_matrix
            )
        if self.evo:
            np.savez_compressed(
                os.path.join(directory, "evolution.npz"),
                rate=self.evo["rate"], exchangeability=self.evo["exchangeability"],
                dist=self.evo["dist"],
                stats=np.array([self.evo["indel_slope"], self.evo["n_subst_pairs"]]),
                indel_bins=self.evo.get("indel_bins", np.zeros(1)),
                indel_curve=self.evo.get("indel_curve", np.zeros(1)),
                dist_corrected=self.evo.get("dist_corrected", np.zeros((1, 1))),
                inserts=(self._member_inserts if self._member_inserts is not None
                         else np.zeros(1)),
            )
        if self.coupling:
            # The DCA and MI fits carry different arrays, so save whichever
            # are present and record the method. Writing a fixed key set would
            # raise on a DCA profile and, worse, a partial write would reload
            # as an MI model scored with DCA statistics.
            payload = {
                "pairs": self.coupling["pairs"], "f1": self.coupling["f1"],
                "method": np.array([self.coupling.get("method", "mi")]),
                "stats": np.array([self.coupling["neff"],
                                   self.coupling["member_mean"],
                                   self.coupling["member_sd"],
                                   self.coupling.get("q", 20)]),
            }
            for key in ("e", "f2", "mi_apc", "di_apc"):
                if key in self.coupling:
                    payload[key] = self.coupling[key]
            np.savez_compressed(os.path.join(directory, "coupling.npz"), **payload)
        meta = {
            "version": __version__,
            "name": self.name,
            "random_state": self.random_state,
            "n_sequences": self.n_sequences,
            "M": int(self.hmm.M),
            "feature_reference": {k: list(v) for k, v in self.feature_reference.items()},
            "feature_reference_n": self.feature_reference_n,
            "feature_reference_holdout": self.feature_reference_holdout,
            "placement": self.placement,
            "meff": self.meff,
        }
        with open(os.path.join(directory, "profile.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        LOG.info("Profile saved to %s", directory)

    @classmethod
    def load(cls, directory: str) -> "FamilyProfile":
        with open(os.path.join(directory, "profile.json")) as fh:
            meta = json.load(fh)
        obj = cls(name=meta["name"], random_state=meta.get("random_state", 0))
        with pyhmmer.plan7.HMMFile(os.path.join(directory, "family.hmm")) as hf:
            obj.hmm = hf.read()
        obj.n_sequences = meta["n_sequences"]
        obj.feature_reference = {
            k: (float(v[0]), float(v[1]))
            for k, v in meta.get("feature_reference", {}).items()
        }
        obj.feature_reference_n = meta.get("feature_reference_n", 0)
        obj.feature_reference_holdout = meta.get("feature_reference_holdout", False)
        obj.placement = meta.get("placement")
        obj.meff = meta.get("meff")
        members_path = os.path.join(directory, "members.fasta.gz")
        if os.path.exists(members_path):
            obj._members = [
                CleanSequence(seq=s, name=n) for n, s in parse_fasta(members_path)
            ]
            obj._member_kmers = [kmer_set(s.seq, 4) for s in obj._members]
        obj._compute_position_statistics()
        matrix_path = os.path.join(directory, "members.npz")
        if os.path.exists(matrix_path):
            obj._member_matrix = np.load(matrix_path)["matrix"]
        elif obj._members:
            obj._index_members()
        evo_path = os.path.join(directory, "evolution.npz")
        if os.path.exists(evo_path):
            z = np.load(evo_path)
            slope, npairs = z["stats"]
            obj.evo = {
                "rate": z["rate"], "exchangeability": z["exchangeability"],
                "dist": z["dist"], "indel_slope": float(slope),
                "n_subst_pairs": int(npairs),
                "indel_bins": z["indel_bins"], "indel_curve": z["indel_curve"],
                "dist_corrected": z["dist_corrected"],
            }
            obj._member_inserts = z["inserts"]
            LOG.info("Loaded evolutionary model (%d substitution pairs).", int(npairs))
        coupling_path = os.path.join(directory, "coupling.npz")
        if os.path.exists(coupling_path):
            z = np.load(coupling_path)
            stats = z["stats"]
            neff, mean, sd = stats[0], stats[1], stats[2]
            q = int(stats[3]) if len(stats) > 3 else 20
            method = str(z["method"][0]) if "method" in z else "mi"
            obj.coupling = {
                "method": method, "pairs": z["pairs"], "f1": z["f1"],
                "neff": float(neff), "member_mean": float(mean),
                "member_sd": float(sd), "q": q,
            }
            for key in ("e", "f2", "mi_apc", "di_apc"):
                if key in z:
                    obj.coupling[key] = z[key]
            LOG.info("Loaded %d %s-coupled pairs.", len(z["pairs"]), method.upper())
        LOG.info("Profile %r loaded (%d positions).", obj.name, obj.hmm.M)
        return obj


# ===========================================================================
# 7. Feature extraction
# ===========================================================================


@dataclass(frozen=True)
class FeatureSpec:
    """One signal: how to centre/scale it and how to describe it in a report.

    `evidence_key` names the count this signal's reliability depends on, and
    `evidence_ref` is the count at which it is fully trusted. A composition
    log-odds is a mean over residues, so its standard error shrinks as
    1/sqrt(L); trusting a 60-residue estimate as much as a 400-residue one is
    how a detector ends up confidently wrong about short sequences. Signals are
    attenuated by sqrt(min(1, n / evidence_ref)) accordingly.
    """

    key: str
    center: float
    scale: float
    needs_profile: bool
    description: str
    evidence_key: Optional[str] = None
    evidence_ref: float = 1.0
    #: True when the signal is already a z-score against its own out-of-fold
    #: reference. Re-standardising such a signal against profile members double
    #: counts the calibration and makes every unseen natural look like an
    #: outlier, because members were scored on statistics fit to themselves.
    self_calibrated: bool = False
    #: True when a family profile makes this signal redundant AND harmful. An
    #: absolute comparison against the global background is anti-discriminative
    #: whenever the family itself departs from that background, and the
    #: family-relative version of the same measurement is strictly better.
    superseded_by_profile: bool = False
    #: True when the signal is computed against a generator fingerprint. Those
    #: signals are the strongest when the generator is known and worthless when
    #: it is not, so they are dropped in generator-agnostic mode rather than
    #: silently scored against the wrong prior.
    needs_fingerprint: bool = False
    #: Which evidence layer this signal belongs to. Kept on the spec so a
    #: caller cannot mislabel it - a hand-written lookup in a report silently
    #: filed the phylogenetic placement signals under "composition".
    layer: str = "composition"
    #: True when only how unusual the value is matters, not which way it departs.
    #: An idealised de novo scaffold is extreme; a redesign on a natural
    #: backbone is typical. Forcing a direction on such a signal makes it fire
    #: backwards on families that sit on the far side of the global average.
    two_sided: bool = False
    #: True when the signal requires the query to be aligned to the profile.
    #: Signals that only need the family's reference composition do not, and so
    #: remain usable on short windows where alignment is unreliable.
    needs_alignment: bool = False


FEATURE_SPECS: Tuple[FeatureSpec, ...] = (
    FeatureSpec("comp_llr_background", 0.020, 0.025, False,
                "composition matches the generator rather than UniProt background",
                evidence_key="length", evidence_ref=250.0,
                superseded_by_profile=True,
                needs_fingerprint=True),
    FeatureSpec("comp_llr_family", 0.000, 0.025, True,
                "composition matches the generator rather than this protein family",
                evidence_key="length", evidence_ref=250.0,
                needs_fingerprint=True),
    FeatureSpec("cys_depletion", 0.60, 1.20, False,
                "cysteines missing relative to the expected count",
                evidence_key="length", evidence_ref=250.0,
                needs_fingerprint=True),
    FeatureSpec("rare_depletion", 1.20, 1.20, False,
                "C/M/W/H jointly depleted",
                evidence_key="length", evidence_ref=250.0,
                needs_fingerprint=True),
    FeatureSpec("charge_excess", 0.030, 0.035, False,
                "excess charged surface residues (D/E/K/R)",
                evidence_key="length", evidence_ref=250.0,
                needs_fingerprint=True),
    FeatureSpec("low_complexity", 0.070, 0.045, False,
                "residue alphabet narrower than natural proteins",
                evidence_key="length", evidence_ref=150.0,
                needs_fingerprint=True),
    FeatureSpec("kmer_repeat", 0.040, 0.090, False,
                "repeated sequence motifs typical of idealised scaffolds",
                two_sided=True),
    FeatureSpec("hydrophobic_moment", 0.520, 0.110, False,
                "unusually clean amphipathic helix faces", two_sided=True),
    FeatureSpec("variable_llr", -0.020, 0.055, True,
                "surface positions carry generator-typical residues",
                evidence_key="residual_n_variable", evidence_ref=40.0,
                needs_alignment=True,
                needs_fingerprint=True,
                layer="structural"),
    FeatureSpec("exposed_llr", -0.020, 0.060, True,
                "solvent-exposed positions carry generator-typical residues",
                evidence_key="residual_n_exposed", evidence_ref=30.0,
                needs_alignment=True,
                needs_fingerprint=True,
                layer="structural"),
    FeatureSpec("stratum_contrast", 0.000, 0.080, True,
                "surface rewritten while the buried core is left intact",
                evidence_key="residual_n_exposed", evidence_ref=30.0,
                needs_alignment=True,
                needs_fingerprint=True,
                layer="structural"),
    FeatureSpec("coupling_violation", 0.000, 1.000, True,
                "breaks position couplings that the natural family preserves",
                evidence_key="coupling_n_pairs", evidence_ref=100.0,
                self_calibrated=True, needs_alignment=True,
                layer="covariation"),
    FeatureSpec("subst_implausibility", 0.000, 1.000, True,
                "substitutions this family does not normally interchange",
                evidence_key="n_substitutions", evidence_ref=25.0,
                needs_alignment=True,
                layer="evolutionary"),
    FeatureSpec("directional_drift", 0.000, 0.300, True,
                "net drift toward the generator's equilibrium, not reversible change",
                evidence_key="n_substitutions", evidence_ref=25.0,
                needs_alignment=True,
                needs_fingerprint=True,
                layer="evolutionary"),
    FeatureSpec("indel_deficit", 0.000, 1.500, True,
                "diverged from its nearest relative without the indels that implies",
                evidence_key="n_substitutions", evidence_ref=25.0,
                needs_alignment=True,
                layer="evolutionary"),
    FeatureSpec("rate_incongruence", 0.000, 0.200, True,
                "changes fall where the family does not actually vary",
                evidence_key="n_substitutions", evidence_ref=25.0,
                needs_alignment=True,
                layer="evolutionary"),
    FeatureSpec("equilibrium_drift", 0.000, 0.120, True,
                "net substitution flow away from the family's own equilibrium",
                evidence_key="n_substitutions", evidence_ref=25.0,
                needs_alignment=True,
                layer="evolutionary"),
    FeatureSpec("pendant_excess", 0.000, 1.000, True,
                "sits further from its inferred ancestor than a real relative does",
                evidence_key="n_placement_substitutions", evidence_ref=25.0,
                needs_alignment=True, self_calibrated=True,
                layer="phylogenetic"),
    FeatureSpec("ancestral_implausibility", 0.000, 1.000, True,
                "substitutions from the inferred ancestor are ones this family avoids",
                evidence_key="n_placement_substitutions", evidence_ref=25.0,
                needs_alignment=True, self_calibrated=True,
                layer="phylogenetic"),
    FeatureSpec("tree_violation", 0.000, 0.100, True,
                "distorts the four-point condition the family's tree satisfies",
                evidence_key="n_quartets", evidence_ref=100.0,
                needs_alignment=True,
                layer="evolutionary"),
    FeatureSpec("orphan_fold", 0.380, 0.110, True,
                "matches the family fold but has no close natural relative",
                needs_alignment=True,
                layer="structural"),
    FeatureSpec("construct_motifs", 0.0, 1.0, False,
                "expression-construct motifs (tags, linkers, protease sites)"),
)
FEATURE_KEYS = tuple(f.key for f in FEATURE_SPECS)
SPEC_BY_KEY = {f.key: f for f in FEATURE_SPECS}

#: Hand-set prior weights, NOT fitted. Replace with Detector.calibrate().
DEFAULT_WEIGHTS: Dict[str, float] = {
    "comp_llr_background": 0.45,
    "comp_llr_family": 0.90,
    "cys_depletion": 0.40,
    "rare_depletion": 0.35,
    "charge_excess": 0.45,
    "low_complexity": 0.20,
    "kmer_repeat": 0.45,
    "hydrophobic_moment": 0.20,
    "variable_llr": 0.70,
    "exposed_llr": 0.90,
    "stratum_contrast": 0.80,
    "coupling_violation": 0.90,
    "subst_implausibility": 0.70,
    "directional_drift": 1.00,
    "indel_deficit": 0.60,
    "rate_incongruence": 0.70,
    "tree_violation": 0.50,
    "pendant_excess": 1.10,
    "ancestral_implausibility": 0.90,
    "equilibrium_drift": 0.70,
    "orphan_fold": 1.20,
    "construct_motifs": 0.90,
}
DEFAULT_BIAS = -1.6
#: Composition-only mode carries less evidence, so it needs its own intercept.
DEFAULT_BIAS_NO_PROFILE = -1.2


def extract_features(
    cs: CleanSequence,
    fingerprint: GeneratorFingerprint,
    profile: Optional["FamilyProfile"] = None,
    alignment: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute every signal. Returns (features, diagnostics).

    Features are oriented so that larger always means "more likely designed".
    Diagnostics are raw measurements kept for the human-readable report.
    """
    seq = cs.seq
    L = len(seq)
    comp = composition(seq)
    counts = Counter(seq)

    feats: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}
    diag: Dict[str, float] = {"length": float(L)}

    # --- composition -------------------------------------------------------
    feats["comp_llr_background"] = fingerprint.llr_per_residue(comp, NATURAL_BACKGROUND)

    # Expectations come from the family when one is available. A family that is
    # natively cysteine-free leaves no room for a generator to deplete further,
    # and scoring it against the global background would report that absence as
    # evidence of design.
    reference = (
        profile.family_composition
        if profile is not None and getattr(profile, "family_composition", None) is not None
        else NATURAL_BACKGROUND
    )

    def orient(gen_value: float, ref_value: float, gap_scale: float) -> float:
        """Soft orientation toward the generator, instead of a hard sign flip.

        A hard sign is a coin toss when the generator and the family sit at
        nearly the same value on a measure: the signal then carries no real
        information but is still scored at full weight, and which way it points
        can change from one profile build to the next. Scaling by tanh of the
        gap sends the signal smoothly to zero as that gap closes, so a
        measurement that cannot discriminate contributes nothing rather than
        noise at full strength.
        """
        return math.tanh((gen_value - ref_value) / gap_scale)

    exp_c = reference[AA_INDEX["C"]] * L
    obs_c = counts.get("C", 0)
    cys_dir = -orient(float(fingerprint.freqs[AA_INDEX["C"]]),
                      float(reference[AA_INDEX["C"]]), 0.004)
    feats["cys_depletion"] = (exp_c - obs_c) / math.sqrt(exp_c + 1.0) * cys_dir
    diag["cysteines"] = float(obs_c)
    diag["cysteines_expected"] = float(exp_c)

    rare = "CMWH"
    exp_r = sum(reference[AA_INDEX[a]] for a in rare) * L
    gen_r = sum(fingerprint.freqs[AA_INDEX[a]] for a in rare) * L
    obs_r = sum(counts.get(a, 0) for a in rare)
    rare_dir = -orient(gen_r / max(L, 1), exp_r / max(L, 1), 0.012)
    feats["rare_depletion"] = (exp_r - obs_r) / math.sqrt(exp_r + 1.0) * rare_dir

    # Signed toward the generator rather than simply "more charged". Assuming
    # designs are always the charged ones inverts on a family that is already
    # more charged than the generator - and such families exist. Orienting by
    # the generator keeps the sign correct whichever way the gap runs.
    charged = sum(comp[AA_INDEX[a]] for a in "DEKR")
    ref_charged = sum(reference[AA_INDEX[a]] for a in "DEKR")
    gen_charged = sum(fingerprint.freqs[AA_INDEX[a]] for a in "DEKR")
    direction = orient(float(gen_charged), float(ref_charged), 0.015)
    feats["charge_excess"] = float((charged - ref_charged) * direction)
    diag["charged_fraction"] = float(charged)
    diag["charged_fraction_reference"] = float(ref_charged)

    # Oriented toward the generator, like charge above. Whether a generator
    # narrows or widens the residue alphabet relative to a given family depends
    # on the family; assuming "designs are always the low-complexity ones"
    # inverts on a family that is natively repetitive.
    ent = shannon_entropy(seq)
    ref_ent = float(-(reference * np.log2(np.clip(reference, 1e-12, None))).sum())
    gen_ent = float(-(fingerprint.freqs
                      * np.log2(np.clip(fingerprint.freqs, 1e-12, None))).sum())
    ent_direction = -orient(gen_ent, ref_ent, 0.12)
    feats["low_complexity"] = (ref_ent - ent) / math.log2(20) * ent_direction
    diag["entropy_direction"] = ent_direction
    diag["entropy_bits"] = ent

    feats["kmer_repeat"] = kmer_repetition(seq, k=6)
    diag["max_homopolymer"] = float(max_homopolymer_run(seq))
    feats["hydrophobic_moment"] = hydrophobic_moment(seq)
    diag["hydrophobic_moment"] = feats["hydrophobic_moment"]

    motifs = find_construct_motifs(seq)
    feats["construct_motifs"] = 1.0 if motifs else 0.0
    diag["n_construct_motifs"] = float(len(motifs))

    # Family composition is usable even on a short window; only the aligned
    # signals below need the query mapped onto model positions.
    if profile is not None and getattr(profile, "family_composition", None) is not None:
        feats["comp_llr_family"] = fingerprint.llr_per_residue(
            comp, profile.family_composition
        )

    # --- alignment-dependent -----------------------------------------------
    if alignment and profile is not None and profile.hmm is not None:
        hit = profile.score(cs)
        diag["profile_bits"] = hit.bits
        diag["profile_evalue"] = hit.evalue
        diag["profile_coverage"] = hit.coverage

        res = profile.residual_analysis(hit, fingerprint)
        feats["variable_llr"] = res["variable_llr"]
        feats["exposed_llr"] = res["exposed_llr"]
        feats["stratum_contrast"] = res["stratum_contrast"]
        diag.update({f"residual_{k}": v for k, v in res.items()})

        evo = profile.evolutionary_signals(cs, fingerprint)
        for k in ("subst_implausibility", "directional_drift", "equilibrium_drift",
                  "indel_deficit", "rate_incongruence", "tree_violation"):
            feats[k] = evo[k]
        diag["n_substitutions"] = evo["n_substitutions"]
        diag["n_quartets"] = evo["n_quartets"]
        diag.update({f"evo_{k}": v for k, v in evo.items()})

        place = profile.placement_signals(cs)
        feats["pendant_excess"] = place["pendant_excess"]
        feats["ancestral_implausibility"] = place["ancestral_implausibility"]
        diag["n_placement_substitutions"] = place["n_placement_substitutions"]
        diag.update({f"placement_{k}": v for k, v in place.items()})

        cv, n_pairs = profile.coupling_violation(cs)
        feats["coupling_violation"] = cv
        diag["coupling_n_pairs"] = float(n_pairs)
        diag["coupling_violation_z"] = cv

        identity = profile.nearest_neighbour_identity(cs)
        diag["nearest_identity"] = identity
        # "Fold match without a relative": only meaningful when the fold is
        # actually matched, so gate the isolation term on core agreement.
        fold_match = 1.0 / (1.0 + math.exp(-(res["core_agreement"] - 0.5) * 2.0))
        feats["orphan_fold"] = fold_match * max(0.0, 1.0 - identity)
        diag["fold_match_strength"] = fold_match

    return feats, diag


# ===========================================================================
# 8. Detector
# ===========================================================================


@dataclass
class SignalContribution:
    key: str
    raw: float
    standardized: float
    weight: float
    contribution: float
    description: str
    reference: Optional[Tuple[float, float]] = None


@dataclass
class DetectionResult:
    name: str
    length: int
    verdict: str
    p_designed: float
    score: float
    generator_guess: Optional[str]
    generator_scores: Dict[str, float]
    signals: List[SignalContribution]
    diagnostics: Dict[str, float]
    flags: List[str]
    used_profile: bool
    calibrated: bool
    generator_agnostic: bool = False
    family: Optional[str] = None
    q_value: Optional[float] = None
    fdr_significant: Optional[bool] = None
    p_value: Optional[float] = None
    p_value_extrapolated: bool = False
    fragility: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [asdict(s) for s in self.signals]
        return d

    def report(self, top_n: int = 6) -> str:
        """Human-readable explanation. Every verdict should be auditable."""
        lines = [
            f"{self.name}  ({self.length} aa)",
            "-" * 66,
            f"Verdict           : {self.verdict}",
            f"P(designed)       : {self.p_designed:.3f}"
            + ("" if self.calibrated else "   [UNCALIBRATED - prior weights]"),
        ]
        if self.family:
            lines.append(f"Routed to family  : {self.family}")
        if self.q_value is not None:
            lines.append(
                f"q (FDR-adjusted)  : {self.q_value:.4g}"
                f"   {'passes' if self.fdr_significant else 'does not pass'} the FDR cut"
            )
        if self.p_value is not None:
            tag = " [tail-extrapolated]" if self.p_value_extrapolated else ""
            lines.append(
                f"p vs natural null : {self.p_value:.4g}{tag}"
                "   (fraction of known-natural sequences scoring this high)"
            )
        if self.fragility:
            lines.append(f"Robustness        : {self.fragility}")
        if self.generator_guess:
            lines.append(
                f"Closest generator : {self.generator_guess} "
                f"({self.generator_scores.get(self.generator_guess, 0):+.3f} nats/res)"
            )
        if not self.used_profile:
            lines.append("Mode              : composition-only (no family profile)")
        if self.generator_agnostic:
            lines.append("Mode              : generator-agnostic (no fingerprint assumed)")
        lines.append("")
        lines.append("Strongest signals:")
        ordered = sorted(self.signals, key=lambda s: -abs(s.contribution))[:top_n]
        for s in ordered:
            arrow = "designed" if s.contribution > 0 else "natural "
            ref = (
                f"  vs family {s.reference[0]:+.3f}+-{s.reference[1]:.3f}"
                if s.reference else ""
            )
            lines.append(
                f"  {arrow} {s.contribution:+6.2f}  {s.key:<22} "
                f"raw={s.raw:+.3f}{ref}  ({s.description})"
            )
        if self.flags:
            lines.append("")
            lines.append("Flags: " + ", ".join(self.flags))
        if self.notes:
            lines.append("")
            for n in self.notes:
                lines.append(f"Note: {n}")
        return "\n".join(lines)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 200
) -> Tuple[np.ndarray, float]:
    """L2-regularised logistic regression by IRLS. Keeps sklearn optional."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0  # do not penalise the intercept
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) + reg @ w
        s = np.clip(p * (1 - p), 1e-6, None)
        H = Xb.T @ (Xb * s[:, None]) + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[:-1], float(w[-1])


class Detector:
    """Combines the signals into a calibrated probability.

    Out of the box it runs on hand-set prior weights, which are useful for
    triage and ranking but should not be read as probabilities. Run
    `calibrate()` with real natural and real generator output before relying on
    the number.
    """

    def __init__(
        self,
        profile: Optional["FamilyProfile"] = None,
        fingerprint: GeneratorFingerprint = PROTEINMPNN_PRIOR,
        weights: Optional[Dict[str, float]] = None,
        bias: Optional[float] = None,
        threshold_designed: float = 0.70,
        threshold_natural: float = 0.30,
        alpha: float = 0.01,
        generator_agnostic: bool = False,
    ):
        self.profile = profile
        self.fingerprint = fingerprint
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.bias = bias
        self.threshold_designed = threshold_designed
        self.threshold_natural = threshold_natural
        self.calibrated = weights is not None
        self.calibration_info: Dict[str, object] = {}
        self.alpha = alpha
        # Drops every signal that is measured against a generator fingerprint.
        # What remains asks only whether the sequence could have descended from
        # the family, which is a question that does not require knowing - or
        # guessing - which model produced it. This is the mode to use on a
        # generator that has not been fingerprinted, and the honest default when
        # provenance is genuinely unknown.
        self.generator_agnostic = generator_agnostic
        self.null_scores: Optional[np.ndarray] = None
        self.null_info: Dict[str, object] = {}
        self._tail: Optional[Tuple[float, float, float]] = None
        self.window_null: Optional[np.ndarray] = None
        self.window_params: Dict[str, int] = {"window": 80, "step": 20}
        self._local: Optional["Detector"] = None
        self._windowed = False
        self.validation_info: Dict[str, object] = {}

    # -- prediction ---------------------------------------------------------

    def _active_keys(self) -> List[str]:
        have_profile = self.profile is not None and self.profile.hmm is not None
        windowed = getattr(self, "_windowed", False)
        return [
            f.key for f in FEATURE_SPECS
            if (have_profile or not f.needs_profile)
            and not (windowed and f.needs_alignment)
            and not (have_profile and f.superseded_by_profile)
            and not (self.generator_agnostic and f.needs_fingerprint)
        ]

    @staticmethod
    def _attenuation(spec: FeatureSpec, diag: Dict[str, float]) -> float:
        """How much to trust this signal given how much data supports it."""
        if spec.evidence_key is None:
            return 1.0
        n = float(diag.get(spec.evidence_key, 0.0))
        return math.sqrt(min(1.0, max(0.0, n) / spec.evidence_ref))

    def _center_scale(self, key: str) -> Tuple[float, float, bool]:
        """Prefer the family's own distribution over the global defaults.

        A fixed cutoff calibrated on an average protein inverts on a family that
        is genuinely charge-rich or cysteine-free; the family reference asks
        instead whether the query is unusual *for this family*. The spec values
        remain the fallback for composition-only mode.
        """
        spec = SPEC_BY_KEY[key]
        if spec.self_calibrated:
            return spec.center, spec.scale, False
        ref = getattr(self.profile, "feature_reference", None) if self.profile else None
        if ref and key in ref:
            mu, sd = ref[key]
            # A signal with no spread among real members (every member scores
            # zero) would turn any deviation into a huge z-score, so floor the
            # scale at a fraction of the global one.
            return mu, max(sd, 0.25 * spec.scale), True
        return spec.center, spec.scale, False

    def _standardize(
        self, feats: Dict[str, float], keys: Sequence[str],
        diag: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        diag = diag or {}
        out = []
        for k in keys:
            spec = SPEC_BY_KEY[k]
            center, scale, _ = self._center_scale(k)
            z = (feats[k] - center) / scale
            if spec.two_sided:
                # 0.8 is the mean |z| of a standard normal, so a typical family
                # member still lands near zero after folding.
                z = abs(z) - 0.8
            out.append(z * self._attenuation(spec, diag))
        return np.array(out)

    def fit_null(self, natural_sequences: Sequence, holdout: bool = True) -> Dict[str, object]:
        """Record the score distribution on sequences known to be natural.

        This is what turns the output into a usable screening decision. The
        logistic probability is a ranking score whose absolute value depends on
        the intercept and on how many signals happen to be available; a
        threshold of 0.7 on it means nothing in particular. The empirical
        quantile of the natural distribution means something exact: set
        alpha=0.01 and one natural sequence in a hundred is flagged, by
        construction, whatever the family and whatever the feature set.

        Use held-out homologues rather than the profile's own members where you
        can - members were used to fit the very statistics being scored, so
        their scores are optimistic and the resulting threshold runs loose.
        """
        scores, used = [], []
        for s in natural_sequences:
            try:
                cs = s if isinstance(s, CleanSequence) else clean_sequence(s)
            except SequenceError:
                continue
            used.append(cs)
            scores.append(self._score_only(cs))
        if len(scores) < 20:
            raise ValueError(
                f"Only {len(scores)} usable null sequences; need at least 20 "
                "for a meaningful quantile (and far more for a small alpha)."
            )
        ref_ids = getattr(self.profile, "_reference_ids", set()) if self.profile else set()
        overlap = sum(1 for s in used if hash(s.seq) in ref_ids)
        self.null_scores = np.sort(np.asarray(scores, dtype=float))
        self._fit_tail()
        n = len(self.null_scores)
        resolution = 1.0 / (n + 1)
        # The smallest p the empirical null can produce is 1/(n+1). An alpha
        # below that is not a threshold the data can place - every call past it
        # comes from the tail fit. Asking for five null sequences above the
        # threshold keeps it anchored in observations.
        recommended = 5.0 / (n + 1)
        self.null_info = {
            "n": n,
            "mean": float(self.null_scores.mean()),
            "sd": float(self.null_scores.std()),
            "alpha": self.alpha,
            "threshold": float(np.quantile(self.null_scores, 1 - self.alpha)),
            "holdout": bool(holdout),
            "resolution": resolution,
            "tail_extrapolation": self._tail is not None,
            "recommended_alpha": recommended,
        }
        if overlap:
            msg = (
                f"{overlap} of {n} null sequences were also used to fit the "
                "family feature reference. Their scores are optimistic, so the "
                "threshold will run loose and the true false-positive rate will "
                "exceed alpha. Use three disjoint sets: one to build the "
                "profile, one for the feature reference, one for the null."
            )
            self.null_info["overlap_warning"] = msg
            LOG.warning(msg)
        if recommended > self.alpha:
            if self._tail is not None:
                self.null_info["warning"] = (
                    f"alpha={self.alpha} is below what {n} null sequences can "
                    f"support (the empirical floor is {resolution:.3g}, and an "
                    f"alpha of {recommended:.3g} would be the smallest with five "
                    "null sequences above the threshold). Every call past the "
                    "floor comes from the tail fit, which costs detection: "
                    "measured on 433 real kinases, running at alpha=0.01 with a "
                    "47-sequence null detected 86% of designs where alpha=0.05 "
                    "detected 98%. Either raise alpha or enlarge the null."
                )
            else:
                self.null_info["warning"] = (
                    f"alpha={self.alpha} is finer than the null can resolve with "
                    f"{n} sequences (smallest reportable p is {resolution:.3g}), "
                    "and there is too little data to fit a tail. Nothing can be "
                    "flagged at this alpha - raise it or supply more sequences."
                )
            LOG.warning(self.null_info["warning"])
        LOG.info(
            "Null fitted on %d sequences; score threshold at alpha=%.3g is %.2f.",
            n, self.alpha, self.null_info["threshold"],
        )
        return self.null_info

    def _local_detector(self) -> "Detector":
        """Composition-only twin used for window scoring, built once."""
        if self._local is None:
            # Keep the profile - a window still benefits from the family's
            # reference composition, and dropping it was what made window scores
            # fall back to global cutoffs and invert on unusual families. Only
            # the alignment-dependent signals are removed.
            sub = Detector(
                profile=self.profile, fingerprint=self.fingerprint,
                generator_agnostic=self.generator_agnostic,
                weights={k: v for k, v in self.weights.items()
                         if not SPEC_BY_KEY[k].needs_alignment},
                bias=DEFAULT_BIAS_NO_PROFILE, alpha=self.alpha,
            )
            sub.calibrated = self.calibrated
            sub._windowed = True
            self._local = sub
        return self._local

    def _window_scores(self, cs: CleanSequence, window: int, step: int):
        """Yield (start, score) for each window, composition-only."""
        local = self._local_detector()
        for start in range(0, len(cs) - window + 1, step):
            sub = CleanSequence(seq=cs.seq[start : start + window],
                                name=f"{cs.name}:{start}")
            try:
                yield start, local._score_only(sub)
            except SequenceError:
                continue

    def fit_window_null(
        self, natural_sequences: Sequence, window: int = 80, step: int = 20
    ) -> Dict[str, object]:
        """Record the window-score distribution on natural sequences.

        Needed because a window is scored without the family profile, so it
        falls back to global cutoffs that invert on unusual families. The window
        null restores the same question the full detector asks: is this stretch
        unusual *for this family*.
        """
        scores: List[float] = []
        for s in natural_sequences:
            try:
                cs = s if isinstance(s, CleanSequence) else clean_sequence(s)
            except SequenceError:
                continue
            scores.extend(sc for _st, sc in self._window_scores(cs, window, step))
        if len(scores) < 50:
            raise ValueError(
                f"Only {len(scores)} windows collected; need at least 50. "
                "Supply more natural sequences or reduce the step size."
            )
        self.window_null = np.sort(np.asarray(scores, dtype=float))
        self.window_params = {"window": window, "step": step}
        LOG.info("Window null fitted on %d windows (w=%d, step=%d).",
                 len(scores), window, step)
        return {"n_windows": len(scores), "window": window, "step": step}

    def window_p_value(self, score: float) -> Optional[float]:
        if self.window_null is None:
            return None
        n = len(self.window_null)
        n_ge = int(n - np.searchsorted(self.window_null, score, side="left"))
        return float((1 + n_ge) / (1 + n))

    def p_value(self, score: float) -> Optional[float]:
        """Fraction of the natural null scoring at least this high.

        Beyond the largest observed null score the empirical estimate floors at
        1/(n+1), which silently caps how extreme any result can look: with 60
        null sequences nothing can ever beat alpha=0.01, so a sequence a hundred
        sigma out still comes back ambiguous. An exponential tail fitted to the
        upper quantile extends the range past that floor. Extrapolated values
        are approximate and are marked as such on the result.
        """
        if self.null_scores is None:
            return None
        n = len(self.null_scores)
        n_ge = int(n - np.searchsorted(self.null_scores, score, side="left"))
        empirical = float((1 + n_ge) / (1 + n))
        if n_ge > 0 or self._tail is None:
            return empirical
        threshold, beta, q_tail = self._tail
        if beta <= 0:
            return empirical
        extrapolated = q_tail * math.exp(-(score - threshold) / beta)
        return float(min(empirical, max(extrapolated, 1e-12)))

    def _fit_tail(self) -> None:
        """Exponential fit to the upper tail of the null, for p < 1/(n+1)."""
        self._tail = None
        if self.null_scores is None or len(self.null_scores) < 30:
            return
        q_tail = 0.20
        threshold = float(np.quantile(self.null_scores, 1 - q_tail))
        excess = self.null_scores[self.null_scores > threshold] - threshold
        if len(excess) < 5:
            return
        beta = float(excess.mean())
        if beta > 0:
            self._tail = (threshold, beta, q_tail)

    def _score_only(self, cs: CleanSequence) -> float:
        have_profile = self.profile is not None and self.profile.hmm is not None
        feats, diag = extract_features(
            cs, self.fingerprint, self.profile,
            alignment=not getattr(self, "_windowed", False),
        )
        keys = self._active_keys()
        z = self._standardize(feats, keys, diag)
        w = np.array([self.weights.get(k, 0.0) for k in keys])
        bias = self.bias if self.bias is not None else (
            DEFAULT_BIAS if have_profile else DEFAULT_BIAS_NO_PROFILE
        )
        return float(z @ w) + bias

    def predict(self, sequence, name: str = "query") -> DetectionResult:
        cs = sequence if isinstance(sequence, CleanSequence) else clean_sequence(sequence, name=name)
        have_profile = self.profile is not None and self.profile.hmm is not None
        feats, diag = extract_features(
            cs, self.fingerprint, self.profile,
            alignment=not getattr(self, "_windowed", False),
        )

        keys = self._active_keys()
        z = self._standardize(feats, keys, diag)
        w = np.array([self.weights.get(k, 0.0) for k in keys])
        bias = self.bias if self.bias is not None else (
            DEFAULT_BIAS if have_profile else DEFAULT_BIAS_NO_PROFILE
        )
        score = float(z @ w) + bias
        p = _sigmoid(score)

        signals = [
            SignalContribution(
                key=k, raw=feats[k], standardized=float(zi), weight=float(wi),
                contribution=float(zi * wi), description=SPEC_BY_KEY[k].description,
                reference=(lambda c, s, fam: (c, s) if fam else None)(*self._center_scale(k)),
            )
            for k, zi, wi in zip(keys, z, w)
        ]

        comp = composition(cs.seq)
        reference = (
            self.profile.family_composition
            if have_profile and self.profile.family_composition is not None
            else NATURAL_BACKGROUND
        )
        guess, _, gen_scores = best_generator_match(comp, reference)

        flags: List[str] = []
        notes: List[str] = list(cs.notes)
        if diag.get("cysteines", 1) == 0 and len(cs) > 80:
            flags.append("no_cysteine")
        if diag.get("n_construct_motifs", 0) > 0:
            for key, desc, pos in find_construct_motifs(cs.seq):
                flags.append(f"{key}@{pos}")
        if have_profile:
            n_members = len(self.profile._members)
            if n_members < 100:
                flags.append("sparse_family")
                notes.append(
                    f"The family profile holds only {n_members} sequences. The "
                    "'no close natural relative' signal depends on the family "
                    "being well sampled, so genuine naturals with few deposited "
                    "homologues will look more isolated than they are. Expect "
                    "more false positives until the profile is broadened."
                )
            if diag.get("profile_coverage", 0.0) < 0.35:
                flags.append("poor_profile_coverage")
                notes.append(
                    "Query covers <35% of the family profile; profile-based "
                    "signals are unreliable. Is this the right family?"
                )
            if diag.get("residual_n_variable", 0) < 8:
                flags.append("few_variable_positions")
                notes.append(
                    "Fewer than 8 variable positions aligned; the surface-"
                    "composition signal is weak for this query."
                )
        if cs.x_fraction > 0.02:
            flags.append("ambiguous_residues")
        if have_profile and not getattr(self.profile, "feature_reference", None):
            notes.append(
                "Signals are scored against global defaults, not against this "
                "family. Run profile.fit_feature_reference(...) - on an unusual "
                "family (charge-rich, cysteine-free, low-complexity) the fixed "
                "cutoffs can point the wrong way."
            )
        elif have_profile and not self.profile.feature_reference_holdout:
            notes.append(
                f"Self-calibrated on {self.profile.feature_reference_n} profile "
                "members rather than held-out homologues, so the natural spread "
                "is slightly understated."
            )
        if not self.calibrated:
            notes.append(
                "Uncalibrated: weights are hand-set priors. Treat P(designed) "
                "as a ranking score, not a probability."
            )

        evidence = float(np.abs(z * w).sum())
        diag["evidence"] = evidence
        if evidence < 1.5:
            flags.append("low_evidence")
            notes.append(
                "Little evidence either way: the sequence is short, or it "
                "matches none of the registered fingerprints. Treat this as "
                "'not assessed' rather than as a negative result."
            )
            # Shrink toward the midpoint so a weak call cannot look confident.
            p = 0.5 + (p - 0.5) * (evidence / 1.5)

        pval = self.p_value(score)
        extrapolated = bool(
            pval is not None and self.null_scores is not None
            and score > self.null_scores[-1] and self._tail is not None
        )

        # How much of the verdict rests on its single largest contributor? A
        # call carried entirely by one signal is one blunted fingerprint away
        # from flipping, and the reader should be told that rather than left to
        # infer it from the breakdown.
        fragility = None
        if signals:
            top = max(signals, key=lambda s: abs(s.contribution))
            without = score - top.contribution
            flipped = (
                (score >= 0 and without < 0) or (score < 0 and without >= 0)
            )
            share = abs(top.contribution) / max(sum(abs(s.contribution) for s in signals), 1e-9)
            if flipped:
                fragility = (
                    f"fragile - the verdict rests on {top.key} alone "
                    f"({share:.0%} of total evidence) and flips without it"
                )
                flags.append("single_signal_call")
            elif share > 0.5:
                fragility = f"{top.key} carries {share:.0%} of the evidence, but the verdict holds without it"
            else:
                fragility = f"supported by multiple signals (largest is {top.key} at {share:.0%})"

        if pval is not None:
            if pval <= self.alpha:
                verdict = "LIKELY DESIGNED"
            elif pval <= 0.2:
                verdict = "AMBIGUOUS"
            else:
                verdict = "LIKELY NATURAL" if have_profile else "NO DESIGN SIGNAL"
        elif p >= self.threshold_designed:
            verdict = "LIKELY DESIGNED"
        elif p <= self.threshold_natural:
            # Without a family profile the only negative evidence available is
            # "does not match any registered fingerprint", which is not the same
            # as positive evidence of natural origin. Say what is actually known.
            verdict = "LIKELY NATURAL" if have_profile else "NO DESIGN SIGNAL"
        else:
            verdict = "AMBIGUOUS"

        return DetectionResult(
            name=cs.name, length=len(cs), verdict=verdict, p_designed=p, score=score,
            generator_guess=guess, generator_scores=gen_scores, signals=signals,
            diagnostics=diag, flags=flags, used_profile=have_profile,
            calibrated=self.calibrated, generator_agnostic=self.generator_agnostic,
            p_value=pval,
            p_value_extrapolated=extrapolated, fragility=fragility, notes=notes,
        )

    def predict_many(self, sequences: Iterable) -> List[DetectionResult]:
        out = []
        for s in sequences:
            try:
                out.append(self.predict(s))
            except SequenceError as exc:
                LOG.warning("Skipping sequence: %s", exc)
        return out

    # -- calibration --------------------------------------------------------

    def calibrate(
        self,
        natural: Sequence,
        designed: Sequence,
        l2: float = 1.0,
        holdout: float = 0.25,
        seed: int = 0,
    ) -> Dict[str, object]:
        """Fit weights on labelled data. This is the step that turns the score
        into something you can defend.

        Needs real sequences on both sides. 100+ per class is a reasonable
        minimum; below ~30 per class the fit will overfit and the reported
        hold-out metrics will be optimistic.
        """
        keys = self._active_keys()
        rows, labels = [], []
        for group, label in ((natural, 0.0), (designed, 1.0)):
            for s in group:
                try:
                    cs = s if isinstance(s, CleanSequence) else clean_sequence(s)
                except SequenceError as exc:
                    LOG.warning("Calibration input skipped: %s", exc)
                    continue
                feats, cdiag = extract_features(cs, self.fingerprint, self.profile)
                rows.append(self._standardize(feats, keys, cdiag))
                labels.append(label)

        X, y = np.array(rows), np.array(labels)
        if len(set(y.tolist())) < 2:
            raise ValueError("Calibration needs both natural and designed examples.")
        n_min = int(min((y == 0).sum(), (y == 1).sum()))
        if n_min < 30:
            LOG.warning(
                "Only %d examples in the smaller class; weights will overfit "
                "and hold-out metrics will be optimistic.", n_min,
            )

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = int(len(y) * (1 - holdout))
        tr, te = idx[:cut], idx[cut:]

        w, b = _fit_logistic(X[tr], y[tr], l2=l2)
        self.weights = {k: float(v) for k, v in zip(keys, w)}
        self.bias = b
        self.calibrated = True

        def _metrics(ii):
            if len(ii) == 0:
                return {}
            p = 1.0 / (1.0 + np.exp(-np.clip(X[ii] @ w + b, -30, 30)))
            yy = y[ii]
            pred = (p >= 0.5).astype(float)
            acc = float((pred == yy).mean())
            pos, neg = p[yy == 1], p[yy == 0]
            auc = (
                float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
            )
            return {"n": int(len(ii)), "accuracy": acc, "auroc": auc}

        info = {
            "n_natural": int((y == 0).sum()),
            "n_designed": int((y == 1).sum()),
            "features": keys,
            "weights": self.weights,
            "bias": self.bias,
            "train": _metrics(tr),
            "holdout": _metrics(te),
            "fingerprint": self.fingerprint.name,
            "fingerprint_confidence": self.fingerprint.confidence,
        }
        self.calibration_info = info
        if info["train"].get("auroc", 0) >= 0.999:
            info["warning"] = (
                "Perfect training separation. Either the two classes are "
                "genuinely trivial to tell apart, or something is leaking - "
                "designs simulated from the same fingerprint the detector "
                "tests for will always separate perfectly. Check the hold-out "
                "numbers and confirm on real generator output."
            )
            LOG.warning(info["warning"])
        self.calibration_info = info
        LOG.info(
            "Calibrated on %d natural / %d designed. Hold-out AUROC %.3f.",
            info["n_natural"], info["n_designed"],
            info["holdout"].get("auroc", float("nan")),
        )
        return info

    @staticmethod
    def _profile_fingerprint(profile: Optional["FamilyProfile"]) -> Optional[str]:
        """Identify the exact profile a null was fitted against.

        Scores are not comparable across profiles: a different model length,
        a different member set or a different alignment moves the whole score
        distribution. A null fitted on one profile therefore says nothing about
        a query scored on another, and pairing them silently invalidates every
        p-value downstream - which is exactly what happened in a reported run,
        where a saved detector with a 93-sequence null was reloaded against a
        freshly rebuilt profile that had 333 null sequences available.
        """
        if profile is None or profile.hmm is None:
            return None
        parts = [
            str(int(profile.hmm.M)),
            str(int(profile.n_sequences)),
            str(int(profile.hmm.checksum)),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def apply_validation(
        self, report: "ValidationReport", drop_uninformative: bool = True,
        flip_inverted: bool = False, reweight: bool = False,
    ) -> Dict[str, object]:
        """Use a validation report to prune the weights.

        The expensive part of this method is finding out which signals actually
        separate the classes on the family in front of you, and that answer
        should change the detector rather than only describe it. A signal whose
        interval spans chance is carrying no evidence here, and one that sits
        reliably below chance is carrying evidence backwards - measured on real
        PF00069 kinases, `stratum_contrast` scored 0.374 while the same signal
        scored above 0.95 on a simulated family.

        A note on which report to use. It is tempting to validate against the
        hardest designs you expect - composition-matched ones, say - so the
        weights are tuned for that threat. Measured on 3,544 real PF00069
        kinases over 8 independent design samples, that is slightly worse:
        reweighting from raw generator output detected 60.0% of
        composition-matched designs against 56.7% from a report computed on
        matched designs, winning 7 of 8 paired samples. The report's job is to
        RANK the signals, and harder designs separate less, so they estimate
        that ranking less precisely. Use the cleanest designs you have.

        Dropping is the conservative action: it sets the weight to zero, so the
        signal stops contributing instead of being re-pointed on the strength
        of one family. `flip_inverted` is offered for completeness but is off
        by default, because a signal that is backwards on one family is more
        likely mis-specified than sign-flipped, and flipping it bakes this
        family's quirk into the model.
        """
        dropped, flipped, rescaled = [], [], {}
        if reweight:
            # Zeroing the dead signals is not the whole problem. Every signal
            # contributes variance to the aggregate score, so one that does not
            # discriminate still widens the natural null and therefore raises
            # the threshold - which costs detection even while the ranking
            # (AUROC) stays high. Measured on real kinases with designs whose
            # composition was matched to the family, the full detector missed
            # 94% of them at alpha=0.01 while the evolutionary signals alone
            # missed 48%, at an AUROC of 0.972: the ranking was fine and the
            # threshold was not.
            #
            # Scaling each weight by the signal's measured separation shrinks
            # the noisy contributors instead of trusting a hand-set prior.
            for s in report.signals:
                if s.key not in self.weights:
                    continue
                # Scale by how far the signal is from chance, not by where it
                # sits. `max(0, 2*(auroc-0.5))` collapses every inverted signal
                # to exactly zero, and a zero weight cannot be flipped
                # afterwards: `-abs(0.0)` is `0.0`. So reweight=True silently
                # cancelled flip_inverted=True, and an inverted signal carrying
                # real evidence - rate_incongruence measured 0.163 against real
                # ProteinMPNN output, as informative reversed as 0.837 - was
                # discarded either way. Magnitude here, direction below.
                gain = 2.0 * abs(s.auroc_mean - 0.5)
                self.weights[s.key] = float(self.weights[s.key] * gain)
                rescaled[s.key] = round(gain, 4)
        for s in report.signals:
            if s.key not in self.weights:
                continue
            if s.informative:
                continue
            inverted = s.verdict.startswith("inverted")
            if inverted and flip_inverted:
                self.weights[s.key] = -abs(self.weights[s.key])
                flipped.append(s.key)
            elif inverted:
                # An inverted signal must never keep a positive weight. With
                # reweighting on, its magnitude has just been scaled UP by how
                # informative it is, so leaving the sign alone would make the
                # most strongly backwards signal the loudest wrong voice.
                self.weights[s.key] = 0.0
                dropped.append(s.key)
            elif drop_uninformative:
                self.weights[s.key] = 0.0
                dropped.append(s.key)
        self.validation_info = {
            "n_repeats": report.n_repeats, "dropped": dropped, "flipped": flipped,
            "reweighted": rescaled,
            "kept": [s.key for s in report.signals if s.informative],
        }
        LOG.info(
            "Pruned by validation: %d signal(s) zeroed (%s), %d kept%s.",
            len(dropped), ", ".join(dropped) or "none",
            len(self.validation_info["kept"]),
            f", {len(rescaled)} rescaled by measured separation" if rescaled else "",
        )
        return self.validation_info

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(
                {
                    "version": __version__,
                    "weights": self.weights,
                    "bias": self.bias,
                    "calibrated": self.calibrated,
                    "fingerprint": self.fingerprint.name,
                    "fingerprint_freqs": self.fingerprint.freqs.tolist(),
                    "fingerprint_confidence": self.fingerprint.confidence,
                    "threshold_designed": self.threshold_designed,
                    "threshold_natural": self.threshold_natural,
                    "calibration_info": self.calibration_info,
                    "validation_info": self.validation_info,
                    "profile_fingerprint": self._profile_fingerprint(self.profile),
                    "alpha": self.alpha,
                    "generator_agnostic": self.generator_agnostic,
                    "null_scores": (
                        self.null_scores.tolist()
                        if self.null_scores is not None else None
                    ),
                    "null_info": self.null_info,
                    "window_null": (
                        self.window_null.tolist()
                        if self.window_null is not None else None
                    ),
                    "window_params": self.window_params,
                },
                fh, indent=2, default=str,
            )

    @classmethod
    def load(cls, path: str, profile: Optional["FamilyProfile"] = None) -> "Detector":
        with open(path) as fh:
            d = json.load(fh)
        fp = GeneratorFingerprint(
            name=d["fingerprint"],
            freqs=np.array(d["fingerprint_freqs"]),
            confidence=d.get("fingerprint_confidence", "loaded"),
        )
        det = cls(
            profile=profile, fingerprint=fp, weights=d["weights"], bias=d["bias"],
            threshold_designed=d.get("threshold_designed", 0.70),
            threshold_natural=d.get("threshold_natural", 0.30),
        )
        det.calibrated = d.get("calibrated", False)
        det.calibration_info = d.get("calibration_info", {})
        det.validation_info = d.get("validation_info", {})
        saved_fp = d.get("profile_fingerprint")
        live_fp = cls._profile_fingerprint(profile)
        if saved_fp and live_fp and saved_fp != live_fp:
            raise ModelNotBuiltError(
                f"This detector was fitted against a different profile "
                f"(fingerprint {saved_fp} vs {live_fp}). Its null, threshold "
                "and per-family references belong to that profile, and scores "
                "are not comparable across profiles - reusing it here would "
                "make every p-value meaningless. Refit with "
                "Detector(profile=...).fit_null(...) on this profile's own "
                "held-out sequences."
            )
        if saved_fp is None and live_fp is not None and d.get("null_scores"):
            LOG.warning(
                "This detector was saved without a profile fingerprint, so it "
                "cannot be checked against the profile it is being loaded "
                "with. If the profile has been rebuilt since, refit the null."
            )
        det.alpha = d.get("alpha", 0.01)
        det.generator_agnostic = d.get("generator_agnostic", False)
        if d.get("null_scores"):
            det.null_scores = np.sort(np.asarray(d["null_scores"], dtype=float))
            det.null_info = d.get("null_info", {})
            det._fit_tail()
        if d.get("window_null"):
            det.window_null = np.sort(np.asarray(d["window_null"], dtype=float))
            det.window_params = d.get("window_params", {"window": 80, "step": 20})
        return det


# ===========================================================================
# 9. Simulation (threshold setting only - NOT validation)
# ===========================================================================


def simulate_designed(
    profile: "FamilyProfile",
    n: int = 100,
    fingerprint: GeneratorFingerprint = PROTEINMPNN_PRIOR,
    core_recovery: float = 0.80,
    surface_recovery: float = 0.25,
    composition_blend: float = 0.0,
    seed: int = 0,
) -> List[CleanSequence]:
    """Generate inverse-folding-like redesigns of real family members.

    Mimics the characteristic behaviour: high sequence recovery at buried /
    conserved positions, low recovery at variable surface positions, where the
    generator's own composition takes over.

    `composition_blend` mixes the family's own composition into the sampling
    distribution, from 0 (pure generator) to 1 (composition indistinguishable
    from the family). It is the stress test that matters: raising it blunts
    every composition-based signal, which is what high sampling temperature,
    explicit composition constraints, or filtering designs for native-likeness
    all do in practice. Signals that test the evolutionary *process* rather than
    residue statistics should survive it; run the sweep and see which of yours
    do.

    This exists so you can set a threshold without waiting on a GPU. It CANNOT
    validate the fingerprint, because it is generated from that same
    fingerprint. Circular by construction - use it for thresholds, then confirm
    on real generator output.
    """
    if profile.information_content is None or not profile._members:
        raise ModelNotBuiltError("Profile must be built with member sequences.")
    rng = np.random.default_rng(seed)
    ic = profile.information_content
    hi = float(np.quantile(ic, 0.67))
    gen = fingerprint.freqs / fingerprint.freqs.sum()
    if composition_blend > 0 and profile.family_composition is not None:
        fam = profile.family_composition / profile.family_composition.sum()
        gen = (1.0 - composition_blend) * gen + composition_blend * fam
        gen = gen / gen.sum()

    out: List[CleanSequence] = []
    for i in range(n):
        parent = profile._members[int(rng.integers(len(profile._members)))]
        hit = profile.score(parent)
        # target index -> True when that residue sits on a conserved model position
        conserved_at = {
            tgt: (0 <= model - 1 < len(ic) and ic[model - 1] >= hi)
            for model, tgt, _res in hit.aligned_positions
        }
        chars = list(parent.seq)
        for j in range(len(chars)):
            # Unaligned residues (loops, termini) behave like surface positions.
            keep_p = core_recovery if conserved_at.get(j, False) else surface_recovery
            if rng.random() > keep_p:
                chars[j] = AA[int(rng.choice(20, p=gen))]
        out.append(
            CleanSequence(seq="".join(chars), name=f"sim_{fingerprint.name}_{i:04d}")
        )
    LOG.info("Simulated %d %s-like redesigns.", len(out), fingerprint.name)
    return out


# ===========================================================================
# 10. Convenience pipeline
# ===========================================================================


@dataclass
class FamilyAcquisition:
    """What assembling a family actually produced, and where it was lost."""

    sequences: List[CleanSequence]
    source: str
    n_retrieved: int = 0
    n_unclean: int = 0
    n_wrong_length: int = 0
    n_before_dereplication: int = 0
    per_pass: List[Dict[str, object]] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"{len(self.sequences)} sequences from {self.source}"]
        for p in self.per_pass:
            lines.append(
                f"  reviewed={p['reviewed']}: retrieved {p['retrieved']}, "
                f"{p['unclean']} unusable, {p['wrong_length']} outside the "
                f"length window, {p['added']} new -> {p['total']} total"
            )
        if self.n_before_dereplication and self.sequences:
            lines.append(
                f"  dereplication: {self.n_before_dereplication} -> "
                f"{len(self.sequences)}"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def assemble_family(
    term: Optional[str] = None,
    natural_fasta: Optional[str] = None,
    min_length: int = 150,
    max_length: int = 400,
    target: int = 3000,
    min_usable: int = 120,
    identity_threshold: float = 0.90,
    cache_dir: Optional[str] = None,
    fetcher=None,
    seed: int = 0,
) -> FamilyAcquisition:
    """Assemble a natural family: reviewed entries first, then topped up.

    This lives in the module rather than in a notebook cell because the same
    twenty lines of acquisition logic have now produced three separate bugs,
    none of which any test could see:

      1. a doubly-compressed response decoded as text, which returned 14
         unusable records for a query matching thousands;
      2. stopping at the reviewed set, which left a 433-sequence family against
         a 500-position model - SwissProt holds only a few hundred reviewed
         kinases and a Potts fit needs thousands;
      3. a guard added for the local-file path that also matched the UniProt
         path, so the unreviewed top-up silently never ran and the family came
         back at 433 again, which forced mutual information instead of DCA and
         a null too small to place alpha=0.01.

    All three were reported from a Colab run rather than caught here. A
    function takes a `fetcher`, and a fetcher can be faked.

    `fetcher(query, max_sequences, cache_dir) -> [(name, sequence)]` defaults to
    `fetch_uniprot`. Pass `natural_fasta` to skip the network entirely.
    """
    acq = FamilyAcquisition(sequences=[], source="")

    # The local file is a complete alternative to fetching, so it returns
    # rather than setting a flag the fetch loop then has to re-interpret. The
    # flag version is what broke: `SOURCE` was set by BOTH paths, so a guard
    # meaning "a local file was loaded" also fired once the first UniProt pass
    # had set it, and the second pass never ran.
    if natural_fasta:
        kept = [s for s in load_sequences(natural_fasta, min_length=min_length)
                if min_length <= len(s) <= max_length]
        if len(kept) < min_usable:
            raise SequenceError(
                f"{natural_fasta} gave {len(kept)} usable sequences in "
                f"{min_length}-{max_length} aa; at least {min_usable} are "
                "needed to build a profile, fit references and hold out a null."
            )
        acq.n_retrieved = len(kept)
        acq.n_before_dereplication = len(kept)
        acq.sequences = dereplicate(kept, identity_threshold=identity_threshold,
                                    max_keep=target, seed=seed)
        acq.source = f"local file ({os.path.basename(natural_fasta)})"
        return acq

    if not term:
        raise SequenceError("assemble_family needs either `term` or "
                            "`natural_fasta`.")
    fetch = fetcher if fetcher is not None else fetch_uniprot

    pool: List[CleanSequence] = []
    seen = set()
    for reviewed in (True, False):
        if len(pool) >= target:
            break
        query = build_uniprot_query(term, min_length=min_length,
                                    max_length=max_length, reviewed=reviewed)
        acq.queries.append(query)
        try:
            records = fetch(query, max_sequences=target * 2,
                            cache_dir=cache_dir)
        except Exception as exc:                        # network, parse, cache
            acq.notes.append(
                f"the reviewed={reviewed} request failed "
                f"({type(exc).__name__}: {exc})"
            )
            break
        kept, unclean, wrong_length = [], 0, 0
        for name, seq in records:
            try:
                cs = clean_sequence(seq, name=name.split()[0],
                                    min_length=min_length)
            except SequenceError:
                unclean += 1
                continue
            if min_length <= len(cs) <= max_length:
                kept.append(cs)
            else:
                wrong_length += 1
        added = 0
        for cs in kept:
            if cs.name not in seen:
                seen.add(cs.name)
                pool.append(cs)
                added += 1
        acq.n_retrieved += len(records)
        acq.n_unclean += unclean
        acq.n_wrong_length += wrong_length
        acq.per_pass.append({
            "reviewed": reviewed, "retrieved": len(records),
            "unclean": unclean, "wrong_length": wrong_length,
            "added": added, "total": len(pool), "query": query,
        })
        LOG.info("reviewed=%s: %d retrieved, %d new, %d total.",
                 reviewed, len(records), added, len(pool))

    if len(pool) < min_usable:
        acq.n_before_dereplication = len(pool)
        if acq.n_retrieved and acq.n_unclean == acq.n_retrieved:
            acq.notes.append(
                f"none of the {acq.n_retrieved} entries parsed as protein, "
                "which points at the response rather than the query"
            )
        elif acq.n_retrieved < min_usable:
            acq.notes.append(
                f"UniProt matched only {acq.n_retrieved} entries; the query "
                f"is too narrow ({acq.queries[0] if acq.queries else 'n/a'})"
            )
        else:
            acq.notes.append(
                f"{acq.n_wrong_length} of {acq.n_retrieved} fell outside "
                f"{min_length}-{max_length} aa; widen the length window"
            )
        acq.source = ""
        return acq

    acq.n_before_dereplication = len(pool)
    acq.sequences = dereplicate(pool, identity_threshold=identity_threshold,
                                max_keep=target, seed=seed)
    acq.source = f"UniProt ({term})"
    if len(acq.sequences) < 1200:
        acq.notes.append(
            f"only {len(acq.sequences)} sequences after dereplication. A Potts "
            "fit needs the effective count above twice the model length, and "
            "effective counts run below the raw count, so a ~300-position "
            "domain wants roughly 1,200+. Below that the build falls back to "
            "mutual information and says so."
        )
    return acq


def build_family_profile(
    family: Optional[str] = None,
    natural_fasta: Optional[str] = None,
    min_length: int = 80,
    max_length: int = 600,
    max_sequences: int = 2000,
    identity_threshold: float = 0.90,
    cache_dir: str = "./mpnn_cache",
    reviewed: bool = True,
    taxonomy_id: Optional[int] = None,
    check_homogeneity: bool = True,
    iterations: int = 2,
    self_calibrate: bool = True,
    fingerprint: GeneratorFingerprint = PROTEINMPNN_PRIOR,
) -> "FamilyProfile":
    """Fetch (or load) a family, dereplicate it, and build the profile.

    Pass `natural_fasta` to run entirely offline. Pass a Pfam accession as
    `family` (e.g. "PF00069") for the most reliable retrieval.
    """
    if natural_fasta:
        LOG.info("Loading natural sequences from %s", natural_fasta)
        seqs = load_sequences(natural_fasta, min_length=min_length)
        name = os.path.splitext(os.path.basename(natural_fasta))[0]
    elif family:
        query = build_uniprot_query(
            family, min_length=min_length, max_length=max_length,
            reviewed=reviewed, taxonomy_id=taxonomy_id,
        )
        LOG.info("UniProt query: %s", query)
        records = fetch_uniprot(query, max_sequences=max_sequences, cache_dir=cache_dir)
        seqs = []
        for nm, sq in records:
            try:
                seqs.append(clean_sequence(sq, name=nm.split()[0], min_length=min_length))
            except SequenceError:
                continue
        name = family
    else:
        raise ValueError("Provide either `family` or `natural_fasta`.")

    seqs = [s for s in seqs if min_length <= len(s) <= max_length]
    if not seqs:
        raise SequenceError(
            f"No sequences left after filtering to {min_length}-{max_length} aa."
        )
    seqs = dereplicate(seqs, identity_threshold=identity_threshold)
    prof = FamilyProfile(name=name).build(
        seqs, iterations=iterations, check_homogeneity=check_homogeneity
    )
    if self_calibrate:
        prof.fit_feature_reference(extract_features, fingerprint)
    return prof


def screen_fasta(
    fasta_path: str,
    detector: Detector,
    as_dicts: bool = False,
) -> List:
    """Screen every sequence in a FASTA file."""
    seqs = load_sequences(fasta_path)
    results = detector.predict_many(seqs)
    return [r.to_dict() for r in results] if as_dicts else results


# ===========================================================================
# 11. CLI
# ===========================================================================


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="mpnn_detect",
        description="Heuristic screening for designed / inverse-folded proteins.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build and save a family profile.")
    b.add_argument("--family", help="Pfam accession (PF00069) or family/protein name.")
    b.add_argument("--natural-fasta", help="Local FASTA instead of UniProt (offline).")
    b.add_argument("--out", required=True, help="Output directory for the profile.")
    b.add_argument("--min-length", type=int, default=80)
    b.add_argument("--max-length", type=int, default=600)
    b.add_argument("--max-sequences", type=int, default=2000)
    b.add_argument("--no-homogeneity-check", action="store_true")

    s = sub.add_parser("screen", help="Screen sequences in a FASTA file.")
    s.add_argument("fasta")
    s.add_argument("--profile", help="Directory of a saved profile.")
    s.add_argument("--detector", help="Saved detector JSON.")
    s.add_argument("--fingerprint", default="proteinmpnn", choices=sorted(FINGERPRINTS))
    s.add_argument("--json", action="store_true", help="Emit JSON instead of a report.")
    s.add_argument("--csv", help="Also write a results table to this path.")

    c = sub.add_parser("calibrate", help="Fit weights on labelled FASTA files.")
    c.add_argument("--natural", required=True)
    c.add_argument("--designed", required=True)
    c.add_argument("--profile")
    c.add_argument("--out", required=True)
    c.add_argument("--fingerprint", default="proteinmpnn", choices=sorted(FINGERPRINTS))

    f = sub.add_parser("fingerprint", help="Measure a generator's composition.")
    f.add_argument("fasta")
    f.add_argument("--name", required=True)

    a = ap.parse_args(argv)
    LOG.setLevel(logging.DEBUG if a.verbose else logging.INFO)

    if a.cmd == "build":
        prof = build_family_profile(
            family=a.family, natural_fasta=a.natural_fasta,
            min_length=a.min_length, max_length=a.max_length,
            max_sequences=a.max_sequences,
            check_homogeneity=not a.no_homogeneity_check,
        )
        prof.save(a.out)
        return 0

    if a.cmd == "screen":
        prof = FamilyProfile.load(a.profile) if a.profile else None
        det = (
            Detector.load(a.detector, profile=prof)
            if a.detector
            else Detector(profile=prof, fingerprint=FINGERPRINTS[a.fingerprint])
        )
        results = screen_fasta(a.fasta, det)
        if a.csv:
            write_csv(results_to_rows(results), a.csv)
        if a.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            for r in results:
                print(r.report())
                print()
        return 0

    if a.cmd == "calibrate":
        prof = FamilyProfile.load(a.profile) if a.profile else None
        det = Detector(profile=prof, fingerprint=FINGERPRINTS[a.fingerprint])
        info = det.calibrate(
            load_sequences(a.natural), load_sequences(a.designed)
        )
        det.save(a.out)
        print(json.dumps(info, indent=2, default=str))
        return 0

    if a.cmd == "fingerprint":
        fp = fit_fingerprint(a.name, [s.seq for s in load_sequences(a.fasta)])
        register_fingerprint(fp)
        for aa, v in zip(AA, fp.freqs):
            nat = NATURAL_BACKGROUND[AA_INDEX[aa]]
            print(f"  {aa}  {v:.4f}  ({v / nat:5.2f}x natural)")
        return 0

    return 1


# ===========================================================================
# 12. Empirical null, batch screening and segmental scanning
# ===========================================================================


def _chi2_sf_even(x: float, df: int) -> float:
    """Survival function of chi-squared for even degrees of freedom.

    Exact and closed-form, so Fisher's method needs no scipy dependency.
    """
    if x <= 0:
        return 1.0
    k = df // 2
    half = x / 2.0
    term, total = 1.0, 1.0
    for i in range(1, k):
        term *= half / i
        total += term
    return float(min(1.0, math.exp(-half) * total))


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed exactly."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return float(min(1.0, total))


@dataclass
class BatchResult:
    """Population-level verdict over a set of sequences from one source."""

    n: int
    n_flagged: int
    expected_flagged: float
    binomial_p: float
    fisher_p: float
    median_p_designed: float
    mutual_identity: float
    verdict: str
    notes: List[str] = field(default_factory=list)
    results: List[DetectionResult] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"Batch of {self.n} sequences",
            "-" * 66,
            f"Flagged                : {self.n_flagged} "
            f"(expected {self.expected_flagged:.1f} by chance)",
            f"Binomial p             : {self.binomial_p:.3g}",
            f"Fisher combined p      : {self.fisher_p:.3g}",
            f"Median P(designed)     : {self.median_p_designed:.3f}",
            f"Mutual identity        : {self.mutual_identity:.2f}",
            f"Verdict                : {self.verdict}",
        ]
        for n in self.notes:
            lines.append(f"Note: {n}")
        return "\n".join(lines)


def _mutual_identity(profile: Optional["FamilyProfile"],
                     sequences: Sequence[CleanSequence],
                     max_pairs: int = 400, seed: int = 0) -> float:
    """Median pairwise identity within the batch, in profile coordinates."""
    if profile is None or profile.hmm is None or len(sequences) < 2:
        return float("nan")
    vecs = [profile._query_vector(s) for s in sequences[: min(len(sequences), 60)]]
    rng = random.Random(seed)
    idents = []
    for _ in range(min(max_pairs, len(vecs) * (len(vecs) - 1) // 2)):
        i, j = rng.sample(range(len(vecs)), 2)
        a, b = vecs[i], vecs[j]
        shared = (a >= 0) & (b >= 0)
        if shared.sum() >= 20:
            idents.append(float((a[shared] == b[shared]).mean()))
    return float(np.median(idents)) if idents else float("nan")


def screen_batch(
    sequences: Iterable,
    detector: "Detector",
    alpha: Optional[float] = None,
) -> BatchResult:
    """Screen a set of sequences that share a provenance, as a population.

    A single borderline sequence is weak evidence. Fifty sequences from the same
    pipeline, each independently borderline in the same direction, are not -
    the shared bias is the evidence, and it accumulates in a way no
    single-sequence score reflects. This is the realistic screening setting: a
    submission, a design round, a deposited set.

    Two combined statistics are reported because they fail differently. The
    binomial test on the flag count is robust but throws away the magnitude of
    each score. Fisher's method uses the magnitudes but assumes independence,
    which sequences from one design run badly violate - redesigns of a single
    backbone are near-replicates. Treat a large gap between them as a sign that
    the batch is internally correlated rather than as a stronger result, and
    read the mutual-identity figure alongside both.
    """
    results = detector.predict_many(sequences)
    if not results:
        raise SequenceError("No usable sequences in batch.")
    alpha = alpha if alpha is not None else detector.alpha

    have_null = detector.null_scores is not None
    pvals = [r.p_value if r.p_value is not None else (1.0 - r.p_designed) for r in results]
    n = len(results)
    n_flagged = sum(1 for p in pvals if p <= alpha)
    expected = alpha * n if have_null else float("nan")

    binom_p = _binomial_sf(n_flagged, n, alpha) if have_null else float("nan")
    clipped = [min(max(p, 1e-12), 1.0) for p in pvals]
    stat = -2.0 * sum(math.log(p) for p in clipped)
    fisher_p = _chi2_sf_even(stat, 2 * n)

    cs_list = [
        s if isinstance(s, CleanSequence) else clean_sequence(s)
        for s in sequences
    ] if not isinstance(sequences, list) or not sequences or not isinstance(
        sequences[0], CleanSequence
    ) else list(sequences)
    try:
        mutual = _mutual_identity(detector.profile, cs_list)
    except Exception:
        mutual = float("nan")

    notes: List[str] = []
    if not have_null:
        notes.append(
            "No null fitted, so p-values are improvised from the logistic score "
            "and the binomial test is unavailable. Run Detector.fit_null() on "
            "known-natural sequences for real false-positive control."
        )
    if not math.isnan(mutual) and mutual > 0.7 and n >= 5:
        notes.append(
            f"Batch members share {mutual:.0%} mutual identity. They are near-"
            "replicates, so Fisher's method overstates the combined evidence; "
            "rely on the binomial test and on the median score."
        )

    primary = binom_p if have_null else fisher_p
    if primary <= 0.001:
        verdict = "BATCH LIKELY DESIGNED"
    elif primary <= 0.05:
        verdict = "BATCH SUSPICIOUS"
    else:
        verdict = "NO BATCH-LEVEL SIGNAL"

    return BatchResult(
        n=n, n_flagged=n_flagged, expected_flagged=expected,
        binomial_p=binom_p, fisher_p=fisher_p,
        median_p_designed=float(np.median([r.p_designed for r in results])),
        mutual_identity=mutual, verdict=verdict, notes=notes, results=results,
    )


@dataclass
class ScanWindow:
    start: int
    end: int
    p_designed: float
    score: float
    p_value: Optional[float]


def scan_sequence(
    sequence,
    detector: "Detector",
    window: Optional[int] = None,
    step: Optional[int] = None,
    name: str = "query",
) -> List[ScanWindow]:
    """Slide the detector along a sequence to localise designed regions.

    A whole-sequence score averages a chimera into the middle of the range. A
    natural binding domain fused to a designed scaffold - or a design grafted
    into a natural framework - shows up as a local peak that the global score
    hides.

    Windows are scored in composition-only mode, since 80 residues cannot be
    aligned to a family profile reliably. That means window scores must be
    judged against a window-level null built from natural sequences, not
    against the global cutoffs: on a family that is natively charge-rich or
    low-complexity those cutoffs invert, and every natural window reads as
    designed. Call `Detector.fit_window_null()` first; without it the window
    p-values are None and only the relative shape of the profile is meaningful.
    """
    cs = sequence if isinstance(sequence, CleanSequence) else clean_sequence(sequence, name=name)
    window = window or detector.window_params.get("window", 80)
    step = step or detector.window_params.get("step", 20)
    if len(cs) < window:
        return []
    out: List[ScanWindow] = []
    for start, score in detector._window_scores(cs, window, step):
        out.append(ScanWindow(
            start=start, end=start + window,
            p_designed=_sigmoid(score), score=score,
            p_value=detector.window_p_value(score),
        ))
    return out


def signal_heterogeneity(windows: Sequence[ScanWindow]) -> float:
    """Spread of the design signal along the sequence.

    Uses window p-values against the natural window null where available, since
    raw window probabilities are not comparable across families.

    Read this together with the window profile, not instead of it. A high value
    means the signal is uneven, which a chimera produces - but so does a design
    whose signal is simply concentrated in one region, and the two are not
    distinguishable from the summary number alone. Measured here, chimeras and
    whole designs score about the same (1.2 vs 1.3) while natural sequences sit
    near 0.6, so this separates *uneven* from *uniform*, not designed from
    chimeric. What identifies a fusion is the shape: a monotone rise or fall
    across the profile with natural-level windows at one end.
    """
    if len(windows) < 3:
        return 0.0
    if all(w.p_value is not None for w in windows):
        vals = np.array([-math.log10(max(w.p_value, 1e-6)) for w in windows])
        return float(np.quantile(vals, 0.9) - np.quantile(vals, 0.1))
    ps = np.array([w.p_designed for w in windows])
    return float(np.quantile(ps, 0.9) - np.quantile(ps, 0.1))


#: Retained under the old name; `signal_heterogeneity` says what it measures.
chimera_score = signal_heterogeneity


# ===========================================================================
# 13. Demo / offline family simulation
# ===========================================================================

_DEMO_ROOT = (
    "MKLSKEDFLKAVEKGMTGKVLLEIGGVRLTPDGSPEELALAIKKYNTKVIALAHVDYSDTETDFAKVIEKLG"
    "VEITVTDFDGSKEELYELLKEHNVQVLTLIGGNNNSEEELQKLIEKVGADVIVVDTDGDEAEIAKAVEELGA"
    "KVTVLTPLDGDVDEVAKVLKELNVTVIVTGTSDDTDEIAELVKEYNVEVIVLGSPNKAQIEALAKRLGIEVV"
)


def simulate_natural_family(
    n: int = 300,
    root: Optional[str] = None,
    seed: int = 11,
    n_coupled_pairs: int = 35,
    substitution_rate: float = 0.055,
    indel_rate: float = 0.10,
) -> List[CleanSequence]:
    """Simulate a protein family by evolving along a tree.

    For demos, offline testing and sanity-checking the pipeline when no real
    family is to hand. It is NOT a substitute for real data: the detector should
    be built and calibrated on genuine homologues before any result is trusted.

    What it reproduces, and why each part matters for exercising the detector:

      * a Yule tree (repeated branching from random tips), so there is clade
        structure and the four-point condition has something to say;
      * substitutions drawn from a chemistry-structured exchangeability matrix,
        so I<->V is common and I<->D is rare;
      * gamma-distributed site rates, so most positions are slow and a few fast;
      * indels along branches;
      * coupled position pairs, salt-bridge style - either position varies
        freely alone, but the combination is constrained.

    A star phylogeny of uniform random mutants, which is the obvious thing to
    write, has none of these. Built on that, every evolutionary signal would
    measure noise and appear useless.
    """
    rng = np.random.default_rng(seed)
    root = root or _DEMO_ROOT
    L = len(root)

    classes = {**{a: 0 for a in "AVLIM"}, **{a: 1 for a in "FWY"},
               **{a: 2 for a in "STNQ"}, **{a: 3 for a in "KRH"},
               **{a: 4 for a in "DE"}, "G": 5, "P": 6, "C": 7}
    cls = np.array([classes[a] for a in AA])
    exch = np.outer(NATURAL_BACKGROUND, NATURAL_BACKGROUND)
    exch = exch * np.where(cls[:, None] == cls[None, :], 8.0, 1.0)
    np.fill_diagonal(exch, 0.0)
    exch = exch / exch.sum(axis=1, keepdims=True)

    rate = rng.gamma(0.6, 1 / 0.6, size=L)
    rate /= rate.mean()
    fast = [i for i in range(L) if rate[i] >= np.quantile(rate, 0.35)]
    rng.shuffle(fast)
    pairs = [(int(a), int(b)) for a, b in zip(fast[0::2], fast[1::2])
             if abs(a - b) >= 10][:n_coupled_pairs]

    def evolve(seq: str, branch: float) -> str:
        ch = list(seq)
        for i in range(min(len(ch), L)):
            if rng.random() < substitution_rate * branch * rate[i]:
                a = AA_INDEX.get(ch[i])
                if a is not None:
                    ch[i] = AA[int(rng.choice(20, p=exch[a]))]
        if rng.random() < indel_rate * branch:
            pos = int(rng.integers(0, max(1, len(ch) - 6)))
            if rng.random() < 0.5:
                del ch[pos : pos + int(rng.integers(1, 4))]
            else:
                for _ in range(int(rng.integers(1, 4))):
                    ch.insert(pos, AA[int(rng.choice(20, p=NATURAL_BACKGROUND))])
        for a, b in pairs:
            if a < len(ch) and b < len(ch) and rng.random() < 0.93:
                if rng.random() < 0.5:
                    ch[a], ch[b] = "KR"[rng.integers(2)], "DE"[rng.integers(2)]
                else:
                    ch[a], ch[b] = "DE"[rng.integers(2)], "KR"[rng.integers(2)]
        return "".join(ch)

    pop = [root]
    while len(pop) < n:
        i = int(rng.integers(len(pop)))
        pop.append(evolve(pop[i], float(rng.exponential(1.4))))
        pop[i] = evolve(pop[i], float(rng.exponential(0.5)))
    return [CleanSequence(seq=s, name=f"sim_nat_{i:04d}") for i, s in enumerate(pop)]


# ===========================================================================
# 14. Export helpers
# ===========================================================================


def results_to_rows(
    results: Sequence["DetectionResult"], top_signals: int = 3
) -> List[Dict[str, object]]:
    """Flatten DetectionResults into plain dicts for CSV or DataFrame export.

    Deliberately returns built-in types rather than a DataFrame, so the module
    keeps no pandas dependency and stays importable in a bare environment. The
    top contributing signals are inlined because a screening table is unusable
    if the reason for each verdict is buried in a nested structure.
    """
    rows: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "name": r.name,
            "length": r.length,
            "verdict": r.verdict,
            "p_designed": round(r.p_designed, 6),
            "p_value": None if r.p_value is None else float(f"{r.p_value:.4g}"),
            "q_value": None if r.q_value is None else float(f"{r.q_value:.4g}"),
            "fdr_significant": r.fdr_significant,
            "family": r.family,
            "p_value_extrapolated": r.p_value_extrapolated,
            "score": round(r.score, 4),
            "generator_guess": r.generator_guess,
            "calibrated": r.calibrated,
            "used_profile": r.used_profile,
            "fragility": r.fragility,
            "flags": ";".join(r.flags),
        }
        ordered = sorted(r.signals, key=lambda s: -abs(s.contribution))[:top_signals]
        for i, s in enumerate(ordered, 1):
            row[f"signal{i}"] = s.key
            row[f"signal{i}_contribution"] = round(s.contribution, 3)
        for key in ("nearest_identity", "profile_bits", "profile_coverage",
                    "coupling_n_pairs", "n_substitutions", "evidence"):
            if key in r.diagnostics:
                row[key] = round(float(r.diagnostics[key]), 4)
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> str:
    """Write dict rows to CSV using the standard library.

    The column set is the union across all rows, so rows that carry different
    diagnostics (profile mode versus composition-only, say) do not silently
    lose columns to whichever row happened to come first.
    """
    import csv

    rows = list(rows)
    if not rows:
        raise ValueError("Nothing to write: no rows.")
    columns: List[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})
    LOG.info("Wrote %d rows to %s", len(rows), path)
    return path



# ===========================================================================
# 15. Multiple testing and screening libraries
# ===========================================================================


def adjust_pvalues(pvalues: Sequence[float], method: str = "bh") -> List[float]:
    """Benjamini-Hochberg q-values.

    `alpha` controls the error rate for ONE test. A screen runs thousands, and
    at alpha=0.01 ten thousand natural sequences produce a hundred flags by
    construction - every one of them a false positive, and no amount of
    tightening alpha fixes the arithmetic without destroying sensitivity.

    The q-value answers the question a screen actually asks: of the sequences I
    flag, what fraction should I expect to be wrong? Controlling that at 5% is
    a usable operating point; controlling a per-sequence alpha at 1% across
    10,000 tests is not.
    """
    if method != "bh":
        raise ValueError(f"Unknown method {method!r}; only 'bh' is supported.")
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1].")
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, so a q-value is never
    # smaller than that of a less significant test.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return [float(v) for v in q]


def apply_fdr(
    results: Sequence["DetectionResult"], fdr: float = 0.05
) -> List["DetectionResult"]:
    """Attach q-values to a set of results and mark those passing the FDR.

    Modifies the results in place and returns the flagged subset. Sequences with
    no p-value (no null fitted) are left unmarked rather than being assigned a
    q-value that would not mean anything.
    """
    scored = [r for r in results if r.p_value is not None]
    if not scored:
        LOG.warning(
            "No p-values available, so no FDR control. Fit a null with "
            "Detector.fit_null() first."
        )
        return []
    qs = adjust_pvalues([r.p_value for r in scored])
    for r, q in zip(scored, qs):
        r.q_value = q
        r.fdr_significant = q <= fdr
    flagged = [r for r in scored if r.fdr_significant]
    LOG.info(
        "FDR %.3g: %d of %d flagged (per-sequence alpha would flag %d).",
        fdr, len(flagged), len(scored),
        sum(1 for r in scored if r.p_value is not None and r.p_value <= fdr),
    )
    return flagged


class ScreeningLibrary:
    """Several family models with routing, for screening sequences of unknown family.

    A single profile answers "is this a designed member of family X". A screen
    is handed a sequence and does not know X. This holds one calibrated detector
    per family, routes a query to whichever profile actually recognises it, and
    says so plainly when none does - because a sequence no profile recognises
    has not been cleared, it has not been assessed, and reporting those as
    negatives is how a screen quietly fails.
    """

    def __init__(self, fallback: Optional["Detector"] = None):
        self.detectors: Dict[str, "Detector"] = {}
        #: Used when nothing matches; composition-only, so much weaker.
        self.fallback = fallback

    def add(self, name: str, detector: "Detector") -> "ScreeningLibrary":
        if detector.profile is None or detector.profile.hmm is None:
            raise ModelNotBuiltError(f"Detector for {name!r} has no built profile.")
        if detector.null_scores is None:
            LOG.warning(
                "Detector for %r has no null fitted, so its p-values will be "
                "missing and it cannot take part in FDR control.", name,
            )
        self.detectors[name] = detector
        return self

    #: Below this, a routed match is real but weak and the profile-derived
    #: signals get noticeably noisier.
    WEAK_MATCH_BITS = 40.0

    def route(
        self, cs: CleanSequence, min_coverage: float = 0.35, min_bits: float = 10.0
    ) -> Optional[Tuple[str, "Detector", ProfileHit]]:
        """Pick the profile that best recognises this sequence.

        Ranked by bits per aligned model position rather than by raw bit score,
        which would simply favour the longest profile in the library.

        `min_bits` is deliberately low. A redesign keeps the fold - coverage
        stays near 1 - while the bit score falls with the amount rewritten, so a
        high floor rejects the most heavily redesigned sequences first. That is
        backwards for a screen: the sequences most worth assessing are the ones
        it would decline to assess. Measured here, the weakest true family
        member scored 17 bits while the strongest unrelated sequence scored
        about 0, so 10 sits in the gap and separates on coverage and homology
        rather than on how much of the sequence survived redesign.
        """
        best = None
        for name, det in self.detectors.items():
            hit = det.profile.score(cs)
            if not hit.matched or hit.coverage < min_coverage or hit.bits < min_bits:
                continue
            if best is None or hit.bits_per_position > best[2].bits_per_position:
                best = (name, det, hit)
        return best

    def predict(self, sequence, name: str = "query") -> "DetectionResult":
        cs = sequence if isinstance(sequence, CleanSequence) else clean_sequence(sequence, name=name)
        routed = self.route(cs)
        if routed is None:
            if self.fallback is not None:
                result = self.fallback.predict(cs)
                result.flags.append("no_family_match")
                result.notes.append(
                    "No profile in the library recognised this sequence, so it "
                    "was scored on composition alone - much weaker evidence. A "
                    "result here means 'not assessed against a family', not "
                    "'cleared'. Add the relevant family and re-screen."
                )
                return result
            raise ModelNotBuiltError(
                "No profile in the library recognised this sequence and no "
                "fallback detector was supplied. Add the relevant family, or "
                "pass a composition-only Detector as `fallback` - but note that "
                "an unrecognised sequence has not been assessed, and treating "
                "it as negative is how a screen fails silently."
            )
        family, detector, hit = routed
        result = detector.predict(cs)
        result.family = family
        result.diagnostics["routed_bits"] = hit.bits
        result.diagnostics["routed_bits_per_position"] = hit.bits_per_position
        if hit.bits < self.WEAK_MATCH_BITS:
            result.flags.append("weak_family_match")
            result.notes.append(
                f"Routed to {family!r} on a weak match ({hit.bits:.0f} bits, "
                f"{hit.coverage:.0%} coverage). The fold is recognised but the "
                "alignment is poor, so the profile-derived signals carry more "
                "noise than usual here. A closer family would give a firmer "
                "answer."
            )
        return result

    def predict_many(self, sequences: Iterable, fdr: Optional[float] = 0.05) -> List["DetectionResult"]:
        out: List["DetectionResult"] = []
        for s in sequences:
            try:
                out.append(self.predict(s))
            except (SequenceError, ModelNotBuiltError) as exc:
                LOG.warning("Skipping sequence: %s", exc)
        if fdr is not None and out:
            apply_fdr(out, fdr=fdr)
        return out

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for name, det in self.detectors.items():
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            sub = os.path.join(directory, safe)
            det.profile.save(os.path.join(sub, "profile_model"))
            det.save(os.path.join(sub, "detector.json"))
        with open(os.path.join(directory, "library.json"), "w") as fh:
            json.dump({"version": __version__,
                       "families": {re.sub(r"[^A-Za-z0-9._-]", "_", n): n
                                    for n in self.detectors}}, fh, indent=2)
        LOG.info("Library of %d families saved to %s", len(self.detectors), directory)

    @classmethod
    def load(cls, directory: str, fallback: Optional["Detector"] = None) -> "ScreeningLibrary":
        with open(os.path.join(directory, "library.json")) as fh:
            meta = json.load(fh)
        lib = cls(fallback=fallback)
        for safe, name in meta["families"].items():
            sub = os.path.join(directory, safe)
            prof = FamilyProfile.load(os.path.join(sub, "profile_model"))
            lib.detectors[name] = Detector.load(
                os.path.join(sub, "detector.json"), profile=prof)
        LOG.info("Loaded library of %d families.", len(lib.detectors))
        return lib


# ===========================================================================
# 16. Repeated-split validation
# ===========================================================================


def _auroc(neg: Sequence[float], pos: Sequence[float]) -> float:
    """Rank-based AUROC with ties at half weight, fully vectorised.

    The tie handling is done with group boundaries rather than a Python loop:
    the permutation test below calls this thousands of times per signal, and a
    per-element loop there costs minutes instead of seconds.
    """
    a, b = np.asarray(neg, dtype=float), np.asarray(pos, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    v = np.concatenate([a, b])
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    bounds = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1], True])
    starts, ends = bounds[:-1], bounds[1:]
    # mean 1-based rank within each tie group
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks_sorted = np.repeat(avg, ends - starts)
    ranks = np.empty(v.size)
    ranks[order] = ranks_sorted
    r_pos = ranks[a.size:].sum()
    return float((r_pos - b.size * (b.size + 1) / 2.0) / (a.size * b.size))


def _auroc_many(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """AUROC for many permutations at once: neg/pos are (K, n) and (K, m)."""
    K, n = neg.shape
    m = pos.shape[1]
    v = np.concatenate([neg, pos], axis=1)
    order = np.argsort(v, axis=1, kind="mergesort")
    sv = np.take_along_axis(v, order, axis=1)
    ranks = np.empty_like(v)
    base = np.arange(1, n + m + 1, dtype=float)
    np.put_along_axis(ranks, order, np.broadcast_to(base, v.shape).copy(), axis=1)
    # Ties are vanishingly rare for continuous scores; correct them only when
    # present so the common path stays a single vectorised sort.
    tied = (sv[:, 1:] == sv[:, :-1]).any(axis=1)
    for k in np.flatnonzero(tied):
        ranks[k] = np.argsort(np.argsort(v[k], kind="mergesort")) + 1.0
        s = sv[k]
        bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
        st, en = bounds[:-1], bounds[1:]
        avg = (st + en - 1) / 2.0 + 1.0
        rs = np.repeat(avg, en - st)
        tmp = np.empty(n + m)
        tmp[order[k]] = rs
        ranks[k] = tmp
    r_pos = ranks[:, n:].sum(axis=1)
    return (r_pos - m * (m + 1) / 2.0) / (n * m)


def _percentile_ci(values: Sequence[float], level: float = 0.90) -> Tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(v, (1 - level) / 2))
    hi = float(np.quantile(v, 1 - (1 - level) / 2))
    return lo, hi


def _permutation_p(neg: Sequence[float], pos: Sequence[float],
                   n_permutations: int = 2000, seed: int = 0) -> float:
    """One-sided label-permutation p-value for AUROC > 0.5.

    A signal can post a high AUROC on a small sample by luck. Permuting the
    labels gives the distribution of AUROC under no association, which is the
    right reference - and unlike a parametric test it makes no assumption about
    the score distribution, which for these signals is visibly non-normal.
    """
    a, b = np.asarray(neg, dtype=float), np.asarray(pos, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = _auroc(a, b)
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n_neg = a.size
    count = 0
    batch = 250
    done = 0
    while done < n_permutations:
        k = min(batch, n_permutations - done)
        perms = np.array([rng.permutation(pool) for _ in range(k)])
        aur = _auroc_many(perms[:, :n_neg], perms[:, n_neg:])
        count += int((aur >= observed).sum())
        done += k
    return float((count + 1) / (n_permutations + 1))


@dataclass
class SignalPerformance:
    key: str
    auroc_mean: float
    auroc_lo: float
    auroc_hi: float
    permutation_p: float
    net_pull_mean: float
    n_repeats: int

    @property
    def informative(self) -> bool:
        """True when the interval excludes chance in the designed direction.

        Both conditions are required, and they fail for different reasons: a
        wide interval means the signal is unstable across rebuilds even if the
        pooled permutation test is significant, and that instability is itself
        a reason not to lean on it.
        """
        return bool(self.auroc_lo > 0.5 and self.permutation_p <= 0.05)

    #: How far below 0.5 the mean must sit before a signal is called inverted
    #: rather than merely uninformative. Testing the interval edge alone
    #: conflates two different things: a signal that is CONSTANT on both
    #: classes lands at 0.498 with an interval 0.002 wide and looks "excluded
    #: from chance" (construct_motifs on real kinases, where no sequence
    #: carries a purification tag, so its AUROC is 0.5 by construction),
    #: while one that is genuinely backwards sits far below with a wide
    #: interval (stratum_contrast at 0.394). Requiring both a meaningful
    #: effect and an interval that excludes chance separates them.
    INVERSION_MARGIN = 0.02

    @property
    def verdict(self) -> str:
        if self.informative:
            return "informative"
        if self.auroc_hi < 0.5 and self.auroc_mean < 0.5 - self.INVERSION_MARGIN:
            return "inverted - points the wrong way"
        if self.permutation_p > 0.05:
            return "indistinguishable from chance"
        return "unstable - interval spans chance"


@dataclass
class ValidationReport:
    n_repeats: int
    alpha: float
    overall_auroc_mean: float
    overall_auroc_ci: Tuple[float, float]
    fpr_mean: float
    fpr_ci: Tuple[float, float]
    tpr_mean: float
    tpr_ci: Tuple[float, float]
    signals: List[SignalPerformance]
    seconds: float
    recommended_alpha: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    #: Were the designs real generator output, or simulated? A report carries
    #: this because it is the difference between a measurement and a bound, and
    #: downstream code (and anyone reading a saved CSV) cannot tell otherwise.
    designs_were_real: bool = False
    n_designs: int = 0

    def report(self, top: int = 25) -> str:
        lo, hi = self.overall_auroc_ci
        flo, fhi = self.fpr_ci
        tlo, thi = self.tpr_ci
        lines = [
            f"Repeated-split validation: {self.n_repeats} independent splits, "
            f"{self.seconds:.0f}s",
            f"Designs: {self.n_designs or 'n/a'} "
            + ("REAL generator output - these are measurements"
               if self.designs_were_real else
               "simulated - these are upper bounds, not measurements"),
            "-" * 78,
            f"Overall AUROC      : {self.overall_auroc_mean:.4f}  "
            f"[{lo:.4f}, {hi:.4f}]  (90% across splits)",
            f"False positives    : {self.fpr_mean:.3%}  [{flo:.3%}, {fhi:.3%}]  "
            f"at alpha={self.alpha}"
            + ("" if not (self.recommended_alpha and self.recommended_alpha > self.alpha)
               else f"   [alpha below what the null supports; {self.recommended_alpha:.3g} is reachable]"),
            f"Designs detected   : {self.tpr_mean:.3%}  [{tlo:.3%}, {thi:.3%}]",
            "",
            f"{'signal':<30}{'AUROC':>8}{'90% interval':>20}{'perm p':>10}{'  verdict'}",
            "-" * 78,
        ]
        for s in sorted(self.signals, key=lambda s: -s.auroc_mean)[:top]:
            verdict = s.verdict
            lines.append(
                f"{s.key:<30}{s.auroc_mean:>8.3f}"
                f"{f'[{s.auroc_lo:.3f}, {s.auroc_hi:.3f}]':>20}"
                f"{s.permutation_p:>10.4f}  {verdict}"
            )
        for n in self.notes:
            lines.append("")
            lines.append(f"Note: {n}")
        return "\n".join(lines)

    def to_rows(self) -> List[Dict[str, object]]:
        """Per-signal rows for CSV.

        `designs_real` is repeated on every row rather than left to a separate
        header, because a CSV gets opened months later by someone who did not
        run it, and the difference between a measurement and an upper bound is
        the single most important thing about these numbers. Redundant storage
        is cheap; a file that cannot say which it holds is not.
        """
        return [
            {"signal": s.key, "auroc_mean": round(s.auroc_mean, 5),
             "auroc_lo": round(s.auroc_lo, 5), "auroc_hi": round(s.auroc_hi, 5),
             "permutation_p": s.permutation_p,
             "net_pull_mean": round(s.net_pull_mean, 5),
             "informative": s.informative, "verdict": s.verdict,
             "n_repeats": s.n_repeats,
             "designs_real": self.designs_were_real,
             "n_designs": self.n_designs,
             "alpha": self.alpha}
            for s in sorted(self.signals, key=lambda s: -s.auroc_mean)
        ]


def cross_validate(
    sequences: Sequence,
    n_repeats: int = 10,
    alpha: float = 0.01,
    fingerprint: GeneratorFingerprint = PROTEINMPNN_PRIOR,
    generator_agnostic: bool = False,
    designed_sequences: Optional[Sequence] = None,
    n_designed: int = 60,
    composition_blend: float = 0.0,
    coupling_method: str = "dca",
    build_fraction: float = 0.55,
    symfrac: float = 0.6,
    reference_fraction: float = 0.23,
    n_permutations: int = 2000,
    seed: int = 0,
    progress: bool = True,
) -> ValidationReport:
    """Rebuild the whole model on repeated independent splits and report intervals.

    A single split gives a single number, and a single number from one split of
    one family is not evidence of anything - it has no error bar, so there is no
    way to tell a signal that works from one that got lucky. This refits
    everything (profile, couplings, evolutionary model, placement null,
    per-family references, score null) on `n_repeats` disjoint splits and
    reports each signal's AUROC with an interval across splits plus a
    label-permutation p-value.

    This is the expensive part of the method by a wide margin, and it is the
    part that makes the rest defensible. With DCA enabled each repeat refits a
    Potts model, so expect roughly half a minute to a few minutes per repeat
    depending on family size.

    Pass `designed_sequences` to validate against real generator output, which
    is the only way these numbers mean what they appear to mean. Without it the
    designs are simulated, which bounds the result from above - see the module
    docstring.
    """
    seqs = [s if isinstance(s, CleanSequence) else clean_sequence(s) for s in sequences]
    if len(seqs) < 120:
        raise SequenceError(
            f"Only {len(seqs)} sequences; repeated-split validation needs at "
            "least 120 so that each split still has enough to build a profile, "
            "fit references and hold out a null."
        )
    notes: List[str] = []
    real_designs: List[CleanSequence] = []
    design_folds: Optional[List[List[CleanSequence]]] = None
    if designed_sequences is None:
        notes.append(
            "Designs were simulated from the generator fingerprint the detector "
            "tests for, so separation is guaranteed in part by construction. "
            "These intervals bound real performance from above. Pass "
            "designed_sequences=<real output> to measure rather than bound."
        )
    else:
        real_designs = [s if isinstance(s, CleanSequence) else clean_sequence(s)
                        for s in designed_sequences]
        if not real_designs:
            raise SequenceError("designed_sequences was given but is empty.")
        # Scoring the same fixed design set on every split and then pooling the
        # scores pseudo-replicates the design side: the permutation test is fed
        # n_repeats copies of the same sequences, so its null is far too tight
        # and every p-value comes out small. Where there are enough designs,
        # give each split a disjoint fold instead, so pooling is legitimate.
        per_fold = len(real_designs) // n_repeats
        if per_fold >= 20:
            shuffled = list(real_designs)
            random.Random(seed).shuffle(shuffled)
            design_folds = [shuffled[i::n_repeats] for i in range(n_repeats)]
            notes.append(
                f"{len(real_designs)} real designs were split into "
                f"{n_repeats} disjoint folds of about {per_fold}, one per "
                "repeat, so the across-split interval and the permutation test "
                "treat each design once."
            )
        else:
            notes.append(
                f"Only {len(real_designs)} real designs for {n_repeats} "
                "repeats, so every repeat scores all of them. The across-split "
                "interval then reflects model-rebuild variability on a fixed "
                "design set, and the permutation p-values are optimistic "
                "because the pooled design scores repeat the same sequences."
            )
        srcs = {getattr(s, "name", "").split("_design")[0] for s in real_designs}
        if 1 < len(srcs) < 12:
            notes.append(
                f"The designs come from about {len(srcs)} backbones. Designs "
                "from one backbone share whatever that structure does to the "
                "signals, so the effective sample size for 'does this hold on "
                "a new protein' is the number of backbones, not the number of "
                "sequences. Read detection_by_group() alongside the pooled rate."
            )

    t0 = time.time()
    per_signal: Dict[str, List[float]] = {}
    per_pull: Dict[str, List[float]] = {}
    pooled_nat: Dict[str, List[float]] = {}
    pooled_des: Dict[str, List[float]] = {}
    overall, fprs, tprs = [], [], []
    rec_alpha = None

    for rep in range(n_repeats):
        rng = random.Random(seed * 1000 + rep)
        pool = list(seqs)
        rng.shuffle(pool)
        n = len(pool)

        # Allocate all four splits from the alpha the caller asked for, rather
        # than by fixed fractions. A null sized as a fraction of a small family
        # cannot place a small threshold, and a null sized greedily starves the
        # evaluation set - so both get a floor and the build set takes the
        # remainder.
        n_null = int(min(0.35 * n, max(60, 5.0 / alpha)))
        n_eval = int(min(0.20 * n, max(40, n_null * 0.6)))
        n_ref = int(reference_fraction * n)
        n_build = n - n_null - n_eval - n_ref
        if n_build < 80:
            # Too small to satisfy everything; give the build set its minimum
            # and shrink the null, which degrades the threshold rather than the
            # model.
            n_build = min(80, int(build_fraction * n))
            spare = n - n_build - n_ref
            n_null = max(20, int(spare * 0.6))
            n_eval = max(10, spare - n_null)
        build = pool[:n_build]
        ref = pool[n_build : n_build + n_ref]
        null_set = pool[n_build + n_ref : n_build + n_ref + n_null]
        eval_set = pool[n_build + n_ref + n_null :][:n_eval]
        if rep == 0 and progress:
            LOG.info("split sizes: build %d, reference %d, null %d, evaluation %d",
                     len(build), len(ref), len(null_set), len(eval_set))

        profile = FamilyProfile(name=f"cv{rep}", random_state=seed + rep)
        profile.build(build, check_homogeneity=False, coupling_method=coupling_method,
                      symfrac=symfrac)
        profile.fit_feature_reference(extract_features, fingerprint,
                                      reference_sequences=ref)

        if designed_sequences is None:
            designs = simulate_designed(
                profile, n=n_designed, fingerprint=fingerprint,
                composition_blend=composition_blend, seed=seed * 100 + rep,
            )
        elif design_folds is not None:
            designs = design_folds[rep]
        else:
            designs = real_designs

        det = Detector(profile=profile, fingerprint=fingerprint, alpha=alpha,
                       generator_agnostic=generator_agnostic)
        info = det.fit_null(null_set)
        rec_alpha = info.get("recommended_alpha", rec_alpha)

        nat_res = [det.predict(s) for s in eval_set]
        des_res = [det.predict(s) for s in designs]
        overall.append(_auroc([r.score for r in nat_res], [r.score for r in des_res]))
        fprs.append(sum(1 for r in nat_res if r.p_value is not None
                        and r.p_value <= alpha) / max(len(nat_res), 1))
        tprs.append(sum(1 for r in des_res if r.p_value is not None
                        and r.p_value <= alpha) / max(len(des_res), 1))

        keys = [s.key for s in nat_res[0].signals]
        for i, k in enumerate(keys):
            a = [r.signals[i].standardized for r in nat_res]
            b = [r.signals[i].standardized for r in des_res]
            per_signal.setdefault(k, []).append(_auroc(a, b))
            per_pull.setdefault(k, []).append(
                float(np.mean([r.signals[i].contribution for r in des_res])
                      - np.mean([r.signals[i].contribution for r in nat_res])))
            pooled_nat.setdefault(k, []).extend(a)
            pooled_des.setdefault(k, []).extend(b)

        if progress:
            LOG.info(
                "split %d/%d: AUROC %.4f, FPR %.1f%%, TPR %.1f%% (%.0fs elapsed)",
                rep + 1, n_repeats, overall[-1], 100 * fprs[-1], 100 * tprs[-1],
                time.time() - t0,
            )

    signals = []
    for k, aurocs in per_signal.items():
        lo, hi = _percentile_ci(aurocs)
        signals.append(SignalPerformance(
            key=k, auroc_mean=float(np.nanmean(aurocs)), auroc_lo=lo, auroc_hi=hi,
            permutation_p=_permutation_p(pooled_nat[k], pooled_des[k],
                                         n_permutations=n_permutations, seed=seed),
            net_pull_mean=float(np.nanmean(per_pull[k])), n_repeats=len(aurocs),
        ))

    if rec_alpha and rec_alpha > alpha:
        notes.append(
            f"alpha={alpha} is below what a null of this size can support; "
            f"{rec_alpha:.3g} is the smallest alpha with five null sequences "
            "above the threshold. The detection rate reported here is therefore "
            "throttled by tail extrapolation, not only by signal quality."
        )
    uninformative = [s.key for s in signals if not s.informative]
    if uninformative:
        notes.append(
            f"{len(uninformative)} signal(s) did not separate the two classes "
            f"on this family: {', '.join(sorted(uninformative))}. That is "
            "information, not failure - a signal whose interval spans 0.5 is "
            "carrying no evidence here and its weight is doing nothing."
        )

    return ValidationReport(
        n_repeats=n_repeats, alpha=alpha,
        overall_auroc_mean=float(np.nanmean(overall)),
        overall_auroc_ci=_percentile_ci(overall),
        fpr_mean=float(np.mean(fprs)), fpr_ci=_percentile_ci(fprs),
        tpr_mean=float(np.mean(tprs)), tpr_ci=_percentile_ci(tprs),
        signals=signals, seconds=time.time() - t0, notes=notes,
        recommended_alpha=rec_alpha,
        designs_were_real=designed_sequences is not None,
        n_designs=len(real_designs) if designed_sequences is not None
        else n_designed,
    )


# ===========================================================================
# 17. Reading real generator output
# ===========================================================================

#: ProteinMPNN writes one FASTA per input backbone. The FIRST record is the
#: native sequence (header carries `score=` and `designed_chains=`), and every
#: record after it is a sample (header carries `T=` and `sample=`). Treating
#: the native as a design is an easy and silent mistake: it is a natural
#: sequence, so it drags the measured detection rate down and looks like the
#: detector failing.
_PMPNN_NATIVE = re.compile(r"\bdesigned_chains=")
_PMPNN_SAMPLE = re.compile(r"\bT=\s*([0-9.]+).*?\bsample=\s*(\d+)", re.DOTALL)
_PMPNN_FIELD = re.compile(r"\b(score|global_score|seq_recovery)=\s*([0-9.eE+-]+)")


@dataclass
class GeneratedSequence:
    """One sequence from a real generator, with whatever the tool reported."""

    sequence: CleanSequence
    source: str                      #: backbone, i.e. the FASTA file's stem
    run: str = ""                    #: which run produced it, i.e. the folder
    temperature: Optional[float] = None
    sample: Optional[int] = None
    score: Optional[float] = None
    global_score: Optional[float] = None
    seq_recovery: Optional[float] = None
    is_native: bool = False


def read_proteinmpnn_fasta(
    path: str, include_native: bool = False, min_length: int = 30
) -> List[GeneratedSequence]:
    """Parse one ProteinMPNN output FASTA.

    Handles the two header shapes the tool emits - the native record and the
    per-sample records - and keeps the reported temperature, score and sequence
    recovery, which are what you need to stratify a validation by sampling
    temperature later.

    Multi-chain designs are written with chains joined by '/'. Those are split
    and only the longest chain is kept, because the detector scores one chain
    against one family profile; a concatenated multi-chain sequence would align
    badly and look designed for the wrong reason.
    """
    out: List[GeneratedSequence] = []
    stem = os.path.splitext(os.path.basename(path))[0]
    # Two runs of different model variants over the same backbones produce the
    # same file names (vanilla/seqs/tyk2.fa and soluble/seqs/tyk2.fa). Naming
    # sequences from the stem alone makes those collide, and a collision here
    # does not raise - it silently merges two generators into one set, or
    # overwrites entries in any dict keyed by name. So carry the run folder.
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.basename(parent) in ("seqs", "backbones"):
        parent = os.path.dirname(parent)
    run = os.path.basename(parent)
    for header, raw in parse_fasta(path):
        is_native = bool(_PMPNN_NATIVE.search(header)) and "T=" not in header
        if is_native and not include_native:
            continue
        chains = [c for c in raw.split("/") if c.strip()]
        if not chains:
            continue
        body = max(chains, key=len)
        try:
            cs = clean_sequence(
                body,
                name=(f"{run}/" if run else "")
                     + f"{stem}_{'native' if is_native else 'design'}"
                     + f"_{len(out):04d}",
                min_length=min_length,
            )
        except SequenceError as exc:
            LOG.debug("skipping %s: %s", header[:60], exc)
            continue
        fields = {k: float(v) for k, v in _PMPNN_FIELD.findall(header)}
        m = _PMPNN_SAMPLE.search(header)
        out.append(GeneratedSequence(
            sequence=cs, source=stem, run=run, is_native=is_native,
            temperature=float(m.group(1)) if m else None,
            sample=int(m.group(2)) if m else None,
            score=fields.get("score"), global_score=fields.get("global_score"),
            seq_recovery=fields.get("seq_recovery"),
        ))
    return out


def load_generator_output(
    paths: Union[str, Sequence[str]],
    include_native: bool = False,
    temperatures: Optional[Sequence[float]] = None,
    min_length: int = 30,
) -> List[GeneratedSequence]:
    """Read real generator output from one or more files or a directory.

    Accepts ProteinMPNN-style FASTA (headers parsed for temperature and
    recovery) and plain FASTA from any other tool, in which case the metadata
    is simply absent. `temperatures` filters to the listed sampling
    temperatures, which is how to separate "raw output" from the high-
    temperature regime that blunts the composition signals.
    """
    exts = (".fa", ".fasta", ".fas", ".fa.gz", ".fasta.gz")

    def _expand(p: str) -> List[str]:
        """One path -> the FASTA files it names. Directories are descended."""
        if os.path.isdir(p):
            # ProteinMPNN writes into <out_folder>/seqs/, not <out_folder>, and
            # pointing at the out_folder is the usual first mistake. Descend
            # rather than reporting an empty directory.
            root = p
            if not any(f.endswith(exts) for f in os.listdir(root)):
                for sub in ("seqs", "backbones"):
                    cand = os.path.join(root, sub)
                    if os.path.isdir(cand) and any(
                            f.endswith(exts) for f in os.listdir(cand)):
                        LOG.info("Reading from %s/ inside %s.", sub, root)
                        root = cand
                        break
            found = sorted(os.path.join(root, f) for f in os.listdir(root)
                           if f.endswith(exts))
            if not found:
                raise SequenceError(
                    f"No FASTA files in {p!r}. ProteinMPNN writes its designs "
                    "to <out_folder>/seqs/*.fa - check that the run finished "
                    "and that this is the right folder."
                )
            return found
        if os.path.exists(p):
            return [p]
        raise SequenceError(f"No such file or directory: {p!r}.")

    # A list is the natural way to pass several runs at once (one output folder
    # per model variant), so every element gets the same treatment a lone path
    # would: a directory is expanded, a file is taken as-is.
    items = [paths] if isinstance(paths, str) else list(paths)
    files: List[str] = []
    for item in items:
        for f in _expand(item):
            if f not in files:
                files.append(f)
    if not files:
        raise SequenceError(f"No FASTA files found at {paths!r}.")

    out: List[GeneratedSequence] = []
    for f in files:
        out.extend(read_proteinmpnn_fasta(f, include_native=include_native,
                                          min_length=min_length))
    if temperatures is not None:
        wanted = {round(float(t), 4) for t in temperatures}
        kept = [g for g in out
                if g.temperature is not None and round(g.temperature, 4) in wanted]
        if not kept:
            seen = sorted({g.temperature for g in out if g.temperature is not None})
            raise SequenceError(
                f"No sequences at temperature(s) {sorted(wanted)}; the files "
                f"contain {seen or 'no temperature annotations'}."
            )
        out = kept
    if not out:
        raise SequenceError(
            f"Parsed {len(files)} file(s) but found no usable design sequences. "
            "If this is ProteinMPNN output, check that it contains sample "
            "records (headers with 'T=' and 'sample='), not only the native."
        )
    temps = sorted({g.temperature for g in out if g.temperature is not None})
    LOG.info(
        "Loaded %d real design sequences from %d file(s)%s.",
        len(out), len(files),
        f" at temperature(s) {temps}" if temps else "",
    )
    return out


def generator_sequences(gens: Sequence[GeneratedSequence]) -> List[CleanSequence]:
    """Just the sequences, for passing to cross_validate or screen_batch."""
    return [g.sequence for g in gens]


def extract_domains(
    profile: "FamilyProfile",
    sequences: Sequence,
    min_coverage: float = 0.6,
    pad: int = 0,
) -> List[CleanSequence]:
    """Cut each sequence down to the region that aligns to the profile.

    This exists to make a real-design comparison fair, and without it the
    comparison is not. Inverse-folding tools are run on a backbone, and a
    backbone is almost always one domain carved out of a larger protein:
    ProteinMPNN on a TYK2 catalytic domain produces a 288-residue sequence,
    while TYK2 itself is 1,187 residues. Build the natural class by filtering
    whole UniProt entries to 150-400 aa and you have not collected kinase
    domains - you have collected kinases that happen to be short, which is a
    biased and unusual subset.

    Scoring those two sets against each other measures domain context as much
    as design: the designs are bare domains, the naturals carry termini,
    linkers and regulatory segments, and signals that look at coverage,
    composition or placement can separate the classes on that alone. The
    resulting detection rate is real but it is not evidence about design.

    Cutting both classes to the aligned span removes the asymmetry. Sequences
    that do not match the profile, or match over less than `min_coverage` of
    it, are dropped and counted - a partial domain is not a domain.
    """
    out: List[CleanSequence] = []
    n_miss = n_short = 0
    for s in sequences:
        cs = s if isinstance(s, CleanSequence) else clean_sequence(s)
        hit = profile.score(cs)
        if not hit.matched or not hit.aligned_positions:
            n_miss += 1
            continue
        if hit.coverage < min_coverage:
            n_short += 1
            continue
        lo = max(0, hit.aligned_positions[0][1] - pad)
        hi = min(len(cs.seq), hit.aligned_positions[-1][1] + 1 + pad)
        try:
            out.append(clean_sequence(cs.seq[lo:hi], name=cs.name,
                                      min_length=30))
        except SequenceError:
            n_short += 1
    LOG.info(
        "Domain extraction: %d of %d kept (%d no profile match, %d below "
        "%.0f%% coverage).", len(out), len(sequences), n_miss, n_short,
        100 * min_coverage,
    )
    if not out:
        raise SequenceError(
            f"No sequence aligned to the profile over {min_coverage:.0%} of "
            "its length. Check that the profile and the sequences are the same "
            "family."
        )
    return out


@dataclass
class BackboneMembership:
    """Whether one design backbone's native sequence belongs to the family."""

    source: str
    identity: float          #: nearest-neighbour identity to a family member
    coverage: float          #: fraction of the profile it aligns to
    bits_per_position: float
    in_family: bool
    reason: str = ""


def backbone_membership(
    profile: "FamilyProfile",
    natives: Sequence[GeneratedSequence],
    sample: int = 200,
    seed: int = 0,
) -> Tuple[List[BackboneMembership], Dict[str, float]]:
    """Check each design backbone's NATIVE sequence against the family.

    Run this before measuring anything, because a backbone that does not belong
    to the profiled family produces a detection rate that is partly just
    distance. Measured on the six kinase backbones here against a PF00069
    family assembled by filtering whole UniProt entries to 150-400 aa:

        cdk2  98.5%   cdk8  99.6%   p38  99.6%    <- Ser/Thr, in family
        cmet  34.3%   syk   34.3%   tyk2 38.8%    <- tyrosine kinases, not

    That family is dominated by small single-domain Ser/Thr kinases, so the
    three tyrosine kinases are genuine outliers in it. Their designs were
    detected at 100%, 100%, 100% against 91%, 98%, 100% for the in-family
    three, and two of their three natives were themselves called designed. The
    all-six headline of 98.2% is 96.5% on the backbones that actually belong.

    Note what this is NOT. Coverage was 63-90% in both groups and the
    composition was the same, so neither the "wrong family" check (does it
    align at all) nor the "domain context" check (is it a bare fragment) sees
    this. Only distance within the family does.

    Returns the per-backbone findings and the family's own leave-one-out
    identity percentiles, which is what "outlier" is measured against rather
    than a fixed cutoff - a tight family and a diverse one have very different
    notions of distant.
    """
    if profile.hmm is None:
        raise ModelNotBuiltError("Profile not built. Call build() first.")
    members = list(getattr(profile, "_members", None) or [])
    base: List[float] = []
    if members:
        rng = random.Random(seed)
        picked = (members if len(members) <= sample
                  else rng.sample(members, sample))
        for m in picked:
            cs = m if isinstance(m, CleanSequence) else clean_sequence(m)
            try:
                base.append(profile.nearest_neighbour_identity(
                    cs, exclude_self=True))
            except Exception:
                continue
    stats = {
        "n_reference": float(len(base)),
        "p10": float(np.percentile(base, 10)) if base else float("nan"),
        "p50": float(np.percentile(base, 50)) if base else float("nan"),
        "mean": float(np.mean(base)) if base else float("nan"),
    }
    # A fixed identity cutoff is wrong for both a tight family and a diverse
    # one, so "distant" means below where the family's own members sit.
    cut = stats["p10"] if base else 0.40

    seen = set()
    out: List[BackboneMembership] = []
    for g in natives:
        if g.source in seen:
            continue
        seen.add(g.source)
        hit = profile.score(g.sequence)
        ident = profile.nearest_neighbour_identity(g.sequence)
        reason = ""
        if not hit.matched or hit.coverage < 0.5:
            ok, reason = False, "does not align to the profile"
        elif ident < cut:
            ok, reason = False, (
                f"nearest family member is only {ident:.0%} identical, below "
                f"the family's own 10th percentile of {cut:.0%}"
            )
        else:
            ok = True
        out.append(BackboneMembership(
            source=g.source, identity=ident, coverage=hit.coverage,
            bits_per_position=hit.bits_per_position, in_family=ok,
            reason=reason))
    return out, stats


def describe_backbone_membership(
    rows: Sequence[BackboneMembership], stats: Dict[str, float]
) -> str:
    """Readable membership table, naming the backbones to exclude."""
    lines = [f"{'backbone':<10}{'nearest member':>16}{'coverage':>10}"
             f"{'bits/pos':>10}  in family",
             "-" * 60]
    for r in sorted(rows, key=lambda r: -r.identity):
        lines.append(f"{r.source:<10}{r.identity:>16.1%}{r.coverage:>10.1%}"
                     f"{r.bits_per_position:>10.2f}  "
                     f"{'yes' if r.in_family else 'NO'}")
    if stats.get("n_reference"):
        lines.append("")
        lines.append(
            f"The family's own members sit at {stats['mean']:.0%} mean nearest-"
            f"neighbour identity ({int(stats['n_reference'])} sampled), with a "
            f"10th percentile of {stats['p10']:.0%}, which is the bar used here."
        )
    bad = [r for r in rows if not r.in_family]
    if bad:
        lines.append("")
        lines.append(f"{len(bad)} of {len(rows)} backbones do not belong to "
                     f"this family:")
        for r in bad:
            lines.append(f"  {r.source}: {r.reason}")
        lines.append("")
        lines.append(
            "Their designs will be detected partly because the backbone is "
            "distant from the family, not because it was designed. Measure on "
            "the in-family backbones, or profile the family these belong to; "
            "report the restricted number as the result."
        )
    return "\n".join(lines)


def detection_by_group(
    detector: "Detector",
    gens: Sequence[GeneratedSequence],
    by: str = "source",
    alpha: Optional[float] = None,
) -> str:
    """Detection rate broken out by backbone, run or sampling temperature.

    The pooled rate over a few hundred designs looks like a few hundred trials
    and is not. ProteinMPNN conditions on structure, so every design from one
    backbone inherits whatever that backbone does to the signals: a buried core
    that forces hydrophobics, a long disordered insertion, an unusual fold. Six
    backbones that each give 90 designs provide six pieces of evidence about
    generalisation, not 540, and the pooled number cannot show that one of the
    six was missed entirely.

    `by` is "source" (backbone), "run" (model variant) or "temperature".
    """
    if not gens:
        return "no sequences"
    alpha = detector.alpha if alpha is None else alpha
    key = {"source": lambda g: g.source,
           "run": lambda g: g.run or "?",
           "temperature": lambda g: ("?" if g.temperature is None
                                     else f"T={g.temperature:g}")}[by]
    groups: Dict[str, List[GeneratedSequence]] = {}
    for g in gens:
        groups.setdefault(key(g), []).append(g)

    rows = []
    for name in sorted(groups):
        grp = groups[name]
        res = [detector.predict(g.sequence) for g in grp]
        ps = np.array([r.p_value for r in res], dtype=float)
        rec = [g.seq_recovery for g in grp if g.seq_recovery is not None]
        rows.append((name, len(grp), float(np.mean(ps <= alpha)),
                     float(np.median(ps)),
                     float(np.mean(rec)) if rec else float("nan")))

    lines = [f"{by:<16}{'n':>5}{'detected':>10}{'median p':>12}{'recovery':>10}",
             "-" * 53]
    for name, n, det_rate, med, rec in rows:
        lines.append(f"{name:<16}{n:>5}{det_rate:>10.1%}{med:>12.2g}"
                     + (f"{rec:>10.1%}" if rec == rec else f"{'-':>10}"))
    rates = [r[2] for r in rows]
    lines.append("-" * 53)
    lines.append(f"{'pooled':<16}{sum(r[1] for r in rows):>5}"
                 f"{np.mean([r[2] * r[1] for r in rows]) / np.mean([r[1] for r in rows]):>10.1%}")
    if len(rates) > 1:
        lines.append(f"\nworst {by}: {min(rates):.1%}; best: {max(rates):.1%}. "
                     f"The spread across {len(rates)} {by}(s) is the honest "
                     f"error bar on a new one, not the pooled interval.")
    return "\n".join(lines)


def describe_generator_output(gens: Sequence[GeneratedSequence]) -> str:
    """One-screen summary of what a real generator actually produced."""
    if not gens:
        return "no sequences"
    runs = {g.run for g in gens if g.run}
    lines = [f"{len(gens)} sequences from {len({g.source for g in gens})} "
             f"backbone(s)"
             + (f" x {len(runs)} run(s) ({', '.join(sorted(runs))})"
                if len(runs) > 1 else "")]
    by_t: Dict[Optional[float], List[GeneratedSequence]] = {}
    for g in gens:
        by_t.setdefault(g.temperature, []).append(g)
    for temp in sorted(by_t, key=lambda x: (x is None, x)):
        group = by_t[temp]
        rec = [g.seq_recovery for g in group if g.seq_recovery is not None]
        lengths = [len(g.sequence) for g in group]
        label = "T=?" if temp is None else f"T={temp:g}"
        line = (f"  {label:<8} n={len(group):<5} length "
                f"{min(lengths)}-{max(lengths)}")
        if rec:
            line += f"   sequence recovery {np.mean(rec):.1%} ({min(rec):.0%}-{max(rec):.0%})"
        lines.append(line)
    comp = composition("".join(g.sequence.seq for g in gens))
    ratio = comp / NATURAL_BACKGROUND
    up = sorted(zip(AA, ratio), key=lambda x: -x[1])[:4]
    dn = sorted(zip(AA, ratio), key=lambda x: x[1])[:4]
    lines.append(f"  composition vs UniProt background:")
    lines.append(f"    up   {', '.join(f'{a} {v:.1f}x' for a, v in up)}")
    lines.append(f"    down {', '.join(f'{a} {v:.2f}x' for a, v in dn)}")
    return "\n".join(lines)

# The entry point lives at the very end of the module: the CLI reaches helpers
# defined in every section above it, so running it from an earlier point raises
# NameError on whatever has not been defined yet.
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
