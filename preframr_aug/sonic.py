"""Sonic plausibility filter: render an augmented tune to a mel fingerprint and reject it if it falls
outside the natural corpus fingerprint band. The augmentation gate -- a structurally valid transplant
can still be sonic nonsense; this is what keeps the augmented set inside the real corpus distribution.
Reuses ``preframr_audio.fingerprint`` (the generation-quality machinery) as the filter.
"""

from __future__ import annotations

import numpy as np

from preframr_audio.fingerprint import fingerprint_writes

PAL_IRQ = 19656


def fingerprint_ow(ow, frames: int = 600, feature: str = "mel") -> np.ndarray:
    """Mel fingerprint of (a bounded prefix of) an OrderedWrites, rendered through the canonical
    scaffold. Slicing to ``frames`` keeps the render cheap and comparable across tunes.
    """
    triples = [(int(f), int(r), int(v)) for f, r, v in ow.triples() if f < frames]
    return fingerprint_writes(triples, irq=PAL_IRQ, feature=feature)


def band(fingerprints, k: float = 3.0):
    """A per-dimension acceptance band (mean +/- k*std) over reference fingerprints; the natural-corpus
    envelope an augmented tune must sit inside."""
    arr = np.asarray(fingerprints, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean - k * std, mean + k * std


def in_band(fingerprint, bounds, tol: float = 0.02) -> bool:
    """True if at most ``tol`` of the fingerprint's dimensions fall outside the band -- a few outlier
    mel bins are normal; a wholesale departure is the reject signal."""
    lo, hi = bounds
    fp = np.asarray(fingerprint, dtype=np.float64)
    outside = np.mean((fp < lo) | (fp > hi))
    return bool(outside <= tol)
