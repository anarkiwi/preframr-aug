"""Write-domain core: dump <-> OrderedWrites, per-voice partition, frame-ordered rebuild, and a
verified re-encode. Everything else manipulates the ``(frame, reg, val)`` triples these expose and
re-assembles through :func:`rebuild`, so augmentation stays agnostic to the atom encoding.
"""

from __future__ import annotations

import pandas as pd

from preframr_tokens.events import oracle, stream
from preframr_tokens.events.generate import writes_to_dump_df

VOICE_REGS = {0: range(0, 7), 1: range(7, 14), 2: range(14, 21)}
FILTER = "filter"
FILTER_REGS = range(21, 25)
VOICE_KEYS = (0, 1, 2, FILTER)


def voice_key(reg: int):
    """The bucket a register belongs to: voice 0/1/2 (regs 0-6/7-13/14-20) or ``FILTER`` (21-24)."""
    for voice, regs in VOICE_REGS.items():
        if reg in regs:
            return voice
    return FILTER


def load_ow(path) -> oracle.OrderedWrites:
    """Read a ``*dump.parquet`` into an OrderedWrites oracle; reject non-single-speed material up front
    (the event codec's scope), so every downstream transform sees a single-speed tune.
    """
    df = pd.read_parquet(path, columns=["clock", "irq", "chipno", "reg", "val"])
    ow = oracle.ordered_writes(df)
    if not stream.single_speed(ow):
        raise ValueError(f"{path}: not single-speed (out of event-codec scope)")
    return ow


def split_voices(ow: oracle.OrderedWrites) -> dict:
    """Partition the ordered triples into per-voice buckets keyed by :data:`VOICE_KEYS`; each bucket is
    frame-ordered ``(frame, reg, val)``. The filter/global lane (regs 21-24) is its own bucket.
    """
    parts: dict = {key: [] for key in VOICE_KEYS}
    for frame, reg, val in ow.triples():
        parts[voice_key(int(reg))].append((int(frame), int(reg), int(val)))
    return parts


def rebuild(parts: dict) -> oracle.OrderedWrites:
    """Reassemble per-voice buckets into a single frame-ordered OrderedWrites. Within a frame, writes
    are grouped in voice order (0,1,2,filter); canonical_writes re-derives the per-voice canonical
    order, so this grouping reproduces the source stream exactly for an untouched partition.
    """
    by_frame: dict = {}
    for key in VOICE_KEYS:
        for frame, reg, val in parts.get(key, []):
            by_frame.setdefault(int(frame), []).append((int(reg), int(val)))
    flat = [
        (frame, reg, val) for frame in sorted(by_frame) for reg, val in by_frame[frame]
    ]
    return ow_from_triples(flat)


def ow_from_triples(triples) -> oracle.OrderedWrites:
    """Frame-ordered ``(frame, reg, val)`` triples -> OrderedWrites via the render-ready dump df. The
    single re-assembly point; ``writes_to_dump_df`` sets clock=running index, irq=frame.
    """
    df = writes_to_dump_df([(int(f), int(r), int(v)) for f, r, v in triples])
    return oracle.ordered_writes(df)


def reencode(ow: oracle.OrderedWrites) -> list[int]:
    """Verified encode: ``decode(encode(ow)) == canonical_writes(ow)`` self-checks. Raises on a
    round-trip failure (a bug in the transform, not a data condition)."""
    return stream.encode(ow, verify=True)


def emit_dump(ow: oracle.OrderedWrites, path) -> None:
    """Write an augmented ``*dump.parquet`` from an OrderedWrites (real-irq clock pacing preserved via
    the oracle's frame indices)."""
    triples = [(int(f), int(r), int(v)) for f, r, v in ow.triples()]
    writes_to_dump_df(triples).to_parquet(path)


def atoms(ow: oracle.OrderedWrites) -> list[int]:
    """The verified atom stream for an OrderedWrites (alias of :func:`reencode`, for readability)."""
    return reencode(ow)
