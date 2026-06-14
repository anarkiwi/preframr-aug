"""The load-bearing correctness gate: partition + rebuild is identity. Nothing downstream is
trustworthy until ``rebuild(split_voices(load_ow(p)))`` re-encodes to the same atoms as the source on
real corpus tunes."""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import oracle, stream
from preframr_aug import writes

_CORPUS = os.environ.get("PREFRAMR_AUG_CORPUS", "/scratch/preframr/hvsc/MUSICIANS")


def _sample(n=12):
    files = glob.glob(os.path.join(_CORPUS, "**", "*.dump.parquet"), recursive=True)
    if not files:
        return []
    import random

    random.Random(0).shuffle(files)
    return files[:n]


def _synth_ow():
    rows = []
    for f in range(8):
        for v in range(3):
            base = 7 * v
            rows += [
                (f, base + 0, (0x20 + f) & 0xFF),
                (f, base + 1, 0x10),
                (f, base + 4, 0x41 if (f + v) % 3 else 0x40),
                (f, base + 5, 0x08),
                (f, base + 6, 0xA9),
            ]
        rows.append((f, 24, 0x0F))
    rows.sort(key=lambda t: t[0])
    df = pd.DataFrame(
        {
            "clock": np.arange(len(rows)),
            "irq": [r[0] for r in rows],
            "chipno": 0,
            "reg": [r[1] for r in rows],
            "val": [r[2] for r in rows],
        }
    )
    return oracle.ordered_writes(df)


def test_partition_rebuild_identity_synthetic():
    ow = _synth_ow()
    rebuilt = writes.rebuild(writes.split_voices(ow))
    assert writes.reencode(rebuilt) == stream.encode(ow)
    assert stream.decode(writes.reencode(rebuilt)) == stream.canonical_writes(ow)


def test_voice_key_map():
    assert [writes.voice_key(r) for r in (0, 6, 7, 13, 14, 20)] == [0, 0, 1, 1, 2, 2]
    assert (
        writes.voice_key(21) == writes.FILTER and writes.voice_key(24) == writes.FILTER
    )


@pytest.mark.parametrize("path", _sample() or [None])
def test_partition_rebuild_identity_corpus(path):
    if path is None:
        pytest.skip(f"no corpus dumps under {_CORPUS}")
    try:
        ow = writes.load_ow(path)
    except ValueError:
        pytest.skip("out of event-codec scope (not single-speed)")
    rebuilt = writes.rebuild(writes.split_voices(ow))
    assert writes.reencode(rebuilt) == stream.encode(ow), f"identity broke: {path}"
