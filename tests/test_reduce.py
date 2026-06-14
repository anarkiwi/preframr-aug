"""M1 reduction: the prefix becomes target-only, the continuation's content is preserved, and the
result is a valid encodable dump."""

import numpy as np
import pandas as pd

from preframr_tokens.events import oracle
from preframr_aug import reduce, writes


def _three_voice_ow(n_frames=8):
    """Voice 0 (lead) plays every frame; voices 1 and 2 (accompaniment) too -- so the prefix strip
    empties no frame and indices stay stable for a clean assertion."""
    rows = []
    for f in range(n_frames):
        for v in range(3):
            base = 7 * v
            fr = 0x0800 * (3 - v) + 0x40 * f
            rows += [
                (f, base + 0, fr & 0xFF),
                (f, base + 1, (fr >> 8) & 0xFF),
                (f, base + 4, 0x41),
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


def test_prefix_is_target_only_continuation_intact():
    ow = _three_voice_ow(n_frames=8)
    red, info = reduce.reduce_prefix(ow, prefix_frames=4, target=0)
    assert red is not None and info["target_voice"] == 0
    parts = writes.split_voices(red)
    prefix_accomp = [(k, f) for k in (1, 2) for f, _, _ in parts[k] if f < 4]
    assert prefix_accomp == [], "accompaniment must be gone from the prefix"
    target_prefix = [f for f, _, _ in parts[0] if f < 4]
    assert target_prefix, "target (melody) must remain in the prefix"
    orig = writes.split_voices(ow)
    for k in (1, 2):
        assert [t for t in parts[k] if t[0] >= 4] == [
            t for t in orig[k] if t[0] >= 4
        ], "continuation accompaniment must be byte-identical"
    assert parts[0] == orig[0], "target voice untouched everywhere"


def test_reduce_output_is_valid():
    ow = _three_voice_ow(n_frames=8)
    red, _ = reduce.reduce_prefix(ow, prefix_frames=4, target=0)
    assert writes.reencode(red)


def test_strip_ornament_drops_target_pw_in_prefix():
    ow = _three_voice_ow(n_frames=8)
    red, _ = reduce.reduce_prefix(ow, prefix_frames=4, target=0, strip_ornament=True)
    parts = writes.split_voices(red)
    pw_prefix = [(f, r) for f, r, _ in parts[0] if f < 4 and (r % 7) in (2, 3)]
    assert pw_prefix == [], "target PW ornament stripped from the prefix"
    pw_suffix = [(f, r) for f, r, _ in parts[0] if f >= 4 and (r % 7) in (2, 3)]
    assert pw_suffix == [
        (f, r) for f, r, _ in writes.split_voices(ow)[0] if f >= 4 and (r % 7) in (2, 3)
    ], "target PW preserved in the continuation"
