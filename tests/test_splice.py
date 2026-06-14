"""M2 instrument transplant: a self-transplant (donor==host, recurring program) is byte-exact, a
cross-donor transplant stays valid + keeps host pitch, and wiring exclusions are honoured.
"""

import numpy as np
import pandas as pd

from preframr_tokens.events import oracle, stream
from preframr_aug import transplant, voices, writes


def _recurring_ow(n_notes=5, note_len=4, wave=0x40, ad=0x08, sr=0xA9, freqs=None):
    """A one-voice tune whose every note fires the SAME onset program (recurring instrument); voice 0
    only. freqs sets each note's pitch (host melody)."""
    freqs = freqs or [0x1000 + 0x80 * i for i in range(n_notes)]
    rows = []
    frame = 0
    for i in range(n_notes):
        fr = freqs[i]
        rows += [
            (frame, 0, fr & 0xFF),
            (frame, 1, (fr >> 8) & 0xFF),
            (frame, 5, ad),
            (frame, 6, sr),
            (frame, 4, wave | 0x01),
        ]
        rows.append((frame + note_len - 1, 4, wave))
        frame += note_len
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


def test_self_transplant_is_byte_exact():
    ow = _recurring_ow()
    new, _ = transplant.instrument_transplant(ow, ow, 0, 0)
    assert writes.reencode(new) == stream.encode(ow), "donor==host must be identity"


def test_cross_transplant_keeps_host_pitch_and_is_valid():
    host = _recurring_ow(
        wave=0x40, ad=0x08, freqs=[0x1000 + 0x80 * i for i in range(5)]
    )
    donor = _recurring_ow(wave=0x20, ad=0x2C, sr=0x44, freqs=[0x2000] * 5)
    new, _info = transplant.instrument_transplant(host, donor, 0, 0)
    assert writes.reencode(new)
    host_freq = [(f, r, v) for f, r, v in host.triples() if r in (0, 1)]
    new_freq = [(f, r, v) for f, r, v in new.triples() if r in (0, 1)]
    assert new_freq == host_freq, "host pitch line must be preserved"
    donor_wave = [v for f, r, v in new.triples() if r == 4 and v & ~transplant.GATE]
    assert any(v & 0x20 for v in donor_wave), "donor waveform (saw) must appear"


def test_pick_pair_excludes_ring_voice():
    rows = []
    for f in range(8):
        for v in range(3):
            b = 7 * v
            ctrl = 0x41 if v != 1 else (0x41 | 0x04)
            rows += [
                (f, b + 0, 0x00 if v == 0 else (0x40 << 0) & 0xFF),
                (f, b + 1, 0x08 + v),
                (f, b + 4, ctrl),
                (f, b + 5, 0x08),
                (f, b + 6, 0xA9),
            ]
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
    ow = oracle.ordered_writes(df)
    safe = voices.transplantable(ow)
    assert (
        1 not in safe and 0 not in safe
    ), "ring voice 1 and its source voice 0 excluded"


def test_transplant_tune_from_files(tmp_path):
    host = tmp_path / "host.dump.parquet"
    donor = tmp_path / "donor.dump.parquet"
    writes.emit_dump(
        _recurring_ow(wave=0x40, freqs=[0x1000 + 0x80 * i for i in range(5)]), host
    )
    writes.emit_dump(
        _recurring_ow(wave=0x20, ad=0x2C, sr=0x44, freqs=[0x1800] * 5), donor
    )
    new_ow, rec = transplant.transplant_tune(str(host), str(donor))
    assert new_ow is not None and rec["transform"] == "instrument"
    assert rec["donors"] == [str(donor)] and rec["voice"] == 0
    assert writes.reencode(new_ow)
