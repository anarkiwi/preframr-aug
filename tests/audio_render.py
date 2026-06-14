"""Faithful render of a dump or an augmented OrderedWrites to a WAV, reusing the same path that plays
real tunes (RegLogParser -> prepare_df_for_audio -> pyresidfp). Augmentation works in the frame domain,
so an OrderedWrites is first emitted as a raw dump with real single-speed cycle timing (load_ow only
admits single-speed tunes, frame period PAL_IRQ) and then rendered through the identical path, making an
original-vs-augmented A/B faithful and comparable."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_tokens import (
    RegLogParser,
    default_tokenizer_args,
    prepare_df_for_audio,
    read_initial_irq,
)
from preframr_audio import render_to_wav, sidq

PAL_IRQ = 19656
DUMP_COLUMNS = ("clock", "irq", "chipno", "reg", "val")


def real_timing_dump_df(ow, irq: int = PAL_IRQ) -> pd.DataFrame:
    """Frame-indexed OrderedWrites -> a raw-dump DataFrame with real single-speed cycle timing: frame
    ``f`` fires at clock ``f*irq`` and its writes occupy strictly increasing sub-cycle slots, so the
    parser recovers the canonical ~50 Hz raster instead of the synthetic clock ``emit_dump`` uses.
    """
    clocks, irqs, regs, vals = [], [], [], []
    last_frame = -1
    slot = 0
    for frame, reg, val in ow.triples():
        if frame != last_frame:
            last_frame = frame
            slot = 0
        cycle = int(frame) * irq
        clocks.append(cycle + slot)
        irqs.append(cycle)
        regs.append(int(reg))
        vals.append(int(val))
        slot += 1
    return pd.DataFrame(
        {
            "clock": np.array(clocks, dtype=np.int64),
            "irq": np.array(irqs, dtype=np.int64),
            "chipno": np.zeros(len(clocks), dtype=np.int64),
            "reg": np.array(regs, dtype=np.int64),
            "val": np.array(vals, dtype=np.int64),
        }
    )


def render_dump_to_wav(dump_path, wav_path, cents: int = 50) -> int:
    """Render a ``*.dump.parquet`` to a WAV via the canonical play path; returns the sample count."""
    parser = RegLogParser(args=default_tokenizer_args(cents=cents))
    rotations = parser.parse(str(dump_path), max_perm=1, require_pq=False, reparse=True)
    xdf = next(iter(rotations), None)
    if xdf is None:
        raise ValueError(f"no rotations parsed from {dump_path}")
    irq = read_initial_irq(xdf)
    df, reg_widths = prepare_df_for_audio(
        xdf, {}, irq, sidq(), strict=False, cents=cents
    )
    return int(render_to_wav(df, str(wav_path), reg_widths, irq, cents=cents))


def render_ow_to_wav(ow, wav_path, cents: int = 50) -> int:
    """Render an (augmented) OrderedWrites to a WAV by emitting a real-timing dump and reusing
    :func:`render_dump_to_wav`; returns the sample count."""
    workdir = Path(tempfile.mkdtemp(prefix="aug-render-"))
    tmp = workdir / "aug.dump.parquet"
    try:
        real_timing_dump_df(ow).to_parquet(tmp)
        return render_dump_to_wav(tmp, wav_path, cents=cents)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
