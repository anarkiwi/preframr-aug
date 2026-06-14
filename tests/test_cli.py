"""CLI smoke: the reduce arm over a synthetic train dir writes valid dumps + provenance + list."""

import numpy as np
import pandas as pd
from types import SimpleNamespace

from preframr_tokens.events import oracle
from preframr_aug import cli, writes


def _emit_synth(path, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(12):
        for v in range(3):
            b = 7 * v
            fr = 0x0800 * (3 - v) + int(rng.integers(0, 0x100))
            rows += [
                (f, b + 0, fr & 0xFF),
                (f, b + 1, (fr >> 8) & 0xFF),
                (f, b + 4, 0x41),
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
    writes.emit_dump(oracle.ordered_writes(df), path)


def test_cli_reduce_arm(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for i in range(4):
        _emit_synth(train / f"t{i}.dump.parquet", i)
    out = tmp_path / "aug"
    args = SimpleNamespace(
        train_dir=str(train),
        out=str(out),
        arm="reduce",
        dose=1.0,
        seed=0,
        prefix_frames=4,
        strip_ornament=False,
        band=None,
    )
    written = cli.run(args)
    assert written >= 1
    assert (out / "provenance.jsonl").exists()
    assert (out / "augmented.list").exists()
    for line in (out / "augmented.list").read_text().splitlines():
        assert writes.reencode(writes.load_ow(line.strip()))
