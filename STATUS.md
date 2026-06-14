# Status

Built against the current event substrate (preframr-tokens 0.51.0 / preframr-audio 0.5.9),
write-domain only, torch-free. `./run_tests.sh` green (black + pylint + pyright + pytest, coverage
94%); the load-bearing partition+rebuild identity gate passes on real corpus tunes.

## Landed

- **CORE** `writes.py` — dump <-> OrderedWrites, per-voice partition, frame-ordered rebuild, verified
  re-encode, dump emit. Identity test green on 12 corpus tunes + synthetic.
- **`voices.py`** — bass/lead/perc role hints; sync/ring/filter wiring; `transplantable()` exclusions
  (never separate a modulator from its carrier; SID voice v is sync/ring-driven by voice (v-1) mod 3).
- **M1 `reduce.py`** — reduction augmentation (melody-only prefix -> intact continuation). The first
  trainable arm; pure write-domain prefix deletion.
- **M2 `transplant.py`** — instrument transplant: donor onset-program (AD/SR + waveform/PW walk)
  replayed at host onsets, gate from host, host pitch/phrasing kept. Self-transplant byte-exact;
  wiring exclusions honoured; leakage-guarded.
- **`provenance.py`** — per-tune records + the hard train-split leakage guard.
- **`sonic.py`** — mel-fingerprint plausibility band (the augmentation filter).
- **`cli.py`** — `preframr-aug <train-dir> --out --arm reduce|instrument --dose --seed [--filter]`
  -> augmented dumps + `provenance.jsonl` + `augmented.list`. Zero training-pipeline change.

## Run

```
preframr-aug <train-dump-dir> --out <aug-dir> --arm reduce --dose 0.25 --seed 0
preframr-aug <train-dump-dir> --out <aug-dir> --arm instrument --dose 0.25 --filter
```

## Deferred (by the plan's own gating)

- **M3 melody transplant** — build after the M1+M2 dosage A/B (riskiest arm).
- **M4 instrument bank + sonic clustering** — gate on the P0.0 onset-program-recurrence measurement
  first; the bank then enriches M2/M3 donor diversity. Bank artifacts are cached, never committed
  (derived from HVSC; provenance pointers only).
- The decision experiment (baseline vs +25% vs +100%) is xpt-side and consumes this repo.
