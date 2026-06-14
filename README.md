# preframr-aug

[![CI](https://github.com/anarkiwi/preframr-aug/actions/workflows/ci.yml/badge.svg)](https://github.com/anarkiwi/preframr-aug/actions/workflows/ci.yml)

Write-domain corpus augmentation for the preframr SID event model — reduction +
transplant arms that attack copy-dominance, torch-free.

## Why

The event model *learns* (teacher-forced) but *cannot free-run*: generation collapses
because the corpus rewards **copying** the memorised span over applying the generative
rule (copyable next-token acc 0.535 vs novel 0.194). `preframr-aug` breaks that reward by
making structure **non-literally-copyable** — the same real music, recombined so the copy
no longer wins. The full rationale and build plan live in
[IMPLEMENTATION.md](IMPLEMENTATION.md).

Everything operates on **ordered register writes** — `(frame, reg, val)` triples — never
event atoms. The encoder re-derives lanes from the writes, so augmentation is agnostic to
the atom encoding (it survives codec changes) and structural validity is free:
`encode(ow, verify=True)` self-checks the lossless round-trip. Plausibility (that the
output still sounds like the corpus) is enforced separately by the wiring exclusions and
the sonic fingerprint band.

## Status

Built against the current event substrate (preframr-tokens ≥ 0.51.0 /
preframr-audio ≥ 0.5.9). Two trainable arms have landed; the training runner consumes the
output exactly like a real dump dir — **zero pipeline change**.

| Arm | Module | What it does |
|---|---|---|
| **CORE** | `writes.py` | dump ↔ OrderedWrites, per-voice partition, frame-ordered rebuild, verified re-encode |
| **M1 · reduce** | `reduce.py` | melody-only prefix → intact continuation (pure write-domain deletion; first trainable arm) |
| **M2 · instrument** | `transplant.py` | donor onset-program replayed at host onsets — host pitch/phrasing kept, timbre swapped |

Supporting modules: `voices.py` (role + wiring exclusions), `provenance.py` (records +
the train-split leakage guard), `sonic.py` (mel-fingerprint plausibility band), `cli.py`
(batch driver).

Deferred by the plan's own gating: **M3** melody transplant (build after the M1+M2 dosage
A/B), **M4** instrument bank + sonic clustering (gate on the onset-program-recurrence
measurement first). See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the sequencing and the
honest risks.

## Install

```sh
pip install preframr-aug
```

From source:

```sh
git clone https://github.com/anarkiwi/preframr-aug
cd preframr-aug
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Runtime deps: numpy, pandas, pyarrow, preframr-tokens,
preframr-audio (no torch).

## CLI

```sh
preframr-aug <train-dump-dir> --out <aug-dir> --arm reduce --dose 0.25 --seed 0
preframr-aug <train-dump-dir> --out <aug-dir> --arm instrument --dose 0.25 --filter
```

Selects `dose` of train-split hosts, applies the arm, and writes augmented
`*.dump.parquet` files plus `provenance.jsonl` (one record per tune: host, donor(s),
voice, transform, anchor) and `augmented.list`.

| Flag | Default | Meaning |
|---|---|---|
| `--arm` | `reduce` | `reduce` (M1) or `instrument` (M2) |
| `--dose` | `0.25` | fraction of corpus hosts to augment |
| `--seed` | `0` | RNG seed (host/donor selection) |
| `--prefix-frames` | `400` | reduce: frames of melody-only seed |
| `--strip-ornament` | off | reduce: also strip the target's own PW ornament from the prefix |
| `--filter` | off | gate each output on the sonic plausibility band |
| `--band-sample` | `40` | tunes sampled to build that band |

## API

The canonical pipeline — load, transform on the triples, verify, emit:

```python
from preframr_aug import writes, reduce, transplant

ow = writes.load_ow("tune.dump.parquet")          # -> OrderedWrites (rejects multispeed)

# M1 — reduction
reduced, info = reduce.reduce_prefix(ow, prefix_frames=400)
writes.reencode(reduced)                           # verified encode; raises on round-trip failure
writes.emit_dump(reduced, "tune__reduce400.dump.parquet")

# M2 — instrument transplant (leakage-guarded end to end)
new_ow, record = transplant.transplant_tune("host.dump.parquet", "donor.dump.parquet")
```

### `writes` — CORE

Everything manipulates the triples these expose and re-assembles through `rebuild`, so
augmentation stays agnostic to the atom encoding.

- `load_ow(path) -> OrderedWrites` — read a `*.dump.parquet`; raises on non-single-speed material (out of codec scope).
- `split_voices(ow) -> dict` — partition triples into buckets keyed `0, 1, 2, "filter"`. Voice register map: **v0 = regs 0–6, v1 = 7–13, v2 = 14–20, filter/global = 21–24**.
- `rebuild(parts) -> OrderedWrites` — reassemble buckets into one frame-ordered stream (the single re-assembly point).
- `reencode(ow) -> list[int]` (alias `atoms`) — verified encode: `decode(encode(ow)) == canonical_writes(ow)`.
- `emit_dump(ow, path) -> None` — write the augmented dump.
- `voice_key(reg) -> 0 | 1 | 2 | "filter"`, plus `VOICE_REGS`, `FILTER_REGS`, `VOICE_KEYS`.

### `voices` — role + wiring (the exclusion logic)

- `roles(ow) -> {voice: "bass" | "lead" | "perc" | "unknown"}` — heuristic (pitch median, noise %, gate rhythm); a mislabel weakens an augmentation, never invalidates it.
- `wiring(ow) -> {voice: {sync, ring, filter_routed}}` — read from ctrl bits 1/2 and the reg-23 routing bits.
- `transplantable(ow) -> set[int]` — voices safe as donor/host: excludes any sync/ring voice **and its modulation source** (SID voice `v` is driven by voice `(v-1) mod 3`) — never separate a modulator from its carrier.

### `reduce` — M1

- `reduce_prefix(ow, prefix_frames, target=None, strip_ornament=False) -> (OrderedWrites | None, info)` — strip non-target voice writes from frames `< K`, keep the continuation intact. Returns `(None, info)` when no melodic target is found (the caller skips).

### `transplant` — M2

- `instrument_transplant(host_ow, donor_ow, host_voice, donor_voice) -> (OrderedWrites | None, info)` — replay the donor's onset program at each host onset; host freq + gate edges kept.
- `pick_pair(host_ow, donor_ow) -> (host_voice, donor_voice) | None` — same-role pair, both transplant-safe; percussion is never transplanted.
- `transplant_tune(host_path, donor_path) -> (OrderedWrites | None, record | None)` — leakage-guarded; returns `(None, None)` when no safe role-matched pair exists.

### `provenance` — records + leakage guard

- `guard_train_split(*paths, allow=("train", "unknown")) -> None` — raises `LeakageError` if any path is outside the allowed split (pass `allow=("train",)` to enforce strictly).
- `split_of(path) -> str`, `record(out_path, host, transform, voice, donors=None, anchor=None, **extra) -> dict`, `write_jsonl(records, path) -> None`.

### `sonic` — plausibility filter

- `fingerprint_ow(ow, frames=600, feature="mel") -> np.ndarray` — mel fingerprint via the canonical render scaffold.
- `band(fingerprints, k=3.0) -> (lo, hi)` — per-dimension acceptance band (mean ± k·std) over reference fingerprints.
- `in_band(fingerprint, bounds, tol=0.02) -> bool` — reject if more than `tol` of dimensions fall outside the band.

## Development

```sh
pip install -e ".[dev]"
./run_tests.sh        # black --check, pylint, pyright, pytest --cov (floor 80%)
```

CI runs the suite on Python 3.10 / 3.11 / 3.12. Tests cover the load-bearing
partition+rebuild identity gate, per-voice splice byte-exactness, train-split leakage
enforcement, the reduce and instrument arms, and a fixture-based audio render of each arm.

## Releases

Tagged releases publish to PyPI via a
[trusted publisher](https://docs.pypi.org/trusted-publishers/) (OIDC — no API token) —
see [`.github/workflows/release.yml`](.github/workflows/release.yml). To cut a release,
push a `vX.Y.Z` tag and publish a GitHub Release; setuptools-scm derives the version from
the tag.

> One-time PyPI setup: register the trusted publisher for project `preframr-aug` —
> owner `anarkiwi`, repository `preframr-aug`, workflow `release.yml`, environment `pypi`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
