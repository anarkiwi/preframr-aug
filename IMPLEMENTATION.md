# preframr-aug — implementation plan

**Purpose.** Corpus augmentation that attacks the **free-running pathology** (the live blocker): the
event model *learns* (teacher-forced healthy) but *cannot generate* — free-running collapses, root
cause **copy-dominance (M4)**: copyable next-token acc 0.535 vs novel **0.194** (Tier-0 audits,
`preframr-xpt/data/audit/copy_novel_audit_v2.json`). The model copies because the corpus rewards
copying. Augmentation breaks that reward by making structure *non-literally-copyable*, forcing the
generative rule to beat the memorised span.

This doc is the build plan. The *why* lives in the two design docs this realises (read them first):
- `preframr-xpt/design/generation/transplant_augmentation_design.md` (melody×timbre debinding)
- `preframr-xpt/design/generation/prompt_interface_design.md` §"Reduction augmentation" (generate-from-seed)

**Clean slate.** This repo currently tracks only LICENSE + .gitignore. Build fresh against the
**current event substrate** — do not resurrect the deleted parse-domain melody-transfer/splice code.

## Non-negotiable architecture

1. **Write-domain only.** All augmentation manipulates ordered register writes — `(frame, reg, val)`
   triples — never event atoms or lanes. The encoder re-derives lanes (`NOTE_TABLE`, intervals,
   note-on programs) from the writes. This makes augmentation **agnostic to the atom encoding**
   (survives v2→v3 codec changes) and makes validity free.
2. **Validity is free, but only structural.** `stream.encode(ow, verify=True)` self-checks
   `decode(encode(ow)) == canonical_writes(ow)` — lossless round-trip for *any* well-formed write
   stream. It guarantees the augmented tune is a valid dump (zero training-pipeline change: the runner
   just sees a bigger dump dir + list file). It does **not** guarantee the music is *plausible* —
   that is the exclusion rules (§Wiring) + the render-fingerprint filter (§Gates).
3. **torch-free.** Depends on `preframr-tokens` (`events.oracle`, `events.stream`, `events.gestures`,
   `events.dataset`) and `preframr-audio` (`fingerprint`, `audio_driver`) as libraries. No model code.
4. **Leakage is a hard rule.** Donors AND hosts come from the **train split only**. eval_a/eval_b
   composers never contribute melody or instrument material. Augmented sets carry their own
   dataset-cache hash; the metrics ledger already flags cross-hash comparisons.

## The substrate APIs (verified present, 2026-06-14)

| Need | API |
|---|---|
| dump df → ordered writes | `events.oracle.ordered_writes(df) -> OrderedWrites` |
| writes / per-frame view | `OrderedWrites.triples() -> list[(frame,reg,val)]`, `.by_frame() -> list[list[(reg,val)]]` |
| writes → atoms (verified) | `events.stream.encode(ow, verify=True) -> list[int]` |
| atoms → writes | `events.stream.decode(tokens) -> list[(frame,reg,val)]` |
| oracle ground truth | `events.stream.canonical_writes(ow)` |
| single-speed guard | `events.stream.single_speed(ow) -> bool` |
| note segmentation | `events.gestures.cover/replay` |
| frame boundaries | `events.dataset.unit_starts(n_ids)` |
| sonic fingerprint | `preframr_audio.fingerprint.fingerprint_writes / fingerprint_batch` |
| isolated voice render | `preframr_audio.audio_driver.render_per_voice` (muted modulators keep oscillating) |

Dump df columns are `clock, irq, chipno, reg, val`. Voice register map: **v0 = regs 0–6, v1 = 7–13,
v2 = 14–20, filter/global = 21–24** (ctrl regs 4/11/18; gate = ctrl bit 0).

## Module layout

```
preframr_aug/
  writes.py       # CORE: dump <-> OrderedWrites, per-voice partition, splice, re-encode verified
  voices.py       # voice role hints (bass/lead/perc) + wiring analysis (sync/ring, filter routing)
  reduce.py       # reduction augmentation (strip voices/ornament from a prefix)
  transplant.py   # P1 instrument transplant, P2 melody transplant
  bank.py         # P0 instrument-program miner (onset-anchored program templates + dedupe)
  sonic.py        # sonic clustering of bank entries via fingerprint_writes (scaffold render)
  provenance.py   # per-tune sidecar (host, donor(s), voice, transform, anchor) + leakage guard
  cli.py          # batch: corpus dump dir -> augmented dump dir + list file + provenance
tests/
  test_roundtrip.py   # identity: dump -> writes -> dump re-encodes byte-exact
  test_splice.py      # per-voice swap is byte-exact when donor==host; exclusions honoured
  test_leakage.py     # train-split-only enforced; cross-split donor raises
  test_reduce.py      # reduction keeps target intact, strips only the prefix accompaniment
```

## CORE — `writes.py` (everything depends on this; build first)

- `load_ow(path) -> OrderedWrites` — read a `*dump.parquet`, `ordered_writes(df)`. Reject non
  single-speed (`single_speed(ow)` False) and digis up front.
- `split_voices(ow) -> {voice: list[(frame,reg,val)], "filter": [...]}` — partition triples by the
  voice register map. Filter/global (21–24) is its own bucket.
- `rebuild(parts) -> OrderedWrites` — reassemble buckets into a frame-ordered triple list, build a
  dump df (`clock` = running index, `irq` = frame, as `events.generate.writes_to_dump_df`), and
  `ordered_writes(df)`. The single re-assembly point.
- `reencode(ow) -> list[int]` — `encode(ow, verify=True)`; raises on round-trip failure (a bug, not a
  data condition). `emit_dump(ow, path)` writes the augmented `*dump.parquet`.
- **Invariant test (`test_roundtrip`):** `emit_dump(rebuild(split_voices(load_ow(p))))` re-encodes to
  the *same atoms* as `load_ow(p)` — partition+rebuild is identity. This is the load-bearing
  correctness gate; nothing else is trustworthy until it is green on a corpus sample.

## `voices.py` — role + wiring (the exclusion logic)

- `roles(ow) -> {voice: "bass"|"lead"|"perc"|"unknown"}` — heuristic only (pitch-range median, noise
  waveform %, gate rhythm). Role drives donor↔host matching and percussion exclusion. Mis-label is
  cheap (a worse augmentation, never an invalid one).
- `wiring(ow) -> {sync, ring, filter_routed}` per voice, read from registers: sync/ring = ctrl bits
  1/2; filter routing = reg-23 routing bits. **Exclusions (from the lane-demux wiring analysis,
  hard):** never transplant onto/from a voice with sync/ring set *or its ring neighbour* (the edge,
  keyed by physical voice index, carries the music — never separate a modulator from its carrier);
  filter routing always stays with the host; never drop a silent-but-modulating voice. An
  unclassifiable span is left untouched, never forced.

## Build order — fastest signal on M4 first

The transplant design lists P0 (bank) first for its standalone value. **This plan inverts that**: the
live goal is to *progress the pathology*, and the cheapest arms that produce a trainable M4-attacking
corpus need only the CORE + `voices`. Sequence by signal-per-effort:

### M1 — Reduction augmentation (`reduce.py`) — cheapest, load-bearing
Derive `(melody-only prefix → full-texture continuation)` pairs from the corpus itself: take a real
window, strip voices 1–2 (and optionally the lead's ornament) from the **first K frames only**, keep
the target intact. Teaches arrange-from-a-seed directly (attacks M4 *and* M1, and is the load-bearing
bet for phrase-prompting later). Pure write-domain deletion on the prefix; no donor, no bank. **First
trainable arm.**

### M2 — Instrument transplant, intra-corpus (`transplant.py` P1)
Keep host music, replace a voice's *timbre*: at each host-voice onset, fire a donor voice's onset
program (donor AD/SR + ctrl/waveform walk + PW + HR; host pitch/durations/phrasing). Donor drawn from
another **train-split** tune's same-role voice (the bank, M4 below, only enriches donor diversity —
not required to start). Rewrite voice v's writes from `host note list × donor program template`,
`reencode`. Honour all §Wiring exclusions. Breaks the melody×timbre binding.

### M3 — Melody transplant (`transplant.py` P2) — riskiest, build after A/B on M1+M2
Keep host arrangement, replace a voice's *line*: donor note list (onsets, durations, intervals) drives
the host voice's instrument program. Re-keying: snap donor median pitch to host pitch-class histogram
mode, clamp to host voice range; keep the scorer trivial (range fit + out-of-key count) and let the
dosage A/B decide if naive anchoring suffices before building consonance logic. Whole phrases cut at
gate-off boundaries; role-matched.

### M4 — Instrument bank + sonic clustering (`bank.py`, `sonic.py`) — enrichment, standalone value
**Gate this on a measurement first (P0.0):** the design assumes ~98% exact onset-program recurrence
per tune — *measure it on the real corpus before building the miner* (a one-off audit: per (tune,
voice) collapse exact-recurring onset programs, report the recurrence rate and instruments/tune). If
recurrence is high, build the miner (onset-anchored program template, exact-key dedupe within tune,
cross-corpus dedupe) and the sonic clustering (`fingerprint_writes` with an instrument scaffold —
single voice, others silent, vol 15, two fixed pitches C2+C4, cold AND warm renders for the
prior-state ADSR delta; calibrate the metric against re-render INERT pairs and waveform-flip CONTRAST
pairs per the `fidelity.calibrate()` precedent). The bank then feeds M2/M3 with diverse, sonically
clustered donors and the future phrase compiler's patch realism. Artifact: parquet + JSON index,
**cached/regenerated, never committed** (derived from copyrighted HVSC — provenance pointers only).

## Output contract (`cli.py`)

`preframr-aug <train-dump-dir> --out <aug-dir> --arm reduce|instrument|melody --dose 0.25 --seed N`
→ writes augmented `*dump.parquet` files + a `provenance.jsonl` sidecar (one record per tune: host,
donor(s), voice, transform, anchor) + a list file. The training runner consumes `<aug-dir>` exactly
like a real dump dir — **zero pipeline change**. Provenance keeps the memorization audit honest (a
generated continuation matching a *transplant* is still corpus material).

## Gates (every arm)

1. **Structural** — `encode(verify=True)` passes (automatic) + `test_roundtrip` green on a sample.
2. **Plausibility filter** — render + `fingerprint_writes`; reject tunes outside the corpus
   fingerprint band (reuse the generation-quality-gate machinery as the *augmentation* filter);
   report rejection rate.
3. **Leakage** — `test_leakage`: donors/hosts train-split only; a cross-split donor must raise.
4. **Pre-train cheap read** — `learnability_triage` on augmented vs natural corpus: the augmented set
   must look like *more corpus*, not a new dialect (per-frame h_k and copy-fraction within the natural
   band). Run before spending a training run.

## The decision experiment

Dosage A/B at canonical tier, target arm first: baseline vs +25% vs +100% augmented train set
(reduction first, then instrument-only, then +melody). **Decision metric: eval_b held-out-composer
content tier** (the recombination claim is cross-composer) + the **copy/novel split re-measured**
(does novel-fraction rise? — the direct M4 read) + quality-gate no-regression + the free-running gap
audit re-run (does the free-running collapse soften?). Confound rules and the train-split leakage rule
per the transplant design.

## Honest risks

- A transplant has **no correctness oracle** — it is new music. The gates make it *valid* and
  *plausible*; only the dosage A/B makes it *useful*. Reduction (M1) is the safer bet (it only deletes
  real corpus material) — that is why it leads.
- The ~98% program-recurrence and the role heuristics are **assumptions to measure**, not givens.
  Measure (P0.0, role-label spot-checks) before building on them.
- If the A/B shows no novel-fraction lift at any dose, the corpus-recombination hypothesis for M4 is
  wrong and the program escalates to Tier-4 (DAgger with the re-canonicalisation oracle), which is a
  *model-side* bet and out of this repo's scope.

## Non-goals

Synthetic de-novo instruments/melodies (this is recombination of real material); cross-engine program
translation; digi/multispeed material (encoder scope); any model or alphabet change.
