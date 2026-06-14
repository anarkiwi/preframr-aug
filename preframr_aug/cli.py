"""Batch augmentation CLI: a train dump dir -> an augmented dump dir + provenance.jsonl + list file.
The training runner consumes the output exactly like a real dump dir (zero pipeline change). Arms:
``reduce`` (M1, melody-seed prefixes), ``instrument`` (M2, donor-timbre transplants)."""

from __future__ import annotations

import argparse
import glob
import os
import random

from . import provenance, reduce, transplant, writes


def _dumps(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.dump.parquet"), recursive=True))


def _load(path):
    try:
        return writes.load_ow(path)
    except (ValueError, OSError):
        return None


def _run_reduce(paths, _rng, args, out_dir):
    """One reduction per selected host: melody-only prefix into the intact continuation."""
    records = []
    for host in paths:
        ow = _load(host)
        if ow is None:
            continue
        reduced, info = reduce.reduce_prefix(
            ow, args.prefix_frames, strip_ornament=args.strip_ornament
        )
        if reduced is None:
            continue
        name = f"{_stem(host)}__reduce{args.prefix_frames}.dump.parquet"
        if not _emit(reduced, os.path.join(out_dir, name), args):
            continue
        records.append(
            provenance.record(
                name,
                host=host,
                transform="reduce",
                voice=info["target_voice"],
                anchor=args.prefix_frames,
                kept_voices=info.get("kept_voices"),
            )
        )
    return records


def _run_instrument(paths, rng, args, out_dir):
    """One instrument transplant per selected host with a random train-split donor."""
    records = []
    for host in paths:
        donor = rng.choice([p for p in paths if p != host]) if len(paths) > 1 else None
        if donor is None:
            continue
        try:
            new_ow, rec = transplant.transplant_tune(host, donor)
        except (provenance.LeakageError, ValueError, OSError):
            continue
        if new_ow is None:
            continue
        if not _emit(new_ow, os.path.join(out_dir, rec["out"]), args):
            continue
        records.append(rec)
    return records


def _emit(ow, path, args) -> bool:
    """Write the augmented dump, optionally gating on the sonic plausibility band."""
    if args.band is not None:
        from .sonic import (
            fingerprint_ow,
            in_band,
        )  # pylint: disable=import-outside-toplevel

        if not in_band(fingerprint_ow(ow), args.band):
            return False
    writes.emit_dump(ow, path)
    return True


def _stem(path) -> str:
    name = os.path.basename(path)
    return name[: -len(".dump.parquet")] if name.endswith(".dump.parquet") else name


def run(args) -> int:
    """Select a dose of train-split hosts, apply the chosen arm, write dumps + provenance + list."""
    os.makedirs(args.out, exist_ok=True)
    paths = _dumps(args.train_dir)
    provenance.guard_train_split(*paths)
    rng = random.Random(args.seed)
    n = max(1, round(len(paths) * args.dose))
    hosts = rng.sample(paths, min(n, len(paths)))
    arms = {"reduce": _run_reduce, "instrument": _run_instrument}
    records = arms[args.arm](hosts, rng, args, args.out)
    provenance.write_jsonl(records, os.path.join(args.out, "provenance.jsonl"))
    with open(
        os.path.join(args.out, "augmented.list"), "w", encoding="utf-8"
    ) as handle:
        for rec in records:
            handle.write(os.path.join(args.out, rec["out"]) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="preframr write-domain augmentation")
    parser.add_argument("train_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=("reduce", "instrument"), default="reduce")
    parser.add_argument("--dose", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-frames", type=int, default=400)
    parser.add_argument("--strip-ornament", action="store_true")
    parser.add_argument("--filter", action="store_true", help="sonic plausibility gate")
    parser.add_argument("--band-sample", type=int, default=40)
    args = parser.parse_args()
    args.band = _corpus_band(args) if args.filter else None
    written = run(args)
    print(f"wrote {written} augmented dumps to {args.out}")


def _corpus_band(args):
    """Build the sonic acceptance band from a sample of the source corpus (the natural envelope an
    augmented tune must sit inside)."""
    from .sonic import band, fingerprint_ow  # pylint: disable=import-outside-toplevel

    rng = random.Random(args.seed)
    paths = _dumps(args.train_dir)
    rng.shuffle(paths)
    fps = []
    for path in paths:
        ow = _load(path)
        if ow is not None:
            fps.append(fingerprint_ow(ow))
        if len(fps) >= args.band_sample:
            break
    return band(fps) if fps else None


if __name__ == "__main__":
    main()
