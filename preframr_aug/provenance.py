"""Augmentation provenance + the train-split leakage guard. Every augmented tune carries a record
(host, donors, voice, transform, anchor) so the memorization audit stays honest, and donors/hosts are
hard-restricted to the train split -- eval composers never contribute melody or instrument material.
"""

from __future__ import annotations

import json
from pathlib import Path

_SPLITS = ("train", "eval_a", "eval_b")


class LeakageError(Exception):
    """A donor or host came from outside the allowed (train) split."""


def split_of(path) -> str:
    """The corpus split a dump path belongs to, by its path components (``train`` / ``eval_a`` /
    ``eval_b``); ``unknown`` when no split segment is present."""
    parts = Path(path).parts
    for split in _SPLITS:
        if split in parts:
            return split
    for part in parts:
        if part.startswith("eval_b"):
            return "eval_b"
    return "unknown"


def guard_train_split(*paths, allow=("train", "unknown")) -> None:
    """Raise :class:`LeakageError` unless every path is in an allowed split. ``unknown`` is allowed so a
    flat dump dir (no split segment) still works; pass ``allow=("train",)`` to enforce strictly.
    """
    for path in paths:
        split = split_of(path)
        if split not in allow:
            raise LeakageError(f"{path}: split {split!r} not in {allow}")


def record(out_path, host, transform, voice, donors=None, anchor=None, **extra) -> dict:
    """Build one provenance record. ``host``/``donors`` are source dump paths; ``voice`` the touched
    voice; ``anchor`` the transform's alignment point (e.g. prefix length)."""
    rec = {
        "out": str(Path(out_path).name),
        "transform": transform,
        "host": str(host),
        "host_split": split_of(host),
        "donors": [str(d) for d in (donors or [])],
        "voice": voice,
        "anchor": anchor,
    }
    rec.update(extra)
    return rec


def write_jsonl(records, path) -> None:
    """Append-write provenance records as JSON lines."""
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
