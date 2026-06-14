"""Reduction augmentation (M1): strip the accompaniment from the FIRST K frames only, leaving a
melody-only prefix that resolves into the intact full-texture continuation. Pure write-domain deletion
on real corpus material -- no donor, no bank -- so it teaches arrange-from-a-seed directly while only
ever removing writes that were really there."""

from __future__ import annotations

from . import voices, writes

_PW_OFFSETS = (2, 3)


def _target_voice(role_map) -> int | None:
    """The voice the melody prefix keeps: the lead, else the highest-pitched non-perc voice."""
    for voice, role in role_map.items():
        if role == "lead":
            return voice
    pitched = [v for v, r in role_map.items() if r in ("bass", "unknown")]
    return max(pitched) if pitched else None


def _keep_voices(ow, target) -> set:
    """Voices whose prefix writes survive: the target plus any voice that sync/ring-drives it (removing
    a modulator of the kept voice would change the kept voice's sound)."""
    wired = voices.wiring(ow)
    keep = {target}
    if wired[target]["sync"] or wired[target]["ring"]:
        keep.add((target - 1) % 3)
    return keep


def reduce_prefix(ow, prefix_frames: int, target=None, strip_ornament: bool = False):
    """Strip non-target voice writes (optionally the target's own PW ornament) from frames ``< K``;
    the continuation's writes are preserved in content (frames emptied by the strip collapse under the
    codec's frame densification, so the seed is re-paced shorter -- intended). Returns
    ``(reduced_ow, info)`` or ``(None, info)`` when no melodic target is found (caller skips).
    """
    role_map = voices.roles(ow)
    target = _target_voice(role_map) if target is None else target
    info = {
        "transform": "reduce",
        "prefix_frames": int(prefix_frames),
        "target_voice": target,
        "strip_ornament": bool(strip_ornament),
    }
    if target is None:
        return None, info
    keep = _keep_voices(ow, target)
    info["kept_voices"] = sorted(keep)
    parts = writes.split_voices(ow)
    for key in writes.VOICE_KEYS:
        if key == writes.FILTER or key in keep:
            if not (strip_ornament and key == target):
                continue
        kept = []
        for frame, reg, val in parts[key]:
            in_prefix = frame < prefix_frames
            is_ornament = key == target and (reg % 7) in _PW_OFFSETS
            if in_prefix and (key not in keep or (strip_ornament and is_ornament)):
                continue
            kept.append((frame, reg, val))
        parts[key] = kept
    return writes.rebuild(parts), info
