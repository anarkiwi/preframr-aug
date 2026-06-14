"""Instrument transplant (M2): keep the host's music (pitch, durations, phrasing) and replace a
voice's TIMBRE with a donor voice's onset program (AD/SR + waveform/PW walk), gate timing taken from
the host. Breaks the melody x timbre binding that drives copy-dominance. Donor and host are
same-role train-split voices; all wiring exclusions are honoured."""

from __future__ import annotations

from . import provenance, voices, writes

GATE = 0x01
_FREQ = (0, 1)
_TIMBRE = (2, 3, 4, 5, 6)
_CTRL = 4


def _voice_settled(ow, voice):
    """Per-frame settled ``[r0..r6]`` for ``voice`` and the raw per-frame writes; the substrate for
    onset detection and program extraction."""
    by_frame = {}
    for frame, reg, val in ow.triples():
        if reg < 21 and reg // 7 == voice:
            by_frame.setdefault(frame, []).append((reg % 7, val))
    grid = []
    cur = [0] * 7
    for frame in range(ow.n_frames):
        for off, val in by_frame.get(frame, []):
            cur[off] = val
        grid.append(cur.copy())
    return grid, by_frame


def _onsets(grid):
    """Frames where the voice gate rises 0->1 (note starts)."""
    out = []
    prev = 0
    for frame, state in enumerate(grid):
        gate = state[_CTRL] & GATE
        if gate and not prev:
            out.append(frame)
        prev = gate
    return out


def _program(by_frame, onsets):
    """The donor onset program: the first note's timbre WRITE EVENTS as ordered ``(rel, off, val)``
    (rel = frame - onset), preserving intra-frame write order; ctrl events keep their value with the
    gate bit stripped (the gate comes from the host). Returns ``(events, span)``."""
    if not onsets:
        return None, 0
    start = onsets[0]
    end = onsets[1] if len(onsets) > 1 else max(by_frame, default=start) + 1
    events = []
    for frame in range(start, end):
        for off, val in by_frame.get(frame, []):
            if off in _TIMBRE:
                events.append(
                    (frame - start, off, val & ~GATE if off == _CTRL else val)
                )
    return events, end - start


def instrument_transplant(host_ow, donor_ow, host_voice, donor_voice):
    """Rewrite ``host_voice``'s timbre from ``donor_voice``'s onset program: host freq + pre-onset
    writes kept verbatim; the donor's note write-events replay at each host onset (clipped to the host
    note), ctrl gated by the host, with host gate-edges overlaid. Returns ``(new_ow, info)``.
    """
    info = {
        "transform": "instrument",
        "host_voice": host_voice,
        "donor_voice": donor_voice,
    }
    host_grid, host_by = _voice_settled(host_ow, host_voice)
    donor_grid, donor_by = _voice_settled(donor_ow, donor_voice)
    host_onsets = _onsets(host_grid)
    events, _ = _program(donor_by, _onsets(donor_grid))
    if not host_onsets or not events:
        return None, info
    base = 7 * host_voice
    first = host_onsets[0]
    bounds = host_onsets[1:] + [host_ow.n_frames]

    new = [
        (frame, base + off, val)
        for frame in sorted(host_by)
        for off, val in host_by[frame]
        if off in _FREQ or frame < first
    ]
    emitted = set()
    wave = 0
    for onset, note_end in zip(host_onsets, bounds):
        for rel, off, val in events:
            frame = onset + rel
            if frame >= note_end:
                continue
            if off == _CTRL:
                wave = val
                val = val | (host_grid[frame][_CTRL] & GATE)
            new.append((frame, base + off, val))
            if off == _CTRL:
                emitted.add(frame)
    prev_gate = host_grid[first - 1][_CTRL] & GATE if first > 0 else 0
    for frame in range(first, host_ow.n_frames):
        gate = host_grid[frame][_CTRL] & GATE
        if gate != prev_gate and frame not in emitted:
            new.append((frame, base + _CTRL, wave | gate))
        prev_gate = gate

    parts = writes.split_voices(host_ow)
    parts[host_voice] = sorted(new, key=lambda t: t[0])
    return writes.rebuild(parts), info


def pick_pair(host_ow, donor_ow):
    """Choose a same-role (host_voice, donor_voice) both transplant-safe (no sync/ring edge), or None.
    Percussion is never transplanted."""
    host_roles, donor_roles = voices.roles(host_ow), voices.roles(donor_ow)
    host_ok, donor_ok = voices.transplantable(host_ow), voices.transplantable(donor_ow)
    for hv in sorted(host_ok):
        role = host_roles[hv]
        if role in ("perc", "unknown"):
            continue
        for dv in sorted(donor_ok):
            if donor_roles[dv] == role:
                return hv, dv
    return None


def transplant_tune(host_path, donor_path):
    """Leakage-guarded instrument transplant between two dumps; returns ``(new_ow, record)`` or
    ``(None, None)`` when no safe role-matched pair exists."""
    provenance.guard_train_split(host_path, donor_path)
    host_ow, donor_ow = writes.load_ow(host_path), writes.load_ow(donor_path)
    pair = pick_pair(host_ow, donor_ow)
    if pair is None:
        return None, None
    hv, dv = pair
    new_ow, info = instrument_transplant(host_ow, donor_ow, hv, dv)
    if new_ow is None:
        return None, None
    rec = provenance.record(
        f"{_stem(host_path)}__inst{hv}_from_{_stem(donor_path)}v{dv}.dump.parquet",
        host=host_path,
        transform="instrument",
        voice=hv,
        donors=[donor_path],
        anchor=info.get("donor_voice"),
    )
    return new_ow, rec


def _stem(path) -> str:
    name = str(path).split("/")[-1]
    return name[: -len(".dump.parquet")] if name.endswith(".dump.parquet") else name
