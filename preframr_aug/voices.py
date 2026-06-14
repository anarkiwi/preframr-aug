"""Voice role hints and register-level wiring analysis. Roles (bass/lead/perc) drive donor<->host
matching; wiring (sync/ring/filter routing) drives the hard transplant exclusions -- a modulator and
its carrier are one musical unit and must never be separated, and filter routing stays with the host.
"""

from __future__ import annotations

GATE = 0x01
SYNC = 0x02
RING = 0x04
NOISE = 0x80
_WAVE_MASK = 0xF0
VOICES = (0, 1, 2)


def _settled_per_voice(ow):
    """Per voice, the settled ``(freq, ctrl)`` at each frame it is touched -- the raw material for the
    role and wiring heuristics. Freq is the 16-bit lo/hi pair; ctrl is reg+4."""
    freq = {v: [0, 0] for v in VOICES}
    ctrl = {v: 0 for v in VOICES}
    seen = {v: [] for v in VOICES}
    cur_frame = None
    for frame, reg, val in ow.triples():
        if frame != cur_frame and cur_frame is not None:
            for v in VOICES:
                seen[v].append(((freq[v][1] << 8) | freq[v][0], ctrl[v]))
        cur_frame = frame
        v = reg // 7
        if v in VOICES:
            off = reg % 7
            if off == 0:
                freq[v][0] = val
            elif off == 1:
                freq[v][1] = val
            elif off == 4:
                ctrl[v] = val
    if cur_frame is not None:
        for v in VOICES:
            seen[v].append(((freq[v][1] << 8) | freq[v][0], ctrl[v]))
    return seen


def roles(ow) -> dict:
    """Heuristic per-voice role: ``perc`` if noise-dominant while gated, else ``bass``/``lead`` for the
    lowest/highest median gated pitch, ``unknown`` otherwise. Mislabels only weaken an augmentation,
    never invalidate it."""
    seen = _settled_per_voice(ow)
    medians = {}
    out = {v: "unknown" for v in VOICES}
    for v in VOICES:
        gated = [(f, c) for f, c in seen[v] if c & GATE and f > 8]
        if len(gated) < 4:
            continue
        noise_frac = sum(1 for _, c in gated if c & _WAVE_MASK & NOISE) / len(gated)
        if noise_frac > 0.5:
            out[v] = "perc"
            continue
        pitched = sorted(f for f, _ in gated)
        medians[v] = pitched[len(pitched) // 2]
    if medians:
        lo = min(medians, key=medians.get)
        hi = max(medians, key=medians.get)
        out[lo] = "bass"
        if hi != lo:
            out[hi] = "lead"
    return out


def wiring(ow) -> dict:
    """Per-voice ``{sync, ring, filter_routed}`` read from the registers: sync/ring = ctrl bits 1/2
    (ever set); filter_routed = reg-23 routing bit for the voice (ever set)."""
    seen = _settled_per_voice(ow)
    route = 0
    for _, reg, val in ow.triples():
        if reg == 23:
            route |= val
    return {
        v: {
            "sync": any(c & SYNC for _, c in seen[v]),
            "ring": any(c & RING for _, c in seen[v]),
            "filter_routed": bool(route & (1 << v)),
        }
        for v in VOICES
    }


def transplantable(ow) -> set:
    """Voices safe to use as transplant donor or host: exclude any voice with sync/ring set AND its
    modulation source (SID voice v is sync/ring-driven by voice (v-1) mod 3) -- the edge carries the
    music, never separate a modulator from its carrier."""
    wired = wiring(ow)
    excluded = set()
    for v in VOICES:
        if wired[v]["sync"] or wired[v]["ring"]:
            excluded.add(v)
            excluded.add((v - 1) % 3)
    return set(VOICES) - excluded
