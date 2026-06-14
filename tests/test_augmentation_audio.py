"""Audio augmentation fixture test: fetch real HVSC tunes on demand (cached, never committed), apply
the reduce and instrument arms, and render the original plus each augmentation to WAVs under
``$PREFRAMR_AUG_WAV_OUT`` so the result is audible. Asserts every render is non-silent and audibly
differs from the original; skips cleanly when the Docker fixture can't be built."""

import os
from pathlib import Path

import numpy as np
import pytest

from preframr_aug import provenance, reduce, transplant, voices, writes
from tests.audio_render import render_dump_to_wav, render_ow_to_wav
from tests.sid_fixtures import (
    CAMEROCK,
    GRID_RUNNER,
    TRAP,
    FixtureUnavailable,
    cache_dir,
    ensure_dump,
)

REDUCE_PREFIX_FRAMES = 300
SILENCE_RMS = 5.0


def _wav_out_dir() -> Path:
    base = os.environ.get("PREFRAMR_AUG_WAV_OUT")
    out = Path(base) if base else cache_dir() / "wav"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ensure(spec) -> Path:
    try:
        return ensure_dump(spec)
    except FixtureUnavailable as err:
        pytest.skip(str(err))


def _read_wav(path: Path):
    from scipy.io import wavfile  # pylint: disable=import-outside-toplevel

    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    return int(rate), data.astype(np.float64)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def _assert_audible(orig_path: Path, aug_path: Path, label: str) -> None:
    """Both renders are long enough, non-silent, and audibly distinct (mean abs difference over the
    common span is a non-trivial fraction of the original's own scale)."""
    rate, orig = _read_wav(orig_path)
    _, aug = _read_wav(aug_path)
    assert aug.size > rate, f"{label}: augmented render shorter than 1s"
    assert _rms(orig) > SILENCE_RMS, f"{label}: original render is silent"
    assert _rms(aug) > SILENCE_RMS, f"{label}: augmented render is silent"
    span = min(orig.size, aug.size)
    diff = float(np.mean(np.abs(orig[:span] - aug[:span])))
    assert diff > 0.05 * (_rms(orig[:span]) + 1.0), f"{label}: render matches original"


@pytest.fixture(scope="module")
def reduce_dump() -> Path:
    return _ensure(GRID_RUNNER)


INSTRUMENT_PAIRINGS = [
    (TRAP, CAMEROCK, "bass"),
    (CAMEROCK, TRAP, "lead"),
]


def _role_safe_voice(ow, role: str):
    """A transplant-safe voice playing ``role`` (bass/lead), or None."""
    role_map = voices.roles(ow)
    safe = voices.transplantable(ow)
    return next((v for v in sorted(safe) if role_map[v] == role), None)


def test_reduce_augmentation_is_audible(reduce_dump):
    ow = writes.load_ow(reduce_dump)
    reduced, info = reduce.reduce_prefix(ow, prefix_frames=REDUCE_PREFIX_FRAMES)
    assert reduced is not None, "reduce found no melodic target in the fixture"
    out = _wav_out_dir()
    orig_wav = out / f"{GRID_RUNNER.slug}_original.wav"
    aug_wav = out / f"{GRID_RUNNER.slug}_reduce.wav"
    render_dump_to_wav(reduce_dump, orig_wav)
    render_ow_to_wav(reduced, aug_wav)
    _assert_audible(orig_wav, aug_wav, "reduce")
    print(
        f"\nreduce target_voice={info['target_voice']} kept={info.get('kept_voices')}"
    )
    print(f"original: {orig_wav}")
    print(f"reduce:   {aug_wav}")


@pytest.mark.parametrize(
    "host_spec, donor_spec, role",
    INSTRUMENT_PAIRINGS,
    ids=[f"{h.slug}-{r}" for h, _, r in INSTRUMENT_PAIRINGS],
)
def test_instrument_augmentation_is_audible(host_spec, donor_spec, role):
    """Same-role cross-tune transplant: the host voice keeps its line but adopts a same-role donor
    instrument whose envelope timescale fits, so the result is audibly re-voiced yet musical (bass<-bass
    is a subtle tonal shift, lead<-lead more distinct). The leakage guard runs as in the real pipeline;
    the corpus-scale sonic band gate is unreliable at fixture scale and left to the CLI --filter flag.
    """
    host_dump = _ensure(host_spec)
    donor_dump = _ensure(donor_spec)
    host = writes.load_ow(host_dump)
    donor = writes.load_ow(donor_dump)
    host_voice = _role_safe_voice(host, role)
    donor_voice = _role_safe_voice(donor, role)
    if host_voice is None or donor_voice is None:
        pytest.skip(
            f"no transplant-safe {role} voice in {host_spec.slug}/{donor_spec.slug}"
        )
    provenance.guard_train_split(str(host_dump), str(donor_dump))
    new_ow, info = transplant.instrument_transplant(
        host, donor, host_voice, donor_voice
    )
    if new_ow is None:
        pytest.skip("no onset program available for the chosen voice pair")
    out = _wav_out_dir()
    orig_wav = out / f"{host_spec.slug}_original.wav"
    aug_wav = out / f"{host_spec.slug}_{role}_instrument.wav"
    render_dump_to_wav(host_dump, orig_wav)
    render_ow_to_wav(new_ow, aug_wav)
    _assert_audible(orig_wav, aug_wav, f"instrument:{role}")
    print(
        f"\ninstrument {host_spec.slug} v{info['host_voice']} "
        f"<- {donor_spec.slug} v{info['donor_voice']} ({role})"
    )
    print(f"original:   {orig_wav}")
    print(f"instrument: {aug_wav}")
