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


@pytest.fixture(scope="module")
def instrument_host() -> Path:
    return _ensure(TRAP)


@pytest.fixture(scope="module")
def instrument_donor() -> Path:
    return _ensure(CAMEROCK)


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


def test_instrument_augmentation_is_audible(instrument_host, instrument_donor):
    """Same-role cross-tune transplant: the host bass keeps its line but adopts a donor bass instrument
    whose envelope timescale fits, so the result is audibly re-voiced yet musical. The leakage guard runs
    as in the real pipeline; the sonic band gate is corpus-scale (unreliable at fixture scale) so it is
    left to production via the CLI --filter flag."""
    host = writes.load_ow(instrument_host)
    donor = writes.load_ow(instrument_donor)
    host_voice = _role_safe_voice(host, "bass")
    donor_voice = _role_safe_voice(donor, "bass")
    if host_voice is None or donor_voice is None:
        pytest.skip("no transplant-safe bass voice in host/donor")
    provenance.guard_train_split(str(instrument_host), str(instrument_donor))
    new_ow, info = transplant.instrument_transplant(
        host, donor, host_voice, donor_voice
    )
    if new_ow is None:
        pytest.skip("no onset program available for the chosen voice pair")
    out = _wav_out_dir()
    orig_wav = out / f"{TRAP.slug}_original.wav"
    aug_wav = out / f"{TRAP.slug}_instrument.wav"
    render_dump_to_wav(instrument_host, orig_wav)
    render_ow_to_wav(new_ow, aug_wav)
    _assert_audible(orig_wav, aug_wav, "instrument")
    print(
        f"\ninstrument host={TRAP.slug} v{info['host_voice']} "
        f"<- donor={CAMEROCK.slug} v{info['donor_voice']}"
    )
    print(f"original:   {orig_wav}")
    print(f"instrument: {aug_wav}")
