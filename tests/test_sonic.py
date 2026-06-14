"""Sonic plausibility band: in-distribution fingerprints pass, a wholesale outlier is rejected."""

import numpy as np

from preframr_aug import sonic


def test_band_accepts_inliers_rejects_outlier():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(50, 16))
    lo, hi = sonic.band(ref, k=3.0)
    assert sonic.in_band(ref[0], (lo, hi)), "a corpus member is in-band"
    assert not sonic.in_band(
        np.full(16, 100.0), (lo, hi)
    ), "a wholesale outlier is rejected"


def test_in_band_tolerates_few_outlier_dims():
    lo = np.zeros(20)
    hi = np.ones(20)
    fp = np.full(20, 0.5)
    fp[0] = 5.0
    assert sonic.in_band(fp, (lo, hi), tol=0.1), "one outlier dim of 20 is tolerated"
    fp[:5] = 5.0
    assert not sonic.in_band(fp, (lo, hi), tol=0.1), "5/20 outlier dims rejected"
