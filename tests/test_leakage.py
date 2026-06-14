"""Leakage is a hard rule: donors/hosts must be train-split only; a cross-split donor must raise."""

import pytest

from preframr_aug import provenance


def test_split_of_classifies_paths():
    assert provenance.split_of("/c/train/DRAX/A.1.dump.parquet") == "train"
    assert provenance.split_of("/c/eval_a/Goto80/B.1.dump.parquet") == "eval_a"
    assert provenance.split_of("/c/eval_b_daglish/X/C.1.dump.parquet") == "eval_b"
    assert provenance.split_of("/flat/dir/D.1.dump.parquet") == "unknown"


def test_guard_allows_train_and_flat():
    provenance.guard_train_split("/c/train/A/a.dump.parquet", "/flat/b.dump.parquet")


def test_guard_rejects_eval_donor():
    with pytest.raises(provenance.LeakageError):
        provenance.guard_train_split(
            "/c/train/A/host.dump.parquet",
            "/c/eval_a/Goto80/donor.dump.parquet",
        )


def test_strict_allow_rejects_unknown():
    with pytest.raises(provenance.LeakageError):
        provenance.guard_train_split("/flat/x.dump.parquet", allow=("train",))


def test_record_carries_provenance(tmp_path):
    rec = provenance.record(
        tmp_path / "aug_0001.dump.parquet",
        host="/c/train/A/host.dump.parquet",
        transform="instrument",
        voice=1,
        donors=["/c/train/B/donor.dump.parquet"],
        anchor=128,
    )
    assert rec["transform"] == "instrument" and rec["host_split"] == "train"
    assert rec["donors"] == ["/c/train/B/donor.dump.parquet"]
    out = tmp_path / "provenance.jsonl"
    provenance.write_jsonl([rec], out)
    assert out.read_text().strip().startswith("{")
