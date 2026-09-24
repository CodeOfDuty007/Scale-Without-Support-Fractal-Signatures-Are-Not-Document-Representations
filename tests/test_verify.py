"""Tests for the reference-output comparator."""
import json

from fractalsig.verify import compare_file, main


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_identical_files_pass(tmp_path):
    obj = {"a": 1.0, "b": [1, 2, {"c": 0.5}], "runtime_sec": 10}
    r = _write(tmp_path / "r.json", obj)
    n = _write(tmp_path / "n.json", {**obj, "runtime_sec": 99})
    rep = compare_file(r, n, 1e-6, 1e-9, strict=True)
    assert rep.ok and rep.n_numeric == 4


def test_numeric_drift_fails(tmp_path):
    r = _write(tmp_path / "r.json", {"x": 0.6000})
    n = _write(tmp_path / "n.json", {"x": 0.6010})
    assert not compare_file(r, n, 1e-6, 1e-9, strict=False).ok


def test_structure_change_fails(tmp_path):
    r = _write(tmp_path / "r.json", {"x": [1, 2, 3]})
    n = _write(tmp_path / "n.json", {"x": [1, 2]})
    assert not compare_file(r, n, 1e-6, 1e-9, strict=False).ok


def test_statistical_profile(tmp_path, monkeypatch):
    import fractalsig.verify as V
    monkeypatch.setitem(V.PROFILES, "t2_step1_noise_floor.json", [
        (r"^(S_hat_mean|S_bias)$", ("z", "S_hat_sd", "n_trials", 3.0)),
        (r"^(S_hat_sd|S_rmse|S_r2_mean)$", ("atol", 0.15))])
    ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"
    ref_dir.mkdir(); new_dir.mkdir()
    row = {"H_hat_mean": 0.67, "S_hat_mean": 0.20, "S_bias": -0.45, "S_hat_sd": 0.28}
    ref = {"n_trials": 40, "results": [row]}
    _write(ref_dir / "t2_step1_noise_floor.json", ref)
    ok = {"n_trials": 40, "results": [{**row, "S_hat_mean": 0.21, "S_bias": -0.44}]}
    _write(new_dir / "t2_step1_noise_floor.json", ok)
    assert main(["--reference-dir", str(ref_dir), "--new-dir", str(new_dir)]) == 0
    bad = {"n_trials": 40, "results": [{**row, "H_hat_mean": 0.68}]}
    _write(new_dir / "t2_step1_noise_floor.json", bad)
    assert main(["--reference-dir", str(ref_dir), "--new-dir", str(new_dir)]) == 1


def test_nothing_to_compare(tmp_path):
    assert main(["--new-dir", str(tmp_path)]) == 2
