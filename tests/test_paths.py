"""Path configuration resolves from environment variables without side effects."""
import importlib


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("FRACTALSIG_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("FRACTALSIG_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("GAGLE_MANIFEST_DIR", raising=False)
    monkeypatch.delenv("FRACTALSIG_STATS_DIR", raising=False)
    monkeypatch.delenv("FRACTALSIG_GAGLE_ZIP", raising=False)
    import fractalsig.paths as paths
    paths = importlib.reload(paths)
    assert paths.GAGLE_ZIP == (tmp_path / "ws" / "gagle.zip").resolve()
    assert paths.MANIFEST_SHARD_DIR == (tmp_path / "ws" / "gpu_manifest_shards").resolve()
    assert paths.STATS_DIR == (tmp_path / "out" / "stats").resolve()
    assert not (tmp_path / "ws").exists()
    paths.ensure_dirs()
    assert paths.DATA_DIR.is_dir() and paths.FIGURES_DIR.is_dir()


def test_reference_outputs_present():
    import fractalsig.paths as paths
    paths = importlib.reload(paths)
    assert len(list(paths.REFERENCE_STATS_DIR.glob("*.json"))) >= 30
    assert len(list(paths.REFERENCE_FIGURES_DIR.glob("*.png"))) == 10


def test_worker_env(monkeypatch):
    from fractalsig.runtime import default_workers
    monkeypatch.setenv("FRACTALSIG_WORKERS", "3")
    assert default_workers() == 3
    monkeypatch.delenv("FRACTALSIG_WORKERS")
    assert default_workers() >= 1
