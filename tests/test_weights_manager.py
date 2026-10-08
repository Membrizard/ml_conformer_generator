import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.mlconfgen import MLConformerGenerator, MLConformerGeneratorONNX
from src.mlconfgen.utils.weights_manager import WeightsManager

REPO = Path(__file__).resolve().parents[1]

TORCH_WEIGHTS = {
    "edm_moi_chembl_15_39.pt": REPO / "edm_moi_chembl_15_39.pt",
    "adj_mat_seer_chembl_15_39.pt": REPO / "adj_mat_seer_chembl_15_39.pt",
}
ONNX_WEIGHTS = {
    "egnn_chembl_15_39.onnx": REPO / "egnn_chembl_15_39.onnx",
    "adj_mat_seer_chembl_15_39.onnx": REPO / "adj_mat_seer_chembl_15_39.onnx",
}


@pytest.fixture
def wm(tmp_path):
    return WeightsManager(cache_dir=tmp_path / "cache")


@pytest.fixture
def fake_hf(monkeypatch):
    """Stub huggingface_hub so tests never touch the network or pip."""
    calls = {"download": [], "list": []}

    def hf_hub_download(repo, filename, local_dir):
        calls["download"].append((repo, filename, local_dir))
        dest = Path(local_dir) / filename
        dest.write_bytes(b"remote")
        return str(dest)

    def list_repo_files(repo):
        calls["list"].append(repo)
        return []

    mod = ModuleType("huggingface_hub")
    mod.hf_hub_download = hf_hub_download
    mod.list_repo_files = list_repo_files
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setattr(
        "src.mlconfgen.utils.weights_manager._ensure_hf_hub", lambda: None
    )
    return calls, mod


@pytest.fixture
def hf_hub_from_local(tmp_path, monkeypatch):
    """Route WeightsManager HF downloads to local weight files via a temp cache."""
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    monkeypatch.setattr(
        "src.mlconfgen.utils.weights_manager.MLCONFGEN_CACHE", cache
    )
    # default filenames must not resolve as local paths in the repo cwd
    monkeypatch.chdir(tmp_path)

    sources = {**TORCH_WEIGHTS, **ONNX_WEIGHTS}
    calls = []

    def hf_hub_download(repo, filename, local_dir):
        if filename not in sources or not sources[filename].exists():
            raise FileNotFoundError(filename)
        calls.append(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sources[filename], dest)
        return str(dest)

    mod = ModuleType("huggingface_hub")
    mod.hf_hub_download = hf_hub_download
    mod.list_repo_files = lambda repo: list(sources)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setattr(
        "src.mlconfgen.utils.weights_manager._ensure_hf_hub", lambda: None
    )
    return calls, cache

def test_init_creates_cache_dir(tmp_path):
    cache = tmp_path / "weights"
    wm = WeightsManager(cache_dir=cache)
    assert cache.is_dir()
    assert wm.cache_dir == cache


def test_resolve_existing_local_path(wm, tmp_path):
    local = tmp_path / "local_weights.pt"
    local.write_bytes(b"local")
    assert wm.resolve(str(local)) == local


def test_resolve_uses_cache(wm, fake_hf):
    calls, _ = fake_hf
    cached = wm.cache_dir / "model.pt"
    cached.write_bytes(b"cached")
    out = wm.resolve("model.pt")
    assert out == cached
    assert calls["download"] == []


def test_resolve_downloads_when_missing(wm, fake_hf):
    calls, _ = fake_hf
    out = wm.resolve("remote.pt")
    assert out == wm.cache_dir / "remote.pt"
    assert out.read_bytes() == b"remote"
    assert len(calls["download"]) == 1


def test_resolve_force_download(wm, fake_hf):
    calls, mod = fake_hf
    (wm.cache_dir / "model.pt").write_bytes(b"old")

    def hf_hub_download(repo, filename, local_dir):
        calls["download"].append((repo, filename, local_dir))
        dest = Path(local_dir) / filename
        dest.write_bytes(b"new")
        return str(dest)

    mod.hf_hub_download = hf_hub_download
    out = wm.resolve("model.pt", force_download=True)
    assert out.read_bytes() == b"new"
    assert len(calls["download"]) == 1


def test_resolve_download_failure(wm, fake_hf):
    _, mod = fake_hf

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    mod.hf_hub_download = boom
    with pytest.raises(ValueError, match="Could not resolve"):
        wm.resolve("missing.pt")


def test_clear_cache(wm):
    f = wm.cache_dir / "model.pt"
    f.write_bytes(b"x")
    wm.clear_cache()
    assert wm.cache_dir.is_dir()
    assert not f.exists()
    assert list(wm.cache_dir.iterdir()) == []


def test_list_available_weights(wm, fake_hf, monkeypatch):
    _, mod = fake_hf
    monkeypatch.setattr(
        "src.mlconfgen.utils.weights_manager.MLCONFGEN_CACHE", wm.cache_dir
    )
    (wm.cache_dir / "local.pt").write_bytes(b"x")
    (wm.cache_dir / "local.onnx").write_bytes(b"x")
    (wm.cache_dir / "notes.txt").write_bytes(b"x")
    mod.list_repo_files = lambda repo: ["a.pt", "b.onnx", "readme.md", "c.PT"]

    out = wm.list_available_weights()
    assert set(out["remote"]) == {"a.pt", "b.onnx", "c.PT"}
    assert set(out["local"]) == {"local.pt", "local.onnx"}


def test_list_available_weights_suffix_filter(wm, fake_hf, monkeypatch):
    _, mod = fake_hf
    monkeypatch.setattr(
        "src.mlconfgen.utils.weights_manager.MLCONFGEN_CACHE", wm.cache_dir
    )
    (wm.cache_dir / "local.pt").write_bytes(b"x")
    mod.list_repo_files = lambda repo: ["a.pt", "b.onnx"]

    out = wm.list_available_weights(suffixes={".onnx"})
    assert out["remote"] == ["b.onnx"]
    assert out["local"] == []


def test_file_lock_timeout(wm):
    lock = wm.cache_dir / "model.lock"
    lock.touch()
    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        with wm._file_lock(lock, timeout=0.3):
            pass


def test_file_lock_releases(wm):
    lock = wm.cache_dir / "model.lock"
    with wm._file_lock(lock, timeout=1):
        assert lock.exists()
    assert not lock.exists()


# --- model init via WeightsManager / HF hub path ---


@pytest.mark.slow
@pytest.mark.skipif(
    not all(p.exists() for p in TORCH_WEIGHTS.values()),
    reason="torch weight files not present",
)
def test_torch_generator_init_from_hf(hf_hub_from_local):

    calls, cache = hf_hub_from_local
    gen = MLConformerGenerator(
        diffusion_steps=10,
        device="cpu",
    )
    assert gen.generative_model is not None
    assert gen.adj_mat_seer is not None
    assert set(calls) == set(TORCH_WEIGHTS)
    assert (cache / "edm_moi_chembl_15_39.pt").exists()
    assert (cache / "adj_mat_seer_chembl_15_39.pt").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not all(p.exists() for p in ONNX_WEIGHTS.values()),
    reason="onnx weight files not present",
)
def test_onnx_generator_init_from_hf(hf_hub_from_local):
    calls, cache = hf_hub_from_local
    gen = MLConformerGeneratorONNX(diffusion_steps=10)
    assert gen.generative_model is not None
    assert gen.adj_mat_seer is not None
    assert set(calls) == set(ONNX_WEIGHTS)
    assert (cache / "egnn_chembl_15_39.onnx").exists()
    assert (cache / "adj_mat_seer_chembl_15_39.onnx").exists()
