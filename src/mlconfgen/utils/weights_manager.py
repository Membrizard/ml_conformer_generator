from __future__ import annotations
import subprocess
import sys
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from .config import HF_REPO, MLCONFGEN_CACHE

logger = logging.getLogger(__name__)
PREFIX = '[MLConfGen Weights Manager]'



def _ensure_hf_hub():
    try:
        import huggingface_hub 
        return
    except ImportError:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "huggingface_hub"],
        stdout=sys.stderr,
    )
    import huggingface_hub 


class WeightsManager:
    """
    A Class to manage weights for MLConfGen models
    """
    def __init__(
        self,
        cache_dir: str | Path | None = None,
    ):
        
        self.cache_dir = Path(
            cache_dir
            or MLCONFGEN_CACHE
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _file_lock(self, lock_path: Path, timeout=60):
        start = time.time()
        while lock_path.exists():
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"{PREFIX} Timeout waiting for lock {lock_path}"
                )
            time.sleep(0.2)

        try:
            lock_path.touch()
            yield
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def resolve(self, filename: str, force_download: bool = False) -> Path:

        p = Path("adj_mat_seer.pt")
        if p.exists():
            return p

        cache_path = self.cache_dir / filename
        lock_path = cache_path.with_suffix(".lock")

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not force_download:
            logger.info(f"{PREFIX} Using cached weights")
            return cache_path

        with self._file_lock(lock_path):
            if cache_path.exists() and not force_download:
                return cache_path

            logger.info(f"{PREFIX} Downloading weights ...")

            try:
                _ensure_hf_hub()
                from huggingface_hub import hf_hub_download
                cache_path = Path(hf_hub_download(HF_REPO, filename, cache_dir=self.cache_dir))
            except Exception as e:
                logger.error(f"{PREFIX} Download failed due to: {e}")

            logger.info(f"{PREFIX} Download complete")

        return cache_path

    def clear_cache(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{PREFIX} Weights cache cleared: %s", self.cache_dir)

    def list_available_weights(self) -> list[str]: 

        suffixes = [".pt",  ".onnx"]
        _ensure_hf_hub()
        from huggingface_hub import list_repo_files
        return sorted(
            f for f in list_repo_files(HF_REPO)
            if Path(f).suffix in suffixes
    )



# def resolve_file(
#     filename: str,
#     *,
#     search_dirs: tuple[Path, ...] = (Path("."),),
#     cache_dir: str | Path | None = None,
#     download: bool = True,
# ) -> Path:
#     """cwd (and search_dirs) first, then Hub cache / download."""
#     for d in search_dirs:
#         p = d / filename
#         if p.is_file():
#             return p.resolve()
#     if not download:
#         raise FileNotFoundError(filename)
#     _ensure_hf_hub()
#     from huggingface_hub import hf_hub_download
#     return Path(hf_hub_download(HF_REPO, filename, cache_dir=cache_dir))














# import subprocess
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Literal

# from .config import HF_REPO

# Backend = Literal["torch", "onnx"]

# def _ensure_hf_hub():
#     try:
#         import huggingface_hub 
#         return
#     except ImportError:
#         pass
#     subprocess.check_call(
#         [sys.executable, "-m", "pip", "install", "huggingface_hub"],
#         stdout=sys.stderr,
#     )
#     import huggingface_hub 


# def list_available_weights(backend: Backend | None = None, *, from_hub: bool = True) -> list[WeightFile]:
#     """Get available files for HuggingFace. `from_hub=False` is offline (catalog only)."""
#     known = [w for w in CATALOG if backend is None or w.backend == backend]
#     if not from_hub:
#         return known
#     _ensure_hf_hub()
#     from huggingface_hub import list_repo_files
#     remote = set(list_repo_files(HF_REPO))
#     listed = [w for w in known if w.filename in remote]
#     extra = sorted(
#         f for f in remote
#         if f.endswith((".pt", ".onnx")) and f not in {w.filename for w in known}
#     )
#     # extras are listed but not auto-loaded — constructors still assume 420 / 2048
#     return listed, extra