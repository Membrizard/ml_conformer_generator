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
        """
        Helper for streaming download
        """
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
        """
        Resolve the path to a required weights file
        """

        # check script folder
        p = Path(filename)
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
                cache_path = Path(hf_hub_download(HF_REPO, filename, local_dir=self.cache_dir))
            except Exception as e:
                logger.error(f"{PREFIX} Download failed due to: {e}")
                raise ValueError(f"Could not resolve path to the weights file: {filename}")

            logger.info(f"{PREFIX} Download complete")

        return cache_path

    def clear_cache(self) -> None:
        """
        Clear weight cache directory
        """
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{PREFIX} Weights cache cleared: %s", self.cache_dir)

    def list_available_weights(self, suffixes: set = {".pt", ".onnx"}) -> dict[str, list[str]]:
        """
        List all available weights on remote and local
        """
        out = {"remote":[], "local": []}

        _ensure_hf_hub()

        from huggingface_hub import list_repo_files

        out['remote'] = [
            f for f in list_repo_files(HF_REPO)
            if Path(f).suffix.lower() in suffixes
        ]

        cache_dir = Path(MLCONFGEN_CACHE)

        if cache_dir.exists():
            out['local'] = [
                str(f.relative_to(cache_dir))
                for f in cache_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in suffixes
            ]

        return out
