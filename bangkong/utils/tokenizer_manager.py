"""
TokenizerManager for Bangkong LLM Training System.

Handles GPT-2 tokenizer resolution with the following priority chain:
  1. Offline cache (assets/tokenizer/gpt2/) -- always preferred.
  2. HuggingFace Hub download -- executed when offline cache is absent.
  3. Explicit error -- raised only when both sources are unavailable.

This module guarantees that the tokenizer is always saved as a
tokenizer.json (fast tokenizer format) so that Kaggle / Colab
environments, which may not merge vocab.json + merges.txt correctly
when using the legacy slow tokenizer, are unaffected.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from transformers import GPT2TokenizerFast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HF model identifier for the base GPT-2 tokenizer.
_HF_GPT2_MODEL_ID = "gpt2"

# Path (relative to the repository root) where the offline tokenizer is stored.
# The value is resolved to an absolute path at runtime via _resolve_asset_dir().
_RELATIVE_OFFLINE_PATH = Path("assets") / "tokenizer" / "gpt2"

# Maximum download retries on HTTP 429 (rate-limit) responses.
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_asset_dir() -> Path:
    """
    Resolve the absolute path of the offline tokenizer directory.

    The function walks upward from this file until it finds the repository
    root (identified by the presence of pyproject.toml or .git), then
    appends the relative path defined in _RELATIVE_OFFLINE_PATH.

    Returns:
        Absolute Path to the offline tokenizer directory.

    Raises:
        RuntimeError: If the repository root cannot be determined.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Limit upward traversal to 10 levels.
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current / _RELATIVE_OFFLINE_PATH
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        "Cannot determine repository root. "
        "Expected to find pyproject.toml or .git in an ancestor directory."
    )


def _is_valid_offline_cache(path: Path) -> bool:
    """
    Return True when *path* contains a complete fast tokenizer.

    A valid cache must contain tokenizer.json. The presence of vocab.json
    and merges.txt alone (slow tokenizer artefacts) is explicitly treated
    as invalid because those files are not merged into tokenizer.json,
    which is exactly the problem being fixed.

    Args:
        path: Directory to inspect.

    Returns:
        True when tokenizer.json exists inside *path*.
    """
    return path.is_dir() and (path / "tokenizer.json").exists()


def _download_from_hf(model_id: str, save_path: Path) -> GPT2TokenizerFast:
    """
    Download the GPT-2 fast tokenizer from HuggingFace Hub and persist it.

    Implements exponential back-off on HTTP 429 responses.

    Args:
        model_id: HuggingFace model identifier (e.g. "gpt2").
        save_path: Local directory where the tokenizer will be saved.

    Returns:
        Loaded GPT2TokenizerFast instance.

    Raises:
        RuntimeError: When all retries are exhausted or a non-rate-limit
            error is encountered.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            logger.info(
                "Downloading GPT-2 fast tokenizer from HuggingFace Hub "
                "(model_id=%s, attempt=%d/%d).",
                model_id,
                attempt + 1,
                _MAX_RETRIES,
            )
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            save_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(save_path))
            logger.info(
                "Tokenizer downloaded and saved to offline cache: %s", save_path
            )
            return tokenizer

        except Exception as exc:
            last_exception = exc
            is_rate_limit = _is_rate_limit_error(exc)

            if not is_rate_limit:
                raise RuntimeError(
                    f"Failed to download tokenizer from HuggingFace Hub: {exc}"
                ) from exc

            if attempt == _MAX_RETRIES - 1:
                break

            delay = _BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "HTTP 429 rate-limit hit. Retrying in %d second(s) "
                "(attempt %d/%d).",
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Exhausted {_MAX_RETRIES} retries downloading tokenizer from "
        f"HuggingFace Hub. Last error: {last_exception}"
    ) from last_exception


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Determine whether *exc* represents an HTTP 429 (rate-limit) error.

    Args:
        exc: Exception to inspect.

    Returns:
        True when the exception signals a rate-limit response.
    """
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code == 429
    exc_str = str(exc)
    return "429" in exc_str and (
        "Too Many Requests" in exc_str or "rate limit" in exc_str.lower()
    )


def _load_from_offline_cache(cache_path: Path) -> GPT2TokenizerFast:
    """
    Load the GPT-2 fast tokenizer from the offline cache directory.

    Args:
        cache_path: Absolute path to the offline cache directory.

    Returns:
        Loaded GPT2TokenizerFast instance.

    Raises:
        RuntimeError: When loading from the cache fails.
    """
    try:
        logger.info("Loading tokenizer from offline cache: %s", cache_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(str(cache_path))
        logger.info("Tokenizer loaded successfully from offline cache.")
        return tokenizer
    except Exception as exc:
        raise RuntimeError(
            f"Offline tokenizer cache at '{cache_path}' is present but "
            f"could not be loaded: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TokenizerManager:
    """
    Centralised tokenizer resolver for the Bangkong training system.

    Resolution order:
        1. Offline cache (assets/tokenizer/gpt2/).
        2. HuggingFace Hub download (saves result to offline cache).
        3. RuntimeError if both are unavailable.

    The manager always uses GPT2TokenizerFast, which serialises to
    tokenizer.json, eliminating the vocab.json / merges.txt merge issue
    that occurs on Kaggle / Colab with the legacy slow tokenizer.

    Usage:
        manager = TokenizerManager()
        tokenizer = manager.load()
        tokenizer.pad_token = tokenizer.eos_token   # caller's responsibility
    """

    def __init__(self, offline_cache_dir: Optional[Path] = None):
        """
        Initialise the manager.

        Args:
            offline_cache_dir: Override the default offline cache directory.
                When None, the directory is resolved automatically relative
                to the repository root.
        """
        if offline_cache_dir is not None:
            self._cache_dir = Path(offline_cache_dir).resolve()
        else:
            self._cache_dir = _resolve_asset_dir()

    @property
    def cache_dir(self) -> Path:
        """Absolute path to the offline tokenizer cache directory."""
        return self._cache_dir

    def is_cached(self) -> bool:
        """
        Return True when a valid offline cache exists.

        Returns:
            True when tokenizer.json is present in the cache directory.
        """
        return _is_valid_offline_cache(self._cache_dir)

    def download_and_cache(self) -> GPT2TokenizerFast:
        """
        Force a fresh download from HuggingFace Hub and update the cache.

        Any existing cache content is replaced.

        Returns:
            Freshly downloaded GPT2TokenizerFast instance.

        Raises:
            RuntimeError: When the download fails after all retries.
        """
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            logger.info("Removed stale offline tokenizer cache: %s", self._cache_dir)

        return _download_from_hf(_HF_GPT2_MODEL_ID, self._cache_dir)

    def load(self) -> GPT2TokenizerFast:
        """
        Load the GPT-2 fast tokenizer using the offline-first strategy.

        Priority:
            1. Offline cache — used when tokenizer.json is present.
            2. HuggingFace download — executed when cache is absent, result
               is persisted to cache for future use.

        Returns:
            GPT2TokenizerFast instance ready for use.

        Raises:
            RuntimeError: When both offline cache and HF download fail.
        """
        if self.is_cached():
            return _load_from_offline_cache(self._cache_dir)

        logger.info(
            "Offline tokenizer cache not found at '%s'. "
            "Attempting download from HuggingFace Hub.",
            self._cache_dir,
        )
        return _download_from_hf(_HF_GPT2_MODEL_ID, self._cache_dir)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def load_gpt2_tokenizer(
    offline_cache_dir: Optional[Path] = None,
) -> GPT2TokenizerFast:
    """
    Load the GPT-2 fast tokenizer using the offline-first strategy.

    This is the primary entry point intended for use throughout the
    Bangkong codebase. Callers should set pad_token after this call:

        tokenizer = load_gpt2_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    Args:
        offline_cache_dir: Optional override for the offline cache path.

    Returns:
        GPT2TokenizerFast instance.

    Raises:
        RuntimeError: When both offline cache and HF download fail.
    """
    manager = TokenizerManager(offline_cache_dir=offline_cache_dir)
    return manager.load()
