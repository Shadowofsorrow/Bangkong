#!/usr/bin/env python3
"""
Script to download and cache the GPT-2 fast tokenizer from HuggingFace Hub.

Run this script once before training to populate the offline tokenizer cache.
After running, the training pipeline will operate fully offline.

Usage:
    python scripts/download_tokenizer.py
    python scripts/download_tokenizer.py --cache-dir /custom/path/to/tokenizer
    python scripts/download_tokenizer.py --force   # overwrite existing cache
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repository root is on sys.path regardless of where this script
# is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from bangkong.utils.tokenizer_manager import TokenizerManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bangkong.download_tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache the GPT-2 fast tokenizer for offline use."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Override the default offline cache directory "
            "(default: assets/tokenizer/gpt2/ inside the repository root)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even when a valid cache already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = TokenizerManager(offline_cache_dir=args.cache_dir)

    logger.info("Offline tokenizer cache directory: %s", manager.cache_dir)

    if manager.is_cached() and not args.force:
        logger.info(
            "Valid offline cache already exists. "
            "Use --force to re-download and overwrite."
        )
        sys.exit(0)

    if args.force:
        logger.info("--force flag set. Overwriting existing cache.")

    try:
        tokenizer = manager.download_and_cache()
    except RuntimeError as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Tokenizer cached successfully. "
        "Vocabulary size: %d. "
        "Cache path: %s",
        tokenizer.vocab_size,
        manager.cache_dir,
    )
    logger.info(
        "Files saved: %s",
        [f.name for f in manager.cache_dir.iterdir()],
    )


if __name__ == "__main__":
    main()
