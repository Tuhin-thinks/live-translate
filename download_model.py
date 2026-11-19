"""
Download a faster-whisper model into ./models_v3 with logging enabled.

Usage:
    pip install faster-whisper huggingface_hub
    python download_model.py                                   # default: large-v3
    python download_model.py --model kotoba-tech/kotoba-whisper-v2.0-faster
    python download_model.py --verify                          # also load on GPU after download
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "models_v3"
LOG_FILE = CACHE_DIR / "download.log"
DEFAULT_MODEL = "Systran/faster-whisper-large-v3"


def setup_logging() -> logging.Logger:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_h.setFormatter(fmt)
    root.handlers = [console, file_h]

    # huggingface_hub internal logs (HTTP requests, file resolution, cache hits)
    from huggingface_hub.utils import logging as hf_logging
    from huggingface_hub.utils import enable_progress_bars
    hf_logging.set_verbosity_info()  # use set_verbosity_debug() for full HTTP traces
    enable_progress_bars()  # tqdm bars per file

    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    return logging.getLogger("downloader")


def download(model_id: str, log: logging.Logger) -> str:
    from huggingface_hub import snapshot_download

    # Keep all HF cache files inside models_v3 (not ~/.cache/huggingface)
    os.environ["HF_HOME"] = str(CACHE_DIR)

    log.info("Model      : %s", model_id)
    log.info("Cache dir  : %s", CACHE_DIR)
    log.info("Log file   : %s", LOG_FILE)

    start = time.time()
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ],
        max_workers=4,
    )
    elapsed = time.time() - start

    files = [p for p in Path(path).iterdir() if p.is_file() or p.is_symlink()]
    total_mb = sum(p.resolve().stat().st_size for p in files) / 1024**2
    for p in files:
        log.info("  %-28s %8.1f MB", p.name, p.resolve().stat().st_size / 1024**2)
    log.info("Downloaded %.1f MB in %.1fs -> %s", total_mb, elapsed, path)
    return path


def verify(path: str, log: logging.Logger) -> None:
    from faster_whisper import WhisperModel

    log.info("Loading model on GPU (int8_float16) to verify...")
    try:
        WhisperModel(path, device="cuda", compute_type="int8_float16")
        log.info("Model loaded OK on CUDA.")
    except Exception as e:
        log.warning("CUDA load failed (%s). Trying CPU int8...", e)
        WhisperModel(path, device="cpu", compute_type="int8")
        log.info("Model loaded OK on CPU.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id (CTranslate2 format)")
    ap.add_argument("--verify", action="store_true", help="Load the model after download")
    args = ap.parse_args()

    log = setup_logging()
    try:
        path = download(args.model, log)
        if args.verify:
            verify(path, log)
        print(f"\nMODEL_PATH={path}")
    except Exception:
        log.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
