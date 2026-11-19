import os
import queue
import re
import threading
import time
import traceback
import warnings
from collections import deque
from pathlib import Path

# Make the CUDA toolkit's cuBLAS DLLs loadable even when its bin folder is not
# on PATH (cuDNN already ships inside the ctranslate2 wheel). Must run before
# ctranslate2 is imported.
if os.name == "nt" and os.environ.get("CUDA_PATH"):
    _cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
    if _cuda_bin.is_dir():
        os.add_dll_directory(str(_cuda_bin))

import ctranslate2  # noqa: E402
import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from PySide6.QtCore import QThread, Signal  # noqa: E402

from segmenter import PhraseSegmenter  # noqa: E402
from speech import SileroStream  # noqa: E402

SAMPLE_RATE = 16000
# Whisper weights are stored inside the project instead of the user-wide HF cache.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Stock phrases Whisper invents on music/noise (learned from YouTube outros and
# subtitle credits). Plain "thank you" is deliberately absent: it is common real dialogue.
HALLUCINATION_PHRASES = {
    "thanks for watching",
    "thank you for watching",
    "thank you for watching the video",
    "thank you so much for watching",
    "thank you for listening",
    "please subscribe",
    "please subscribe to my channel",
    "subscribe to my channel",
    "see you in the next video",
    "see you next time",
    "subtitles by the amaraorg community",
    "translated by",
}

# Whisper's no_speech_prob alone is unreliable on short translated chunks (real
# speech can score 0.9), and the segmenter's VAD already guarantees speech, so
# it only counts when the decode itself is also low-confidence.
NO_SPEECH_SOFT = 0.5
LOW_LOGPROB = -1.0
MAX_COMPRESSION_RATIO = 2.4  # higher means repetitive text, i.e. a decoding loop


def _normalize(text):
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


def rejection_reason(segment):
    """Return why a Whisper segment looks hallucinated, or None if it looks real."""
    if segment.no_speech_prob > NO_SPEECH_SOFT and segment.avg_logprob < LOW_LOGPROB:
        return f"no_speech={segment.no_speech_prob:.2f}, logprob={segment.avg_logprob:.2f}"
    if segment.compression_ratio > MAX_COMPRESSION_RATIO:
        return f"repetitive (compression={segment.compression_ratio:.2f})"
    if _normalize(segment.text) in HALLUCINATION_PHRASES:
        return "stock phrase"
    return None


class TranslateWorker(QThread):
    """Captures system audio, cuts it into phrases at pauses, and translates each to English.

    Capture runs on its own thread in small blocks so no audio is lost while
    Whisper is busy. Phrases are cut at silent pauses and where speech gives way
    to non-speech sound (see ``PhraseSegmenter``) rather than on a fixed timer,
    so words are not split and music alone never reaches Whisper.
    """

    text_ready = Signal(str)
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        model_size="large-v3",
        language="ja",
        device=None,
        compute_type=None,
        min_seconds=1.0,
        max_seconds=6.0,
        pause_ms=250,
        max_pending=2,
    ):
        super().__init__()
        self.model_size = model_size
        self.language = language
        # Auto-detect: CUDA when a GPU is visible, otherwise CPU with int8 weights.
        has_cuda = ctranslate2.get_cuda_device_count() > 0
        self.device = device or ("cuda" if has_cuda else "cpu")
        self.compute_type = compute_type or ("int8_float16" if has_cuda else "int8")
        # One capture block = one Silero window (512 samples = 32 ms at 16 kHz).
        self.segmenter = PhraseSegmenter(
            vad=SileroStream(),
            sample_rate=SAMPLE_RATE,
            block_size=SileroStream.WINDOW,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            pause_ms=pause_ms,
        )
        # If translation falls behind, keep only the freshest phrases: stale
        # captions are worse than skipped ones.
        self._pending = deque(maxlen=max_pending)
        self._blocks = queue.Queue()
        self._running = threading.Event()
        self._capture_error = None

    def stop(self):
        self._running.clear()

    # -- capture thread -------------------------------------------------------

    def _capture_loop(self):
        try:
            # Imported here, not at module level: soundcard calls
            # CoInitializeEx(MULTITHREADED) on the importing thread, and if that is
            # the GUI thread, Qt's OleInitialize() fails with RPC_E_CHANGED_MODE.
            import soundcard as sc

            # Loopback reports harmless "data discontinuity" glitches. soundcard's
            # import installs an "always show" filter, so ours must come after it.
            warnings.filterwarnings("ignore", message="data discontinuity")

            spk = sc.default_speaker()
            mic = sc.get_microphone(spk.name, include_loopback=True)
            self.status.emit(f"Capturing: {spk.name}")
            frames = self.segmenter.block_size
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=frames) as rec:
                while self._running.is_set():
                    block = rec.record(numframes=frames)
                    self._blocks.put(block.reshape(-1).astype(np.float32))
        except Exception as exc:
            self._capture_error = exc
            self._running.clear()

    # -- worker thread ----------------------------------------------------------

    def run(self):
        try:
            self.status.emit(
                f"Loading model '{self.model_size}' on {self.device} "
                f"({self.compute_type}, cache: {MODELS_DIR})..."
            )
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(MODELS_DIR),
            )

            self._running.set()
            capture = threading.Thread(target=self._capture_loop, daemon=True)
            capture.start()
            self.status.emit("Listening...")

            while self._running.is_set():
                self._drain_blocks()
                if not self._pending:
                    continue
                chunk = self._pending.popleft()
                self._translate(model, chunk)

            capture.join(timeout=2)
            if self._capture_error is not None:
                raise self._capture_error
        except Exception as exc:  # surface any failure to the UI instead of dying silently
            self.error.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def _drain_blocks(self):
        """Feed every captured block to the segmenter; wait briefly if none are ready."""
        try:
            block = self._blocks.get(timeout=0.1)
        except queue.Empty:
            return
        while True:
            for chunk in self.segmenter.push(block):
                if len(self._pending) == self._pending.maxlen:
                    self.status.emit("Falling behind: dropped a stale phrase")
                self._pending.append(chunk)
            try:
                block = self._blocks.get_nowait()
            except queue.Empty:
                return

    def _translate(self, model, chunk):
        started = time.perf_counter()
        segments, _ = model.transcribe(
            chunk,
            language=self.language,
            task="translate",
            vad_filter=True,
            condition_on_previous_text=False,  # phrases are independent; avoids repetition loops
            without_timestamps=True,
        )
        kept = []
        for seg in segments:
            reason = rejection_reason(seg)
            if reason:
                self.status.emit(f"  dropped {seg.text.strip()!r}: {reason}")
            else:
                kept.append(seg.text.strip())
        text = " ".join(kept).strip()
        elapsed = time.perf_counter() - started
        audio_s = len(chunk) / SAMPLE_RATE
        self.status.emit(f"[{audio_s:.1f}s audio -> {elapsed:.2f}s] {text or '(nothing)'}")
        if text:
            self.text_ready.emit(text)
