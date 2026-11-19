# Architecture

```
┌────────────────── capture thread ──────────────────┐
│ soundcard loopback (default speaker)               │
│ 16 kHz mono, 512-sample blocks (32 ms) ──► queue   │
└────────────────────────────────────────────────────┘
                         │
┌──────────── worker thread (TranslateWorker) ───────┐
│ drain queue ──► PhraseSegmenter.push(block)        │
│                     │                              │
│                     ▼  finished phrase             │
│                pending deque (max 2, drops oldest) │
│                     │                              │
│                     ▼                              │
│     WhisperModel.transcribe(task="translate")      │
│                     │                              │
│                     ▼                              │
│     rejection_reason() hallucination filter        │
└─────────────────────┬──────────────────────────────┘
                      │ text_ready (Qt signal, queued)
┌─────────────── GUI thread ─────────────────────────┐
│ CaptionOverlay.set_caption()                       │
└────────────────────────────────────────────────────┘
```

There are three threads, so that no stage blocks another:

- The **capture thread** only records audio and puts it on a queue. It never waits for Whisper, so no audio is lost while a phrase is being translated.
- The **worker thread** (`QThread`) owns the model, the segmenter and the filter.
- The **GUI thread** runs the Qt event loop and the overlay. Text reaches it through a queued signal, so it is safe across threads.

## 1. Capture (`pipeline.py`, `_capture_loop`)

`soundcard.get_microphone(default_speaker.name, include_loopback=True)` opens a WASAPI loopback stream, which records exactly what the speakers play. Blocks are 512 samples at 16 kHz. That is the window size Silero VAD expects, so each block maps to exactly one VAD call.

`soundcard` is imported *inside* the capture thread. On import it calls `CoInitializeEx(COINIT_MULTITHREADED)` on the importing thread. If that were the GUI thread, Qt's `OleInitialize()` would fail with `RPC_E_CHANGED_MODE` (0x80010106).

## 2. Phrase segmentation (`segmenter.py`)

Each block gets two numbers:

- **Loudness**: the block's RMS.
- **Speech probability**: from `SileroStream` (`speech.py`). This is a streaming wrapper around the Silero VAD ONNX model bundled with faster-whisper. It keeps the LSTM state and the 64-sample context between calls; faster-whisper's own wrapper resets them on every call.

A block is a **gap** if it is *quiet* or *not speech*:

- **Quiet** means RMS below `max(0.003, 0.3 × recent peak)`. The peak decays slowly, so the threshold follows the playback volume.
- **Not speech** means a speech probability below 0.35.

Cut rules:

| Rule | When | Where it cuts |
|---|---|---|
| Natural | at least 1 s buffered, and the last ≥ 250 ms are a gap | the quietest block in that gap, a local minimum of loudness |
| Forced | 6 s buffered with no gap | the lowest smoothed `prob + loudness/peak` in the second half of the buffer; the rest carries over |

Guards:

- Before any speech is heard, audio is discarded except for a 200 ms pre-roll, so the first syllable isn't clipped.
- A chunk with less than 200 ms of confident speech (prob ≥ 0.5) is never emitted. This stops music, or a stray blip at the edge of a song, from reaching Whisper.

## 3. Translation (`pipeline.py`, `_translate`)

```python
model.transcribe(chunk, language="ja", task="translate", vad_filter=True,
                 condition_on_previous_text=False, without_timestamps=True)
```

- `condition_on_previous_text=False`: phrases are independent, which avoids the repetition loops Whisper falls into on short clips.
- Device auto-detection: CUDA with `int8_float16` if `ctranslate2` sees a GPU, otherwise CPU with `int8`.
- On Windows, `%CUDA_PATH%\bin` is added to the DLL search path before `ctranslate2` is imported, so cuBLAS from the CUDA toolkit is found. cuDNN ships inside the ctranslate2 wheel.

### Backpressure

Finished phrases go into a `deque(maxlen=2)`. If Whisper is slower than real time, the oldest waiting phrase is dropped and `Falling behind: dropped a stale phrase` is logged. A late caption is less useful than a skipped one.

## 4. Hallucination filter (`rejection_reason`)

A Whisper segment is dropped if any of these hold:

- `no_speech_prob > 0.5` **and** `avg_logprob < -1.0`, which is Whisper's own no-speech heuristic. `no_speech_prob` alone is not trusted, because real translated speech often scores above 0.8.
- `compression_ratio > 2.4`, meaning repetitive text from a decoding loop.
- The normalised text is a known stock phrase, such as *"thanks for watching"*, *"please subscribe"* or *"subtitles by the amara.org community"*.

## 5. Overlay (`overlay.py`)

A `QWidget` with these flags:

- `FramelessWindowHint`: no border.
- `WindowStaysOnTopHint`: always on top.
- `Tool`: no taskbar entry.
- `WindowTransparentForInput`: clicks pass through.

It also uses `WA_TranslucentBackground` and `WA_ShowWithoutActivating`, so it never steals focus. It paints a rounded, semi-transparent black box and keeps the last two phrases. It sits at the bottom centre of the primary screen's `availableGeometry()`, so it stays above the taskbar, and hides after 4 s without new text.

## 6. Shutdown (`main.py`)

Qt's event loop runs in C++, and Python only runs signal handlers between bytecodes. A no-op `QTimer` every 200 ms hands control back to Python, so the `SIGINT` (Ctrl+C) and `SIGBREAK` handlers can call `app.quit()`. `aboutToQuit` then stops the worker and waits up to 10 s for an in-flight translation.
